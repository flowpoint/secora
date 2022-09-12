from math import ceil
from time import time
import os
from dataclasses import dataclass
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import yaml

def init_storage(config, rank):
    if rank == 0:
        logdir = os.path.join(config['logdir'], config['training_run_id'])
        checkdir = logdir #os.path.join(config['checkpoint_dir'], config['name'])
        os.makedirs(logdir, exist_ok=True)
        os.makedirs(checkdir, exist_ok=True)

        path = os.path.join(config['logdir'], config['training_run_id'], 'config.yml')
        with open(path, 'w') as f:
            f.write(yaml.dump(config.to_dict()))


def make_logger(config, debug=False, rank=-1):
    if debug == True:
        level = logging.DEBUG
    else: 
        level = logging.INFO

    logger = logging.getLogger('secora')
    logger.setLevel(level)

    logdir = os.path.join(config['logdir'], config['training_run_id'])
    checkdir = os.path.join(config['checkpoint_dir'], config['training_run_id'])

    os.makedirs(logdir, exist_ok=True)
    os.makedirs(checkdir, exist_ok=True)

    path = os.path.join(config['logdir'], config['training_run_id'], f'run_rank{rank}.log')

    fh = logging.FileHandler(path)
    ch = logging.StreamHandler()

    rank_str = f'rank_{rank}'

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - ' + rank_str + ' - %(message)s')

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    if logger.hasHandlers() == False:
        logger.addHandler(fh)
        if debug == True:
            logger.addHandler(ch)

    return logger


class StateTracker:
    def __init__(self, name, logdir, max_checkpoints, logger, **kwargs):
        if not max_checkpoints >= 0:
            raise RuntimeError('max_checkpoints has to be positive')

        self.run_logdir = os.path.join(logdir, name)
        os.makedirs(self.run_logdir, exist_ok=True)
        self.max_checkpoints = max_checkpoints

        self.logger = logger

        if self.max_checkpoints == 0:
            self.logger.warning("max_checkpoints is 0, no checkpoints will be saved")
        if self.max_checkpoints < 0:
            raise ValueError('invalid value for max_checkpoints')

        for k, o in kwargs.items():
            if "state_dict" not in dir(o):
                raise RuntimeError(f'tracked object: {o} need to have a state_dict() method')
            if "load_state_dict" not in dir(o):
                raise RuntimeError(f'tracked object: {o} need to have a load_state_dict() method')

        self.objects = kwargs

    def __getitem__(self,i):
        return self.objects[i]

    def _list_checkpoints(self):
        return list(filter(lambda x: x.startswith("checkpoint"), os.listdir(self.run_logdir)))

    def save(self):
        if self.max_checkpoints == 0:
            self.logger.info(f'skipping saving, as max_checkpoints is 0')
            return 

        timestamp = ceil(time())
        path = os.path.join(self.run_logdir, "checkpoint_" + str(timestamp) + ".pt")
        state = [o.state_dict() for o in self.objects.values()]

        self.logger.info(f'saving state to {path}')
        torch.save(state, path)

        # remove oldest checkpoint, through lexicographic order
        checkpoints = self._list_checkpoints()

        if len(checkpoints) > self.max_checkpoints:
            old_path = os.path.join(self.run_logdir, sorted(checkpoints)[0])
            self.logger.info(f'removing old checkpoint: {old_path}')
            os.remove(old_path)

    def load_latest(self):
        self.logger = logging.getLogger(__name__)

        existing_checkpoints = sorted(self._list_checkpoints())
        if len(existing_checkpoints) < 1:
            self.logger.info('no checkpoints available, not loading anything')
            return 

        self.logger.info(f'found checkpoints in {self.run_logdir}: {existing_checkpoints}')

        checkpoint = existing_checkpoints[-1]
        path = os.path.join(self.run_logdir, checkpoint)

        self.logger.info(f'restoring: {path}')

        if not os.path.isfile(path):
            raise RuntimeError(f"checkpoint {path} doesn't exist")

        restored_state = torch.load(path)
        for (objkey, obj), rs in zip(self.objects.items(), restored_state):
            obj.load_state_dict(rs)


@dataclass
class TrainingProgress:
    ''' epochs and shards that have been begun'''
    # beware that this is for tracking and resuming training progress
    # not for tracking metrics
    # e.g. the total trained samples are a metric
    # total samples dont match epochs*shard*batch_size because samples in incomplete batches are dropped

    # epoch -1 means warmup
    epoch: int = 0
    shard: int = 0 
    step: int = 0 

    def state_dict(self,):
        return { "epoch": self.epoch, 
            "shard": self.shard,
            "step": self.step
            }

    def load_state_dict(self, state):
        self.epoch = state['epoch']
        self.shard = state['shard']
        self.optimizer_step = state['step']

    def epoch_done(self):
        self.shard = 0
        self.epoch += 1

    def shard_done(self):
        self.step = 0
        self.shard += 1

    def step_done(self):
        self.step += 1
