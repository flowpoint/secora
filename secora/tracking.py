from math import ceil
from time import time
import os
from dataclasses import dataclass
import logging

import torch
import torch.distributed as dist

def make_logger(config, log_all_ranks=False, rank=-1):
    if config['debug'] == True:
        log_all_ranks = True
    else: 
        level = logging.INFO

    level = logging.DEBUG

    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    logdir = os.path.join(config['logdir'], config['name'])
    checkdir = os.path.join(config['checkpoint_dir'], config['name'])
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(checkdir, exist_ok=True)

    path = os.path.join(config['logdir'], config['name'], 'run.log')

    fh = logging.FileHandler(path)
    ch = logging.StreamHandler()

    rank_str = f'rank_{rank}'

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - ' + rank_str + ' - %(message)s')

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


class StateTracker:
    def __init__(self, config, **kwargs):
        self.config = config

        if 'checkpoint_dir' not in config.keys():
            raise RuntimeError('config needs to have a checkpoint_dir')
        if 'max_checkpoints' not in config.keys():
            raise RuntimeError('config needs to have a max_checkpoints')
        if not config['max_checkpoints'] >= 1:
            raise RuntimeError('max_checkpoints has to be greater than 1')

        self.checkpoint_dir = os.path.join(config['checkpoint_dir'], config['name'])
        self.max_checkpoints = self.config['max_checkpoints']

        for o in kwargs.values():
            if "state_dict" not in dir(o):
                raise RuntimeError(f'tracked object: {o} need to have a state_dict() method')
            if "load_state_dict" not in dir(o):
                raise RuntimeError(f'tracked object: {o} need to have a load_state_dict() method')

        self.objects = kwargs

    def __getitem__(self,i):
        return self.objects[i]

    def _list_checkpoints(self):
        return list(filter(lambda x: x.startswith("checkpoint"), os.listdir(self.checkpoint_dir)))

    def save(self):
        logger = logging.getLogger(__name__)

        timestamp = ceil(time())
        path = os.path.join(self.checkpoint_dir, "checkpoint_" + str(timestamp) + ".pt")
        state = [o.state_dict() for o in self.objects.values()]

        logger.info(f'saving state to {path}')
        torch.save(state, path)

        # remove oldest checkpoint, through lexicographic order
        checkpoints = self._list_checkpoints()

        if len(checkpoints) > self.max_checkpoints:
            old_path = os.path.join(self.checkpoint_dir, sorted(checkpoints)[0])
            logger.info(f'removing old checkpoint: {old_path}')
            os.remove(old_path)

    def load_latest(self):
        logger = logging.getLogger(__name__)

        existing_checkpoints = sorted(self._list_checkpoints())
        if len(existing_checkpoints) < 1:
            logger.info('no checkpoints available, not loading anything')
            return 

        logger.info(f'found checkpoints in {self.checkpoint_dir}: {existing_checkpoints}')

        checkpoint = existing_checkpoints[-1]
        path = os.path.join(self.checkpoint_dir, checkpoint)

        logger.info(f'restoring: {path}')

        if not os.path.isfile(path):
            raise RuntimeError(f"checkpoint {path} doesn't exist")

        restored_state = torch.load(path)
        for (objkey, obj), rs in zip(self.objects.items(), restored_state):
            obj.load_state_dict(rs)


@dataclass
class TrainingProgress:
    ''' epochs and shards that have been begun'''
    # epoch -1 means warmup
    epoch: int = 0
    shard: int = 0 
    optimizer_step: int = 0 

    def state_dict(self,):
        return { "epoch": self.epoch, 
            "shard": self.shard,
            "optimizer_step": self.optimizer_step
            }

    def load_state_dict(self, state):
        self.epoch = state['epoch']
        self.shard = state['shard']
        self.optimizer_step = state['optimizer_step']
