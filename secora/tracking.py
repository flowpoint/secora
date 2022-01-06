from math import ceil
from time import time
import os
from dataclasses import dataclass

import torch


class StateTracker:
    def __init__(self, config, **kwargs):
        self.config = config
        if 'checkpoint_dir' not in config.keys():
            raise RuntimeError('config needs to have a checkpoint_dir')
        if 'max_checkpoints' not in config.keys():
            raise RuntimeError('config needs to have a max_checkpoints')
        if not config['max_checkpoints'] >= 1:
            raise RuntimeError('max_checkpoints has to be greater than 1')

        self.checkpoint_dir = config['checkpoint_dir']
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
        timestamp = ceil(time())
        path = os.path.join(self.checkpoint_dir, "checkpoint_" + str(timestamp) + ".pt")
        state = [o.state_dict() for o in self.objects.values()]
        torch.save(state, path)

        # remove oldest checkpoint, through lexicographic order
        checkpoints = self._list_checkpoints()

        if len(checkpoints) > self.max_checkpoints:
            old_path = os.path.join(self.checkpoint_dir, sorted(checkpoints)[0])
            os.remove(old_path)

    def load_latest(self):
        existing_checkpoints = sorted(self._list_checkpoints())
        if len(existing_checkpoints) < 1:
            print('no checkpoints available, not loading anything')
            return

        checkpoint = existing_checkpoints[-1]
        path = os.path.join(self.checkpoint_dir, checkpoint)

        if not os.path.isfile(path):
            raise RuntimeError(f"checkpoint {path} doesn't exist")

        restored_state = torch.load(path)
        for o,p in zip(self.objects.values(), restored_state):
            o.load_state_dict(p)

        return self.objects.values()


@dataclass
class TrainingProgress:
    ''' epochs and shards that have been begun'''
    # epoch -1 means warmup
    epoch: int = -1
    shard: int = 0 
    optimizer_step: int = -1

    def state_dict(self,):
        return { "epoch": self.epoch, 
            "shard": self.shard,
            "optimizer_step": self.optimizer_step
            }

    def load_state_dict(self, state):
        self.epoch = state['epoch']
        self.shard = state['shard']
        self.optimizer_step = state['optimizer_step']
