import os
from math import ceil
from time import time
import datetime
import re
import argparse

import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from secora.config import *
from secora.training_tasks import *

TIMEOUT = datetime.timedelta(65)

def training_worker(rank, config, progress, debug):
    world_size = config['num_gpus']
    dist.init_process_group('nccl', rank=rank, world_size=world_size, timeout=TIMEOUT)
    make_logger(config, debug=debug, rank=rank)
    logger = logging.getLogger('secora')

    torch.cuda.set_device(rank)
    train(config, progress=progress, debug=debug, logger=logger)

    torch.cuda.synchronize()
    dist.barrier(group=dist.group.WORLD)
    dist.destroy_process_group()


def clean_init(seed):
    torch.cuda.empty_cache()
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual training script.')
    parser.add_argument('config_file', type=argparse.FileType('r'))
    # these values override the config values if specified
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--max_checkpoints', type=int, default=None) 
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--progress', action='store_true', default=False)
    args = parser.parse_args()

    config = TrainingConfig()

    with args.config_file as f:
        config_candidate = yaml.safe_load(f)

    timestamp = str(ceil(time()))
    if args.name is not None:
        if args.debug == True:
            prefix = 'debug'
        else: 
            prefix = 'run'
            
        config_candidate['name'] = f'{prefix}_{args.name}_t0_utc{timestamp}'

    if args.batch_size is not None:
        config_candidate['batch_size'] = args.batch_size

    if args.max_checkpoints is not None:
        config_candidate['max_checkpoints'] = args.max_checkpoints

    config.parse_from_dict(config_candidate)
    config.check()
    config.final()

    logdir = os.path.join(config['logdir'], config['name'])
    checkdir = os.path.join(config['checkpoint_dir'], config['name'])
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(checkdir, exist_ok=True)

    clean_init(seed=config['seed'])
    torch.backends.cudnn.benchmark = True

    mp.set_start_method('spawn')
    mp.spawn(training_worker, 
            args=(config, args.progress, args.debug),
            nprocs = config['num_gpus'],
            join=True)
