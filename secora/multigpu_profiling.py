import os
import sys
from math import ceil
import argparse

import pdb

import torch
import torch.nn.functional as F

import torch
from torch import profiler
from torch.utils.data import DataLoader

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from model import EmbeddingModel
from data import preprocess_split
from config import config
from losses import contrastive_loss, mrr
from tracking import *

from SM3 import SM3


def test_step(model, optim, batch, config, device='cuda'):
    input_ids = batch['input_ids'].to(device, non_blocking=True)
    token_type_ids = batch['token_type_ids'].to(device, non_blocking=True)
    attention_mask = batch['attention_mask'].to(device, non_blocking=True)

    loss = contrastive_loss(model, input_ids, token_type_ids, attention_mask, config)
    loss.backward()

    s_loss = loss.detach().cpu().numpy()

    optim.step()
    optim.zero_grad(set_to_none=True)


def profile(config):
    model = EmbeddingModel(config)
    model = model.to(dist.get_rank())

    if config['finetune_mode'] == 'all':
        params = model.parameters()
    elif config['finetune_mode'] == 'pooling':
        params = model.pooling.parameters()
    else:
        raise RuntimeError('finetune_mode has to be: all or pooling')

    if config['optim'] == 'adam':
        optim = torch.optim.Adam(params, lr=config['lr'])
    elif config['optim'] == 'sgd':
        optim = torch.optim.SGD(params, lr=config['lr'])
    elif config['optim'] == 'sm3':
        optim = SM3(params, lr=config['lr'])
    else:
        raise RuntimeError('config specifies and unsupported optimizer')

    train_set = preprocess_split('train', config)

    sampler = DistributedSampler(train_set)
    train_loader = DataLoader(
            train_set, 
            batch_size=config['batch_size'], 
            shuffle=(sampler is None),
            drop_last=True, 
            pin_memory=True, 
            num_workers=4, 
            persistent_workers=True, 
            sampler=sampler
            )

    it = iter(train_loader)

    tensorboard_run_path = os.path.join(config['logdir'], config['name'])
    trace_path = os.path.join(config['logdir'], config['name'], 'profile_trace.json')
    stacks_path = os.path.join(config['logdir'], config['name'], 'profile_stacks.txt')

    with profiler.profile(
            with_stack=True, 
            profile_memory=True, 
            record_shapes=True,
            on_trace_ready=profiler.tensorboard_trace_handler(tensorboard_run_path),
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=2),
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ]) as p:
        for batch in range(8):
            test_step(model, optim, next(it), config, device=dist.get_rank())
            p.step()

    p.export_stacks(stacks_path, "self_cuda_time_total")
    print(p.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))


def profiling_worker(rank, config):
    world_size = config['num_gpus']

    host_name = config['hostname']
    port = config['port']

    os.environ['MASTER_ADDR'] = host_name
    os.environ['MASTER_PORT'] = str(port)

    #client_store = dist.TCPStore(host_name, port, world_size, is_master=False)

    # initialize the process group
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    profile(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual profiling script.')
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    logdir = os.path.join(config['logdir'], config['name'])
    checkdir = os.path.join(config['checkpoint_dir'], config['name'])
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(checkdir, exist_ok=True)

    np.random.seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.cuda.empty_cache()

    mp.spawn(profiling_worker, 
            args=(config,),
            nprocs = config['num_gpus'],
            join=True)

