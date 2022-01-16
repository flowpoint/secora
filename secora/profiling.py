import os
import sys
from math import ceil
import argparse

import pdb

import numpy as np

import torch
import torch.nn.functional as F

import torch
from torch import profiler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import BiEmbeddingModel
from data import preprocess_split
from config import load_config
from losses import contrastive_loss, mrr
from tracking import *
from infer import *

from SM3 import SM3


def train_step(model, optim, batch, config, device='cpu'):
    input_ids = batch['input_ids'].to(device)
    token_type_ids = batch['token_type_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    # a step without syncing, simulates grad accum
    loss = contrastive_loss(model, input_ids, token_type_ids, attention_mask, config)

    with model.no_sync():
        loss.backward()

    # a step without syncing, simulates grad accums optimization step
    loss = contrastive_loss(model, input_ids, token_type_ids, attention_mask, config)
    loss.backward()

    optim.step()
    optim.zero_grad(set_to_none=True)


def profile(config, logger, modes=['train','validation', 'embedding']):
    rank = dist.get_rank()

    model = BiEmbeddingModel(config).to(rank)
    model = DDP(model, device_ids=[rank])

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

    if 'train' in modes:
        train_set = preprocess_split('train', config).select(range(config['batch_size']*5))

        train_sampler = DistributedSampler(train_set, drop_last=True)
        train_loader = DataLoader(
                train_set, 
                batch_size=config['batch_size'], 
                shuffle=(train_sampler is None),
                drop_last=True, 
                pin_memory=True, 
                # workers need to use the spawn or forkserver method in a distributed setting
                num_workers=1, 
                multiprocessing_context='spawn',
                persistent_workers=True, 
                sampler=train_sampler)

        train_iter = iter(train_loader)


    if 'validation' in modes or 'embedding' in modes:
        valid_set = preprocess_split('validation', config).select(range(config['batch_size']*5))

        valid_sampler = DistributedSampler(valid_set, drop_last=True, shuffle=False)
        valid_loader = DataLoader(
                valid_set, 
                batch_size=config['batch_size'], 
                shuffle=(valid_sampler is None), 
                drop_last=True, 
                pin_memory=True, 
                # workers need to use the spawn or forkserver method in a distributed setting
                num_workers=1, 
                multiprocessing_context='spawn',
                persistent_workers=True,
                sampler=valid_sampler)


    logger.info('starting profiler')

    tensorboard_run_path = os.path.join(config['logdir'], config['name'])
    trace_path = os.path.join(config['logdir'], config['name'], 'profile_trace.json')
    stacks_path = os.path.join(config['logdir'], config['name'], 'profile_stacks.txt')


    #with profiler.profile()
    with profiler.profile(
        with_stack=True, 
        #profile_memory=True, 
        record_shapes=True,
        on_trace_ready=profiler.tensorboard_trace_handler(tensorboard_run_path),
        schedule=torch.profiler.schedule(
            skip_first=1,
            wait=0,
            warmup=1,
            active=2),
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ]) as p:

        for idx in range(5):
            torch.cuda.synchronize()
            dist.barrier()
            logger.info(f'step {idx}')
            if 'train' in modes:
                model.train()
                logger.info(f'train_step')
                train_step(model, optim, next(train_iter), config, device=rank)


            dist.barrier()
            torch.cuda.synchronize()
            if 'embedding' in modes:
                code_embedding = build_embedding_space(model, valid_loader, config, feature_prefix='code_', embedding_size=config['embedding_size'], device=rank)
                doc_embedding = build_embedding_space(model, valid_loader, config, feature_prefix='doc_', embedding_size=config['embedding_size'], device=rank)

            torch.cuda.synchronize()
            dist.barrier()
            if 'validation' in modes:
                with model.no_sync():
                    with torch.no_grad():
                        model.eval()
                        logger.info(f'eval step')
                        distances, neighbors = k_nearest_neighbors(
                                doc_embedding,
                                code_embedding,
                                embedding_size=config['embedding_size'], 
                                top_k=config['top_k'])
            p.step()

    if rank == 0:
        print(p.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1))
        p.export_stacks(stacks_path, "self_cuda_time_total")


def profiling_worker(rank, config, modes):
    world_size = config['num_gpus']
    host_name = config['hostname']
    port = config['port']

    os.environ['MASTER_ADDR'] = host_name
    os.environ['MASTER_PORT'] = str(port)

    # initialize the process group
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    logger = make_logger(config, rank=rank)
    logger.info('start profiling worker')

    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    profile(config, logger, modes)


def copytest():
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    x = torch.rand([100000,128], device=rank)

    x2 = [torch.zeros_like(x, dtype=torch.float32, device=rank)] * world_size
    dist.all_gather(x2, x)

    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        doc_cpu = [x.cpu().numpy() for x in all_doc_embeddings]
        full_doc_embedding_space = np.concatenate(doc_cpu, 0).astype(np.float32)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual profiling script.')
    parser.add_argument('config_path', type=str)
    parser.add_argument('--modes', type=str, action='append', default=[])
    parser.add_argument('--run_name', type=str, default='')
    args = parser.parse_args()

    if args.modes == []:
        raise RuntimeError('--modes have to be at least one')

    config = load_config(args.config_path)
    if args.run_name != '':
        config['name'] = args.run_name

    modes = args.modes

    logdir = os.path.join(config['logdir'], config['name'])
    checkdir = os.path.join(config['checkpoint_dir'], config['name'])
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(checkdir, exist_ok=True)

    logger = make_logger(config, log_all_ranks=True, rank=-1)

    logger.info(f'logdir: {logdir}')
    logger.info(f'checkdir: {checkdir}')

    np.random.seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.cuda.empty_cache()

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = True

    mp.set_start_method('spawn')
    mp.spawn(profiling_worker, 
            args=(config, modes),
            nprocs = config['num_gpus'],
            join=True)
