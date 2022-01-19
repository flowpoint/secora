import os
import sys

import time
from time import time

from math import ceil
from dataclasses import dataclass
import argparse

import pdb

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import get_linear_schedule_with_warmup

from model import *
from data import *
from config import load_config, overwrite_config
from infer import build_embedding_space, k_nearest_neighbors, validate
from losses import contrastive_loss, mrr
from tracking import *

from SM3 import SM3

import logging


def train_shard(
        state_tracker,
        train_loader,
        config,
        writer,
        **kwargs
        ):

    rank = dist.get_rank()

    logger = kwargs['logger']

    optim = state_tracker['optim']
    scheduler = state_tracker['scheduler']
    training_progress = state_tracker['training_progress']
    scaler = state_tracker['scaler']
    model = state_tracker['model']

    model.train()

    grad_accum = config['grad_accum']
    shard_loss = torch.tensor([0.], device=rank, requires_grad=False)

    if rank == 0 and kwargs['progress'] == True:
        bar = tqdm(
                total=len(train_loader), 
                unit=' batch', 
                desc='train_shard', 
                smoothing=0.03)

    heartbeat = time()

    try:
        for step, batch in enumerate(deviceloader(train_loader, rank)):
            model_inputs = batch['input_ids'], batch['token_type_ids'], batch['attention_mask']

            loss = scaler.scale(
                    contrastive_loss(
                        model, 
                        model_inputs,
                        config))

            shard_loss.add_(loss.detach())

            if rank == 0 and kwargs['progress'] == True:
                bar.update(n=1)

            if rank == 0 and time() - heartbeat > 10:
                logger.info(f"heartbeat: training: epoch: {training_progress.epoch} shard: {training_progress.shard} step: {step}/{len(train_loader)}")
                heartbeat = time()

            # only sync before optimizer step
            if (step+1) % grad_accum != 0:
                logger.debug('unsynced loss.backward')
                with model.no_sync():
                    loss.backward()
            else:
                logger.debug('optimization step')
                loss.backward()

                #gradient clipping
                if 'grad_clip' in config and config['grad_clip'] is not None:
                    if config['grad_clip'] == 0:
                        raise ValueError('grad_clip cant be 0')
                    scaler.unscale_(optim)
                    clip_grad_norm_(
                            model.parameters(), 
                            max_norm=config['grad_clip'])

                if config['precision'] == 'mixed':
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()

                scheduler.step()
                training_progress.optimizer_step += 1
                if rank == 0:
                    writer.add_scalar("lr/train", scheduler.get_last_lr()[0], training_progress.optimizer_step)
                    writer.flush()

                optim.zero_grad(set_to_none=True)


    except Exception as e:
        logger.exception(e)

    dist.all_reduce(shard_loss)
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        avg_loss = shard_loss.cpu().numpy()/step
        writer.add_scalar("avg_loss/train", avg_loss, training_progress.optimizer_step)
        writer.flush()


def train(config, **kwargs):
    rank = dist.get_rank()
    logger = make_logger(config, debug=kwargs['debug'], rank=rank)
    writer = SummaryWriter(log_dir=os.path.join(config['logdir'], config['name']), flush_secs=30)

    if kwargs['debug'] == True:
        limit = 2*config['grad_accum']*config['batch_size']
        train_set = preprocess_split('train', config, limit_samples=limit, **kwargs)
        valid_set = preprocess_split('validation', config, limit_samples=limit, **kwargs)
    else:
        train_set = preprocess_split('train', config, **kwargs)
        valid_set = preprocess_split('validation', config, **kwargs)

    # both sets arent shuffled, shuffle train set every epoch manually
    train_loader = get_loader(train_set, config)
    valid_loader = get_loader(valid_set, config, **kwargs)

    logger.info('building model')
    m = BiEmbeddingModel(config).to(rank)

    logger.info('warming up cuda benchmark')
    for step, batch in zip(range(12), deviceloader(train_loader, rank)):
        model_inputs = batch['input_ids'], batch['token_type_ids'], batch['attention_mask']
        m(*model_inputs)

    if config['cuda_graphs'] == True:
        logger.info('cuda_graphs is True: building the cuda graph')
        m.make_graphed(model_inputs)

    logger.info('building DDP model')
    model = DDP(m, device_ids=[rank])

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
        raise RuntimeError('config specifies an unsupported optimizer')

    num_warmup_steps = ceil(config['warmup_batches'] / config['grad_accum'])
    num_training_steps_per_epoch = ceil(len(train_set) / (config['batch_size'] * config['grad_accum']))
    num_training_steps = config['epochs'] * num_training_steps_per_epoch

    scheduler = get_linear_schedule_with_warmup(
            optim,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps)

    scaler = GradScaler()

    training_progress = TrainingProgress()
    state_tracker = StateTracker(
            config,
            logger,
            model=model,
            optim=optim,
            scheduler=scheduler,
            scaler=scaler,
            training_progress=training_progress)

    # load latest checkpoint 
    torch.cuda.synchronize()
    dist.barrier()
    state_tracker.load_latest()

    if kwargs['debug'] == True:
        num_epochs = 2
        num_shards = 2
        training_progress.epoch = 0
        training_progress.shard = 0
        config['grad_accum'] = 1
    else:
        num_epochs = config['epochs']
        num_shards = config['shards']

    shard_size = len(train_set)/num_shards
    val_size = len(valid_set)

    logger.info(f'shard_size: {shard_size} samples')
    logger.info(f'validation set size: {val_size} samples')

    rank = dist.get_rank()

    try:
        # training
        while(training_progress.epoch < num_epochs):
            logger.info(f'starting epoch: {training_progress.epoch} of {num_epochs}')
            train_set.shuffle()

            while(training_progress.shard < num_shards):
                logger.info(f'training shard: {training_progress.shard}')

                shard = train_set.shard(num_shards, training_progress.shard, contiguous=True)
                train_loader = get_loader(shard, config)

                train_shard(
                    state_tracker,
                    train_loader,
                    config,
                    writer,
                    logger=logger,
                    **kwargs
                    )

                torch.cuda.synchronize()
                dist.barrier()
                if rank == 0:
                    state_tracker.save()

                logger.info(f'validating shard {training_progress.shard}')

                validate(state_tracker['model'], 
                        valid_loader, 
                        config, 
                        writer, 
                        state_tracker['training_progress'],
                        logger=logger,
                        **kwargs)

                training_progress.shard += 1
            training_progress.epoch += 1
            training_progress.shard = 0

        torch.cuda.synchronize()
        dist.barrier(group=dist.group.WORLD)

    except KeyboardInterrupt as e:
        if rank == 0:
            state_tracker.save()
        logger.info('training interrupted')
    except Exception as e:
        if kwargs['debug'] == True:
            pdb.post_mortem()
        logger.exception(e)
    finally:
        dist.destroy_process_group()


def training_worker(rank, config, progress, debug):
    world_size = config['num_gpus']
    host_name = config['hostname']
    port = config['port']

    os.environ['MASTER_ADDR'] = host_name
    os.environ['MASTER_PORT'] = str(port)
    
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    
    # initialize the process group
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    train(config, progress=progress, debug=debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual training script.')
    parser.add_argument('config_path', type=str)

    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--progress', action='store_true', default=False)
    args = parser.parse_args()

    config = load_config(args.config_path)
    config = overwrite_config(args, config)

    logdir = os.path.join(config['logdir'], config['name'])
    checkdir = os.path.join(config['checkpoint_dir'], config['name'])
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(checkdir, exist_ok=True)

    logger = make_logger(config, rank=-1, debug=args.debug)
    logger.info(f'logdir: {logdir}')
    logger.info(f'checkdir: {checkdir}')

    np.random.seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.cuda.empty_cache()

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = True

    mp.set_start_method('spawn')
    mp.spawn(training_worker, 
            args=(config, args.progress, args.debug),
            nprocs = config['num_gpus'],
            join=True)

#neighsamples = valid_set_tokenized.select(neighbors.flatten())['proc_url']
#for dist, s, rid in zip(distances.flatten(), neighsamples)
#for k in range(top_k)
#valid_set.select(neighbors)
