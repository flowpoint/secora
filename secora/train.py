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
from data import preprocess_split
from config import config
from infer import build_embedding_space, k_nearest_neighbors
from losses import contrastive_loss, mrr
from tracking import *

from SM3 import SM3

import logging


def train_shard(
        state_tracker,
        train_loader,
        config,
        writer,
        ):
    ''' trains the model for until the budget is exhausted
    '''

    rank = dist.get_rank()

    logger = logging.getLogger('train')


    optim = state_tracker['optim']
    scheduler = state_tracker['scheduler']
    training_progress = state_tracker['training_progress']
    scaler = state_tracker['scaler']
    model = state_tracker['model']

    model.train()

    grad_accum = config['grad_accum']
    shard_loss = torch.tensor([0.], device=rank, requires_grad=False)

    if rank == 0:
        bar = tqdm(total=len(train_loader)/config['shards']*dist.get_world_size(), unit=' batch', desc='train_shard', smoothing=0.03)

    try:
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'] .to(rank, non_blocking=True)
            token_type_ids = batch['token_type_ids'].to(rank, non_blocking=True)
            attention_mask = batch['attention_mask'].to(rank, non_blocking=True)

            loss = scaler.scale(
                    contrastive_loss(
                        model, 
                        input_ids, 
                        token_type_ids, 
                        attention_mask, 
                        config))

            shard_loss.add_(loss.detach())

            if rank == 0:
                bar.update(n=1*dist.get_world_size())

            # only sync before optimizer step
            if (step+1) % grad_accum != 0:
                with model.no_sync():
                    loss.backward()
            else:
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

                if rank == 0:
                    scheduler.step()
                    training_progress.optimizer_step += 1
                    writer.add_scalar("lr/train", scheduler.get_last_lr()[0], training_progress.optimizer_step)
                    writer.flush()

                optim.zero_grad(set_to_none=True)

            if step > len(train_loader) / config['shards']:
                break


    except Exception as e:
        logger = logging.getLogger('train')
        logger.exception(e)

    dist.all_reduce(shard_loss)
    dist.barrier()
    if rank == 0:
        avg_loss = shard_loss.cpu().numpy()/step
        writer.add_scalar("avg_loss/train", avg_loss, training_progress.optimizer_step)
        writer.flush()


def validate(
        model, 
        valid_loader, 
        config, 
        writer,
        training_progress):
    relevant_ids = range(len(valid_loader))

    if config['num_gpus'] == 0:
        device = 'cpu'
    elif config['num_gpus'] == 1:
        device = 'cuda'
    else:
        device = dist.get_rank()

    distances, neighbors, cosine_similarities = k_nearest_neighbors(
                model, 
                valid_loader, 
                embedding_size=config['embedding_size'], 
                top_k=config['top_k'], 
                config=config,
                device=device)

    neighbors_list = [list(n) for n in neighbors]
    score = mrr(list(relevant_ids), neighbors_list)

    i = training_progress.optimizer_step
    writer.add_scalar("mrr/validation", score, i)
    writer.add_scalar("distances/validation", np.mean(distances), i)
    writer.add_scalar("cosine_similarity/validation", np.mean(cosine_similarities), i)
    writer.flush()


def train(config):
    rank = dist.get_rank()

    make_logger(config)
    logger = logging.getLogger('train')

    writer = SummaryWriter(log_dir=os.path.join(config['logdir'], config['name']), flush_secs=30)

    #model = EmbeddingModel(config).to(rank)
    model = BiEmbeddingModel(config).to(rank)
    model = DDP(model, device_ids=[rank])
    #model = model.to(config['devices'][0])

    if config['finetune_mode'] == 'all':
        params = model.parameters()
    elif config['finetune_mode'] == 'pooling':
        params = model.pooling.parameters()
    else:
        raise RuntimeError('finetune_mode has to be: all or pooling')

    train_set = preprocess_split('train', config)
    valid_set = preprocess_split('validation', config)
    # scroll screen

    sampler = DistributedSampler(train_set)
    train_loader = DataLoader(
            train_set, 
            batch_size=config['batch_size'], 
            shuffle=(sampler is None),
            drop_last=True, 
            pin_memory=True, 
            # workers need to use the spawn or forkserver method in a distributed setting
            num_workers=1, 
            multiprocessing_context='spawn',
            persistent_workers=True, 
            sampler=sampler

            )


    # don't shuffle validation set!
    valid_loader = DataLoader(
            valid_set, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            drop_last=True, 
            pin_memory=True, 
            # workers need to use the spawn or forkserver method in a distributed setting
            num_workers=1, 
            multiprocessing_context='spawn',
            persistent_workers=True)

    print("\n"*8)

    if config['optim'] == 'adam':
        optim = torch.optim.Adam(params, lr=config['lr'])
    elif config['optim'] == 'sgd':
        optim = torch.optim.SGD(params, lr=config['lr'])
    elif config['optim'] == 'sm3':
        optim = SM3(params, lr=config['lr'])
    else:
        raise RuntimeError('config specifies an unsupported optimizer')

    num_warmup_steps = ceil(config['warmup_batches'] / config['grad_accum'])
    num_training_steps_per_epoch = ceil(len(train_loader) / (config['batch_size'] * config['grad_accum']))
    num_training_steps = config['epochs'] * num_training_steps_per_epoch

    scheduler = get_linear_schedule_with_warmup(
            optim,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps)

    scaler = GradScaler()

    training_progress = TrainingProgress()
    state_tracker = StateTracker(
            config,
            model=model,
            optim=optim,
            scheduler=scheduler,
            scaler=scaler,
            training_progress=training_progress)

    # load latest checkpoint 
    dist.barrier()

    state_tracker.load_latest()

    if config['run_type'] == 'debug':
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
    logger = logging.getLogger('train')

    if rank != 0:
        logger = UnLogger()

    logger.info(f'shard_size: {shard_size} samples')
    logger.info(f'validation set size: {val_size} samples')

    rank = dist.get_rank()

    try:
        # training
        while(training_progress.epoch < num_epochs):
            logger.info(f'starting epoch: {training_progress.epoch} of {num_epochs}')

            while(training_progress.shard < num_shards):
                logger.info('training shard')

                train_shard(
                    state_tracker,
                    train_loader,
                    config,
                    writer,
                    )

                dist.barrier()
                if config['run_type'] == 'default' and rank == 0:
                    state_tracker.save()


                if rank == 0:
                    logger.info('validating shard')

                    '''
                    validate(state_tracker['model'], 
                            valid_loader, 
                            config, 
                            writer, 
                            state_tracker['training_progress'])
                    '''

                if rank == 0:
                    training_progress.shard += 1
            if rank == 0:
                training_progress.epoch += 1
                training_progress.shard = 0

    except KeyboardInterrupt as e:
        state_tracker.save()
        logger.info('training interrupted')
    except Exception as e:
        logger = logging.getLogger('train')
        logger.exception(e)
        dist.destroy_process_group()

    logger.info("training finished")
    dist.barrier(group=dist.group.WORLD)
    dist.destroy_process_group()

class UnLogger:
    def info(self, x):
        pass

def training_worker(rank, config):
    world_size = config['num_gpus']

    host_name = config['hostname']
    port = config['port']

    os.environ['MASTER_ADDR'] = host_name
    os.environ['MASTER_PORT'] = str(port)

    #client_store = dist.TCPStore(host_name, port, world_size, is_master=False)

    # initialize the process group
    # TORCH_DISTRIBUTED_DEBUG=DETAIL
    # export NCCL_SOCKET_IFNAME=eno1
    # export NCCL_DEBUG_SUBSYS=ALL
    # export NCCL_DEBUG=INFO
    # export NCCL_IB_DISABLE=1

    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    train(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual training script.')
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    logdir = os.path.join(config['logdir'], config['name'])
    checkdir = os.path.join(config['checkpoint_dir'], config['name'])
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(checkdir, exist_ok=True)


    np.random.seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.cuda.empty_cache()

    torch.backends.cudnn.benchmark = True

    #server_store = dist.TCPStore(host_name, port, world_size, is_master=True)

    torch.use_deterministic_algorithms(True)

    mp.set_start_method('spawn')
    mp.spawn(training_worker, 
            args=(config,),
            nprocs = config['num_gpus'],
            join=True)

#neighsamples = valid_set_tokenized.select(neighbors.flatten())['proc_url']
#for dist, s, rid in zip(distances.flatten(), neighsamples)
#for k in range(top_k)
#valid_set.select(neighbors)
