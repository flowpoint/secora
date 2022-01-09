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
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler

from transformers import get_linear_schedule_with_warmup

from model import *
from data import preprocess_split
from config import config
from infer import build_embedding_space, k_nearest_neighbors
from profiling import profile
from losses import contrastive_loss, mrr
from tracking import *

from SM3 import SM3

import logging

def train_shard(
        state_tracker,
        train_set,
        config,
        writer,
        ):
    ''' trains the model for until the budget is exhausted
    '''

    logger = logging.getLogger('train')

    model = state_tracker['model']
    optim = state_tracker['optim']
    scheduler = state_tracker['scheduler']
    training_progress = state_tracker['training_progress']
    scaler = state_tracker['scaler']

    model.train()

    grad_accum = config['grad_accum']
    shard_loss = torch.tensor([0.], device=config['device'], requires_grad=False)

    train_loader = DataLoader(
            train_set, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            drop_last=True, 
            pin_memory=True, 
            num_workers=4, 
            persistent_workers=True, 
            prefetch_factor=4)

    try:
        bar = tqdm(total=len(train_loader), unit=' batch', desc='train_shard', smoothing=0.03)
        for step, batch in enumerate(train_loader):
            loss = scaler.scale(contrastive_loss(model, batch, config))
            loss.backward()

            shard_loss.add_(loss.detach())

            bar.update(n=1)

            if step % grad_accum == grad_accum - 1:
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
                writer.add_scalar("lr/train", scheduler.get_last_lr()[0], training_progress.optimizer_step)
                writer.flush()

                optim.zero_grad(set_to_none=True)


    except Exception as e:
        logger.exception(e)
        if config['run_type'] == 'debug':
            pdb.post_mortem()

    avg_loss = (shard_loss/step).cpu().numpy()
    writer.add_scalar("avg_loss/train", avg_loss, training_progress.optimizer_step)


def validate(
        model, 
        valid_set, 
        config, 
        writer,
        training_progress):
    relevant_ids = range(len(valid_set))

    distances, neighbors, cosine_similarities = k_nearest_neighbors(
                model, 
                valid_set, 
                embedding_size=config['embedding_size'], 
                top_k=config['top_k'], 
                config=config)

    neighbors_list = [list(n) for n in neighbors]
    score = mrr(list(relevant_ids), neighbors_list)

    i = training_progress.optimizer_step
    writer.add_scalar("mrr/validation", score, i)
    writer.add_scalar("distances/validation", np.mean(distances), i)
    writer.add_scalar("cosine_similarity/validation", np.mean(cosine_similarities), i)
    writer.flush()


def train(config):
    logger = logging.getLogger('train')
    writer = SummaryWriter(log_dir=os.path.join(config['logdir'], config['name']), flush_secs=30)

    model = EmbeddingModel(config)
    model = model.to(config['device'])

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

    train_set = preprocess_split('train', config)
    valid_set = preprocess_split('validation', config)
    # scroll screen
    print("\n"*8)

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
            model=model,
            optim=optim,
            scheduler=scheduler,
            scaler=scaler,
            training_progress=training_progress)

    # load latest checkpoint 
    state_tracker.load_latest()

    if config['run_type'] == 'debug':
        num_epochs = 2
        num_shards = 2
        training_progress.epoch = 0
        training_progress.shard = 0
    else:
        num_epochs = config['epochs']
        num_shards = config['shards']

    shard_size = len(train_set)/num_shards
    val_size = len(valid_set)

    logger.info(f'shard_size: {shard_size} samples')
    logger.info(f'validation set size: {val_size} samples')

    try:
        # training
        logger.info('starting training')
        while(training_progress.epoch < num_epochs):
            logger.info(f'starting epoch: {training_progress.epoch} of {num_epochs}')
            train_set.shuffle()

            while(training_progress.shard < num_shards):
                logger.info('training shard')
                train_shard(
                    state_tracker,
                    train_set.shard(num_shards, training_progress.shard),
                    config,
                    writer,
                    )

                if config['run_type'] == 'default':
                    state_tracker.save()

                if config['run_type'] in ['default', 'debug']:
                    logger.info('validating shard')
                    validate(state_tracker['model'], 
                            valid_set, 
                            config, 
                            writer, 
                            state_tracker['training_progress'])

                training_progress.shard += 1
            training_progress.epoch += 1
            training_progress.shard = 0


    except KeyboardInterrupt as e:
        state_tracker.save()
        logger.info('training interrupted')
    except Exception as e:
        logger.exception(e)
        if config['run_type'] == 'debug':
            pdb.post_mortem()

    logger.info("training finished")


def make_logger(config):
    if config['run_type'] == 'debug':
        level = logging.DEBUG
    else: 
        level = logging.INFO

    logger = logging.getLogger('train')
    logger.setLevel(level)

    path = os.path.join(config['logdir'], config['name'], 'run.log')

    fh = logging.FileHandler(path)
    ch = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual training script.')
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    logdir = os.path.join(config['logdir'], config['name'])
    checkdir = os.path.join(config['checkpoint_dir'], config['name'])
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(checkdir, exist_ok=True)

    make_logger(config)

    np.random.seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.cuda.empty_cache()

    if config['run_type'] == 'profile':
        profile(config)
    else:
        train(config)

#neighsamples = valid_set_tokenized.select(neighbors.flatten())['proc_url']
#for dist, s, rid in zip(distances.flatten(), neighsamples)
#for k in range(top_k)
#valid_set.select(neighbors)
