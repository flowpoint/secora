import os
import sys
from time import time
from math import ceil
from dataclasses import dataclass

import pdb

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import profiler

from transformers import get_linear_schedule_with_warmup

from model import *
from mrr import mrr
from data import preprocess_split
from config import config
from infer import build_embedding_space, k_nearest_neighbors

from SM3 import SM3


def contrastive_loss(model, batch, config):
    input_ids = batch['input_ids'].to(config['device'], non_blocking=True)
    token_type_ids = batch['token_type_ids'].to(config['device'], non_blocking=True)
    attention_mask = batch['attention_mask'].to(config['device'], non_blocking=True)

    emb1 = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    emb2 = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

    # use the exact same loss from the simcse repository
    sim = F.cosine_similarity(emb1.unsqueeze(1), emb2.unsqueeze(0), dim=-1) / config['temp']
    labels = torch.arange(sim.size(0), dtype=torch.int64, device=config['device'])
    loss = F.cross_entropy(sim, labels)
    return loss


def train_shard(
        model,
        optim,
        scheduler,
        train_set,
        config,
        writer,
        training_state
        ):
    ''' trains the model for until the budget is exhausted
    '''

    model.train()
    shard_loss = []

    grad_accum = config['grad_accum']

    # create the batched fast dataloader
    # (only possible with same length batches)
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=False, drop_last=True, pin_memory=True, num_workers=4, persistent_workers=True, prefetch_factor=10)

    try:

        for step, batch in tqdm(enumerate(train_loader)):
            loss = contrastive_loss(model, batch, config)
            loss.backward()

            shard_loss.append(loss.detach().cpu().numpy())

            if step % grad_accum == grad_accum - 1:
                optim.step()
                scheduler.step()

                training_state.optimizer_step += 1
                writer.add_scalar("lr/train", scheduler.get_last_lr()[0], training_state.optimizer_step)
                writer.flush()

                optim.zero_grad(set_to_none=True)


    except Exception as e:
        pdb.post_mortem()

    avg_loss = np.mean(shard_loss)
    writer.add_scalar("avg_loss/train", avg_loss)
    return avg_loss


def test_step(model, optim, batch, config):
    loss = contrastive_loss(model, batch, config)
    loss.backward()

    s_loss = loss.detach().cpu().numpy()

    optim.step()
    optim.zero_grad(set_to_none=True)


def profile(config):
    model = EmbeddingModel(config['model_name'])
    model = model.to(config['device'])

    if config['finetune_mode'] == 'all':
        params = model.parameters()
    elif config['finetune_mode'] == 'pooling':
        params = model.pooling.parameters()
    else:
        raise RuntimeError('finetune_mode has to be: all or pooling')

    train_set = preprocess_split('train', config)

    if config['optim'] == 'adam':
        optim = torch.optim.Adam(params, lr=config['lr'])
    elif config['optim'] == 'sgd':
        optim = torch.optim.SGD(params, lr=config['lr'])
    elif config['optim'] == 'sm3':
        optim = SM3(params, lr=config['lr'])
    else:
        raise RuntimeErroor('config specifies and unsupported optimizer')

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=False, drop_last=True, pin_memory=True, num_workers=4, persistent_workers=True, prefetch_factor=10)

    it = iter(train_loader)
    test_step(model, optim, next(it), config)

    tensorboard_run_path = os.path.join(config['logdir'], config['name'])
    trace_path = os.path.join(config['logdir'], config['name'], 'profile_trace.json')
    stacks_path = os.path.join(config['logdir'], config['name'], 'profile_stacks.txt')

    with profiler.profile(
            with_stack=True, 
            profile_memory=True, 
            record_shapes=True,
            on_trace_ready=profiler.tensorboard_trace_handler(tensorboard_run_path),
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ]) as p:
                test_step(model, optim, next(it), config)

    print(p.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))


##

def validate(
        model, 
        valid_set, 
        config, 
        writer,
        training_state):
    relevant_ids = range(len(valid_set))

    distances, neighbors, cosine_similarities = k_nearest_neighbors(
                model, 
                valid_set, 
                embedding_size=config['embedding_size'], 
                top_k=config['top_k'], 
                config=config)

    neighbors_list = [list(n) for n in neighbors]
    score = mrr(list(relevant_ids), neighbors_list)

    i = training_state.optimizer_step
    writer.add_scalar("mrr/validation", score, i)
    writer.add_scalar("distances/validation", np.mean(distances), i)
    writer.add_scalar("cosine_similarity/validation", np.mean(cosine_similarities), i)
    writer.flush()
    return score

class StateTracker:
    def __init__(self, objects, config):
        self.config = config
        if 'checkpoint_dir' not in config.keys():
            raise RuntimeError('config needs to have a checkpoint_dir')
        if 'max_checkpoints' not in config.keys():
            raise RuntimeError('config needs to have a max_checkpoints')
        if not config['max_checkpoints'] >= 1:
            raise RuntimeError('max_checkpoints has to be greater than 1')

        self.checkpoint_dir = config['checkpoint_dir']
        self.max_checkpoints = self.config['max_checkpoints']

        for o in objects:
            if "state_dict" not in dir(o):
                raise RuntimeError(f'tracked object: {o} need to have a state_dict() method')
            if "load_state_dict" not in dir(o):
                raise RuntimeError(f'tracked object: {o} need to have a load_state_dict() method')

        # list of (object, object
        self.objects = objects

    def _list_checkpoints(self):
        return list(filter(lambda x: x.startswith("checkpoint"), os.listdir(self.checkpoint_dir)))

    def save(self):
        timestamp = ceil(time())
        path = os.path.join(self.checkpoint_dir, "checkpoint_" + str(timestamp) + ".pt")
        state = [o.state_dict() for o in self.objects]
        torch.save(state, path)

        # remove oldest checkpoint, through lexicographic order
        checkpoints = self._list_checkpoints()

        if len(checkpoints) > self.max_checkpoints:
            old_path = os.path.join(self.checkpoint_dir, sorted(checkpoints)[0])
            os.remove(old_path)

    def load(self, checkpoint=""):
        ''' loads latest checkpoint when checkpoint is not set or the one specified checkpoint'''

        existing_checkpoints = sorted(self._list_checkpoints())
        if len(existing_checkpoints) < 1:
            raise RuntimeError("no checkpoints available")

        if checkpoint == "":
            checkpoint = existing_checkpoints[-1] 

        path = os.path.join(self.checkpoint_dir, checkpoint)
        print(path)

        if not os.path.isfile(path):
            raise RuntimeError("checkpoint_doesn't exist, set an existing checkpoint or use '' for the lastest checkpoint")

        restored_state = torch.load(path)
        objects = [o.load_state_dict(p) for o,p in zip(self.objects, restored_state)]
        return objects


@dataclass
class TrainingState:
    ''' finished epochs and shards '''
    epoch: int = -1
    shard: int = -1
    optimizer_step: int = -1

    def state_dict(self,):
        return { "epoch": self.epoch, 
            "shard": self.shard,
            "optimizer_step": self.optimizer_step
            }

    def load_state_dict(self, state):
        self.epoch = state['epoch']
        self.shard = state['shard']
        self.optimizer_step = optimizer_step['batch']




def training_run(config, checkpoint_dir=None):
    writer = SummaryWriter(log_dir=os.path.join(config['logdir'], config['name']), flush_secs=30)

    model = EmbeddingModel(config['model_name'])
    model = model.to(config['device'])

    if config['finetune_mode'] == 'all':
        params = model.parameters()
    elif config['finetune_mode'] == 'pooling':
        params = model.pooling.parameters()
    else:
        raise RuntimeError('finetune_mode has to be: all or pooling')

    train_set = preprocess_split('train', config)
    valid_set = preprocess_split('validation', config)

    if config['optim'] == 'adam':
        optim = torch.optim.Adam(params, lr=config['lr'])
    elif config['optim'] == 'sgd':
        optim = torch.optim.SGD(params, lr=config['lr'])
    elif config['optim'] == 'sm3':
        optim = SM3(params, lr=config['lr'])
    else:
        raise RuntimeErroor('config specifies and unsupported optimizer')

    num_warmup_steps = ceil(config['warmup_batches'] / config['grad_accum'])
    num_training_steps_per_epoch = ceil(len(train_set) / (config['batch_size'] * config['grad_accum']))
    num_training_steps = config['epochs'] * num_training_steps_per_epoch

    scheduler = get_linear_schedule_with_warmup(
            optim,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps)

    training_state = TrainingState()
    state_tracker = StateTracker([model, optim, training_state, scheduler], config)

    if checkpoint_dir:
        model, optim, training_state, scheduler = training_tracker.load()

    try:
        for epoch in range(config['epochs']):
            training_state.epoch = epoch
            train_set.shuffle()

            for k in range(config['shards']):
                shard_loss = train_shard(
                        model,
                        optim,
                        scheduler,
                        train_set.shard(config['shards'], k),
                        config,
                        writer,
                        training_state
                        )

                training_state.shard = k
                state_tracker.save()

                print('validating now')
                # validate(model, valid_set, config, writer, training_state)
                #tune.report(mrr=score, loss=shard_loss)


    except KeyboardInterrupt as e:
        print('training interrupted')
        pdb.post_mortem()


if __name__ == "__main__":
    if config['run_type'] == 'profile':
        profile(config)
    else:
        training_run(config)



#neighsamples = valid_set_tokenized.select(neighbors.flatten())['proc_url']
#for dist, s, rid in zip(distances.flatten(), neighsamples)
#for k in range(top_k)
#valid_set.select(neighbors)
