import os
import sys
import logging
from abc import ABC, abstractmethod
from time import time, sleep

import pdb

from dataclasses import dataclass
from typing import Union
from enum import Enum

#import third_party.faiss as faiss
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import faiss
import ray
from ray import tune

from secora.model import *
from secora.mrr import mrr
from secora.data import get_dataloaders

ray.init(local_mode=True)

config = {}
# experiment name
config['name'] = 'proper_run_1'
config['dryrun'] = False
config['batch_size'] = 4
#config['lr'] = 1e-5

config['shard_steps'] = 2048 #2**16
config['grad_accum'] = 64 // config['batch_size']
# temperature/ weighting of cosine sim
# taken from simcse
config['temp'] = 0.05

config['embedding_size'] = 128
config['top_k'] = 5

config['logdir'] = './output/runs'
config['model_name'] = 'huggingface/CodeBERTa-small-v1'
#model_name = 'bert-base-cased'
#model_name = 'bert-large-cased'

#config['lr'] = tune.sample_from(lambda spec: 10**(-10 * np.random.rand()))
config['lr'] = 1e-5
#config['momentum'] = tune.uniform(0.1,0.9)

#config['finetune_mode'] = 'pooling'
config['finetune_mode'] = 'all'
##
if torch.cuda.is_available():
    config['device'] = torch.device('cuda')
else:
    config['device'] = torch.device('cpu')

# logger = logging.getLogger(__name__)
# writer = SummaryWriter()

##
# the bert models have dropout=0.1 by default

class Budget(ABC):
    @abstractmethod
    def consume(self):
        pass

    @abstractmethod
    def is_exhausted(self) -> bool:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class StepBudget(Budget):
    def __init__(self, steps):
        self.steps = steps
        self.current_step = steps

    def consume(self, steps):
        if self.is_exhausted():
            raise RuntimeError("can't consume an already exhausted budget")
        if steps <= 0:
            raise ValueError("can only consume a positive value")

        self.current_step -= steps

    def is_exhausted(self):
        return self.current_step < 0

    def __str__(self):
        return f"StepBuget: {self.current_step}/{self.steps} remaining"


class TimeBudget(Budget):
    def __init__(self, seconds):
        self.seconds = seconds
        self.starttime = time()
        self.current_time = None

    def consume(self):
        if self.current_time is not None and self.is_exhausted():
            raise RuntimeError("can't consume an already exhausted budget")
        self.current_time = time()

    def is_exhausted(self):
        if self.current_time is None:
            raise RuntimeError("budget has to be consumed() at the time it should be checked")
        return self.starttime + self.seconds < self.current_time

    def __str__(self):
        return f"TimeBudget: {(self.starttime + self.seconds)-self.current_time}/{self.seconds} remaining"

def train_shard(
        model,
        optim,
        train_loader,
        valid_loader,
        config,
        ):
    ''' trains the model for until the budget is exhausted
    '''

    model.train()
    shard_loss = []

    grad_accum = config['grad_accum']
    shard_steps = config['shard_steps']

    try:
        for step, batch in tqdm(enumerate(train_loader)):
            for k, v in batch.items():
                batch[k] = v.to(config['device'])

            input_ids = batch['proc_whole_input_ids']
            token_type_ids = batch['proc_whole_token_type_ids']
            attention_mask = batch['proc_whole_attention_mask']

            emb1 = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            emb2 = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

            sim = F.cosine_similarity(emb1, emb2) / config['temp']
            loss = torch.mean(-torch.log(torch.exp(sim) / torch.exp(sim).sum()))
            loss.backward()


            shard_loss.append(loss.detach().cpu().numpy())

            if step % grad_accum == grad_accum - 1:
                optim.step()
                optim.zero_grad()
                #tune.report(mean_loss=np.mean(shard_loss[-grad_accum:]))

            if step % shard_steps == shard_steps - 1:
                print(f'shard_loss: {np.mean(shard_loss)}')
                break

    except Exception as e:
        pdb.post_mortem()

    return np.mean(shard_loss)

def k_nearest_neighbors(model, valid_loader, embedding_size, top_k, batch_size=2):
    dataset_shape = (len(valid_loader)*batch_size, embedding_size)
    # allocate the dataset_embedding
    code_embedding = np.zeros(dataset_shape, dtype=np.float32)
    doc_embedding = np.zeros(dataset_shape, dtype=np.float32)

    model.eval()
    relevant_ids = range(dataset_shape[0])

    simil = []
    #build the faiss index
    with torch.no_grad():
        for i, batch in tqdm(enumerate(valid_loader)):
            for k, v in batch.items():
                batch[k] = v.to(config['device'])

            code_emb = model(input_ids=batch['proc_code_input_ids'],
                    token_type_ids=batch['proc_code_token_type_ids'],
                    attention_mask=batch['proc_code_attention_mask'])

            doc_emb = model(input_ids=batch['proc_doc_input_ids'],
                    token_type_ids=batch['proc_doc_token_type_ids'],
                    attention_mask=batch['proc_doc_attention_mask'])
            
            simil.append(F.cosine_similarity(code_emb, doc_emb).detach().cpu().numpy())

            code_embedding[i*batch_size:(i+1)*batch_size] = code_emb.detach().cpu().numpy()
            doc_embedding[i*batch_size:(i+1)*batch_size] = doc_emb.detach().cpu().numpy()

    index = faiss.index_factory(embedding_size, 'Flat')
    index.train(code_embedding)
    index.add(code_embedding)

    distances, neighbors = index.search(doc_embedding.astype(np.float32), top_k)

    print(f'simil: {np.mean(simil)}')
    return distances, neighbors, relevant_ids
##

def validate(model, valid_loader, config):#embedding_size=128, top_k=5, batch_size=1):
    distances, neighbors, relevant_ids = k_nearest_neighbors(
                model, 
                valid_loader, 
                embedding_size=config['embedding_size'], 
                top_k=config['top_k'], 
                batch_size=config['batch_size'])

    neighbors_list = [list(n) for n in neighbors]
    score = mrr(list(relevant_ids), neighbors_list)
    print(f'mrr: {score}')

    return score


def training_run(config):
    model = EmbeddingModel(config['model_name'])
    model = model.to(config['device'])

    if config['finetune_mode'] == 'all':
        optim = torch.optim.Adam(model.parameters(), lr=config['lr'])
    elif config['finetune_mode'] == 'pooling':
        optim = torch.optim.Adam(model.pooling.parameters(), lr=config['lr'])
    else:
        raise RuntimeError('finetune_mode has to be: all or pooling')

    train_loader, valid_loader = get_dataloaders(config)

    mini_epochs = 3 * (len(train_loader) // config['shard_steps'])

    for k in range(mini_epochs):
        shard_loss = train_shard(
                model,
                optim,
                train_loader,
                valid_loader,
                config,
                )

        with tune.checkpoint_dir(k) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optim.state_dict()), path)

        print('validating now')

        score = validate(model, valid_loader, config)#128, 5, batch_size=config['batch_size'])
        tune.report(mrr=score, loss=shard_loss)

analysis = tune.run(
        training_run,
        config=config,
        local_dir=config['logdir'],
        metric='mrr',
        mode='max',
        num_samples=1,
        keep_checkpoints_num=10,
        name=config['name'],
        resume="AUTO"
        #resources_per_trial={'cpu':10},
        )

dfs = analysis.trial_dataframes
#[d.mean_accuracy.plot() for d in dfs.values]

'''
score = 0
for dist, nbs, rid in zip(distances, neighbors, relevant_ids):
    if rid in nbs:
        score += 1
'''

#neighsamples = valid_set_tokenized.select(neighbors.flatten())['proc_url']
#for dist, s, rid in zip(distances.flatten(), neighsamples)
#for k in range(top_k)
#valid_set.select(neighbors)


