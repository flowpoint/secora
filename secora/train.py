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

from model import *
from mrr import mrr
from data import get_dataloaders
from config import config

ray.init(local_mode=True)

if torch.cuda.is_available():
    config['device'] = torch.device('cuda')
else:
    config['device'] = torch.device('cpu')


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

            # use the exact same loss from the simcse repository
            sim = F.cosine_similarity(emb1.unsqueeze(1), emb2.unsqueeze(0), dim=-1) / config['temp']
            labels = torch.arange(sim.size(0)).long().to(config['device'])
            loss = F.cross_entropy(sim, labels)

            loss.backward()

            shard_loss.append(loss.detach().cpu().numpy())

            if step % grad_accum == grad_accum - 1:
                optim.step()
                optim.zero_grad(set_to_none=True)

            if step % shard_steps == shard_steps - 1:
                print(f'shard_loss: {np.mean(shard_loss)}')
                break

    except Exception as e:
        pdb.post_mortem()

    return np.mean(shard_loss)

def k_nearest_neighbors(model, valid_loader, embedding_size, top_k, batch_size=config['batch_size']):
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


def training_run(config, checkpoint_dir=None):
    model = EmbeddingModel(config['model_name'])
    model = model.to(config['device'])

    if config['finetune_mode'] == 'all':
        optim = torch.optim.Adam(model.parameters(), lr=config['lr'])
    elif config['finetune_mode'] == 'pooling':
        optim = torch.optim.Adam(model.pooling.parameters(), lr=config['lr'])
    else:
        raise RuntimeError('finetune_mode has to be: all or pooling')

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optim.load_state_dict(optimizer_state)

    train_loader, valid_loader = get_dataloaders(config)

    # only one unsupervised epoch for now
    mini_epochs = 1 * (len(train_loader) // config['shard_steps'])

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



def build_embedding_space(config, batch_size=config['batch_size'], embedding_size=768):
    model = EmbeddingModel(config['model_name']).half()
    model = model.to(config['device'])
    train_loader, valid_loader = get_dataloaders(config)

    dataset_shape = (len(train_loader)*batch_size, embedding_size)
    # allocate the dataset_embedding
    joint_embedding = np.zeros(dataset_shape, dtype=np.float32)

    model.eval()

    with torch.no_grad():
        for i, batch in tqdm(enumerate(train_loader)):
            for k, v in batch.items():
                batch[k] = v.to(config['device'])

            if i == 0:
                model = torch.jit.trace(model.forward, (batch['proc_whole_input_ids'], batch['proc_whole_token_type_ids'],batch['proc_whole_attention_mask']))

            joint_emb = model(batch['proc_whole_input_ids'],
                    batch['proc_whole_token_type_ids'],
                    batch['proc_whole_attention_mask'])

            joint_embedding[i*batch_size:(i+1)*batch_size] = joint_emb.detach().cpu().numpy()
        
    return joint_embedding


#neighsamples = valid_set_tokenized.select(neighbors.flatten())['proc_url']
#for dist, s, rid in zip(distances.flatten(), neighsamples)
#for k in range(top_k)
#valid_set.select(neighbors)
