import logging
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
config['dryrun'] = True
config['step_limit'] = 2
config['batch_size'] = 4
#config['lr'] = 1e-5

config['shard_steps'] = 12
config['grad_accum'] = 10
# temperature/ weighting of cosine sim
# taken from simcse
config['temp'] = 0.05

config['embedding_size'] = 128
config['top_k'] = 5

config['logdir'] = './output/logdir'
config['model_name'] = 'huggingface/CodeBERTa-small-v1'
#model_name = 'bert-base-cased'
#model_name = 'bert-large-cased'

config['lr'] = tune.sample_from(lambda spec: 10**(-10 * np.random.rand()))
config['momentum'] = tune.uniform(0.1,0.9)

if torch.cuda.is_available():
    config['device'] = torch.device('cuda')
else:
    config['device'] = torch.device('cpu')

# logger = logging.getLogger(__name__)
# writer = SummaryWriter()

##
# the bert models have dropout=0.1 by default


def train_shard(
    model,
    optim,
    train_loader,
    valid_loader,
    config
    ):

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

                tune.report(mean_loss=np.mean(shard_loss[-grad_accum:]))

            if step % shard_steps == shard_steps - 1:
                print(f'shard_loss: {np.mean(shard_loss)}')
                break


    except Exception as e:
        pdb.post_mortem()

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

    optim = torch.optim.Adam(model.parameters(), lr=config['lr'])
    train_loader, valid_loader = get_dataloaders(config)

    mini_epochs = 1
    for k in range(mini_epochs):
        train_shard(
                model,
                optim,
                train_loader,
                valid_loader,
                config
                )
        score = validate(model, valid_loader, config)#128, 5, batch_size=config['batch_size'])
        tune.report(mrr=score)

analysis = tune.run(
        training_run,
        config=config,
        num_samples=20,
        resources_per_trial={'cpu':10},
        )

dfs = analysis.trial_dataframes
[d.meaan_accuracy.plot() for d in dfs.values]

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


