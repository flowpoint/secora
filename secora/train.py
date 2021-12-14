import logging
import pdb

from dataclasses import dataclass
from typing import Union
from enum import Enum

#import third_party.faiss as faiss
import faiss
import numpy as np

import torch
import torch.nn.functional as F

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from datasets import load_dataset


from transformers import AutoModel, AutoModelForMaskedLM, AutoModelForPreTraining, AutoTokenizer
from transformers import PreTrainedModel
from tokenizers import Tokenizer

from secora.model import *

from tqdm import tqdm

dryrun = True
step_limit = 2
batch_size = 4

lr = 1e-5

shard_steps = 128
grad_accum = 10
# temperature/ weighting of cosine sim
# taken from simcse
temp = 0.05

embedding_size = 128
top_k = 5

logdir = './output/logdir'
model_name = 'huggingface/CodeBERTa-small-v1'
#model_name = 'bert-base-cased'
#model_name = 'bert-large-cased'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

dataset = load_dataset("code_search_net")

def preproc_valid(sample):
    # delete docstring from code samples
    return {'func_code_string': sample['func_code_string'].replace(sample['func_documentation_string'], '')}

if dryrun == True:
    train_set = dataset['train'].select(range(2048))
    valid_set = dataset['validation'].select(range(2048)).map(preproc_valid, batched=False)
else:
    train_set = dataset['train']
    valid_set = dataset['validation'].map(preproc_valid, batched=False)

##
logger = logging.getLogger(__name__)
writer = SummaryWriter()

##
# the bert models have dropout=0.1 by default
model = EmbeddingModel(model_name)
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

similarity = torch.nn.CosineSimilarity(dim=-1)
cross_entropy = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=lr)

# ! important the dummy inputs must be of the right shape (maximal seqlen)


##
def tokenize_train_sample(batch):
    whole = batch['whole_func_string']
    tokenized_code = tokenizer(whole, padding='max_length', truncation=True, return_token_type_ids=True)
    url = batch['func_code_url']

    for k, v in tokenized_code.items():
        batch['proc_whole_'+k] = v

    batch['proc_url'] = url
    return batch

def tokenize_valid_sample(batch):
    code = batch['func_code_string']
    doc = batch['func_documentation_string']

    url = batch['func_code_url']

    # tokenizer is called twice, so that its impossible to leak data between code and doc
    # instead of the joint tokenization
    tokenized_code = tokenizer(code, padding='max_length', truncation=True, return_token_type_ids=True)
    tokenized_doc = tokenizer(doc, padding='max_length', truncation=True, return_token_type_ids=True)

    for k, v in tokenized_code.items():
        batch['proc_code_'+k] = v

    for k, v in tokenized_doc.items():
        batch['proc_doc_'+k] = v

    batch['proc_url'] = url

    return batch
##

# optional:
# write a custom bucketing data collator to remove the need for padding and truncation
# resulting in dynamic length sequences

# preprocess tokenize the dataset once
# by using batched, the tokenizer automatically pads and truncates to the same length
train_set_tokenized = train_set.map(
        tokenize_train_sample,
        remove_columns=train_set.column_names,
        batched=True)

valid_set_tokenized = valid_set.map(
        tokenize_valid_sample, 
        remove_columns=valid_set.column_names, 
        batched=True)
##
# cast dataset to torch tensors
train_set_tokenized.set_format(type='torch', columns=set(train_set_tokenized.column_names) - {'proc_url'})
valid_set_tokenized.set_format(type='torch', columns=set(valid_set_tokenized.column_names) - {'proc_url'})
##

# create the batched fast dataloader
# (only possible with same length batches)
train_loader = DataLoader(train_set_tokenized, batch_size=batch_size, shuffle=True, drop_last=True)

# don't shuffle validation set!
valid_loader = DataLoader(valid_set_tokenized, batch_size=batch_size, shuffle=False, drop_last=True)
##
##
def train_shard(
    model,
    tokenizer,
    optimizer,
    train_loader,
    valid_loader,
    ):

    model.train()

    shard_loss = []

    try:
        for step, batch in tqdm(enumerate(train_loader)):
            for k, v in batch.items():
                batch[k] = v.to(device)

            input_ids = batch['proc_whole_input_ids']
            token_type_ids = batch['proc_whole_token_type_ids']
            attention_mask = batch['proc_whole_attention_mask']

            emb1 = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            emb2 = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

            sim = similarity(emb1, emb2) / temp
            loss = torch.mean(-torch.log(torch.exp(sim) / torch.exp(sim).sum()))
            loss.backward()

            shard_loss.append(loss.detach().cpu().numpy())

            if step % grad_accum == grad_accum - 1:
                optim.step()
                optim.zero_grad()
                print(f'shard_loss: {np.mean(shard_loss)}')

            if step % shard_steps == shard_steps -1:
                break

    except Exception as e:
        pdb.post_mortem()

def k_nearest_neighbors(model, valid_loader, embedding_size, top_k, batch_size=batch_size):
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
                batch[k] = v.to(device)

            code_emb = model(input_ids=batch['proc_code_input_ids'],
                    token_type_ids=batch['proc_code_token_type_ids'],
                    attention_mask=batch['proc_code_attention_mask'])

            doc_emb = model(input_ids=batch['proc_doc_input_ids'],
                    token_type_ids=batch['proc_doc_token_type_ids'],
                    attention_mask=batch['proc_doc_attention_mask'])
            
            simil.append(similarity(code_emb, doc_emb).detach().cpu().numpy())

            code_embedding[i*batch_size:(i+1)*batch_size] = code_emb.detach().cpu().numpy()
            doc_embedding[i*batch_size:(i+1)*batch_size] = doc_emb.detach().cpu().numpy()

    index = faiss.index_factory(embedding_size, 'Flat')
    index.train(code_embedding)
    index.add(code_embedding)

    distances, neighbors = index.search(doc_embedding.astype(np.float32), top_k)

    print(f'simil: {np.mean(simil)}')
    return distances, neighbors, relevant_ids
##

from secora.mrr import mrr

def validate(model, valid_loader, embedding_size=128, top_k=5, batch_size=batch_size):
    distances, neighbors, relevant_ids = k_nearest_neighbors(
                model, 
                valid_loader, 
                embedding_size, 
                top_k, 
                batch_size)

    neighbors_list = [list(n) for n in neighbors]
    score = mrr(list(relevant_ids), neighbors_list)
    print(f'mrr: {score}')
    
    return score

mini_epochs = 16
for k in range(mini_epochs):
    train_shard(
            model,
            tokenizer,
            optim,
            train_loader,
            valid_loader
            )
    score = validate(model, valid_loader, 128, 5)

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


