import logging

from dataclasses import dataclass
from typing import Union
from enum import Enum

#import third_party.faiss as faiss
import faiss
import numpy as np

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from datasets import load_dataset


from transformers import AutoModel, AutoModelForMaskedLM, AutoModelForPreTraining, AutoTokenizer
from transformers import PreTrainedModel
from tokenizers import Tokenizer

from secora.model import *

dryrun = True
step_limit = 2
batch_size = 2

embedding_size = 128
top_k = 5

#max_len = 
# gradient accum
# lr
logdir = './output/logdir'

#model_name = 'huggingface/CodeBERTa-small-v1'
model_name = 'bert-base-cased'
#model_name = 'bert-large-cased'

device = torch.device('cpu')
dataset = load_dataset("code_search_net")

def preproc_valid(sample):
    # delete docstring from code samples
    return {'func_code_string': sample['func_code_string'].replace(sample['func_documentation_string'], '')}

if dryrun == True:
    train_set = dataset['train'].select(range(100))
    valid_set = dataset['validation'].select(range(100)).map(preproc_valid, batched=False)
else:
    train_set = dataset['train']
    valid_set = dataset['validation'].map(preproc_valid, batched=False)

##
logger = logging.getLogger(__name__)
writer = SummaryWriter()

##
# the bert models have dropout=0.1 by default
model = EmbeddingModel(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

similarity = torch.nn.CosineSimilarity(dim=-1)
optim = torch.optim.Adam(model.parameters())

# ! important the dummy inputs must be of the right shape (maximal seqlen)


##
def tokenize_train_sample(batch):
    whole = batch['whole_func_string']
    tokenized_code = tokenizer(whole, padding=True, truncation=True)
    url = batch['func_code_url']

    for k, v in tokenized_code.items():
        batch['proc_whole_'+k] = v

    batch['proc_url'] = url
    return batch

def tokenize_valid_sample(batch):
    code = batch['func_code_string']
    doc = batch['func_documentation_string']

    url = batch['func_code_url']

    tokenized_code = tokenizer(code, padding=True, truncation=True)
    tokenized_doc = tokenizer(doc, padding=True, truncation=True)

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
valid_set.set_format(type='torch', columns=set(valid_set.column_names) - {'proc_url'})
##

# create the batched fast dataloader
# (only possible with same length batches)
train_loader = DataLoader(train_set_tokenized, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set_tokenized, batch_size=batch_size, shuffle=False, drop_last=True)
##
##
def train_shard(
    model,
    tokenizer,
    optimizer,
    similarity,
    train_loader,
    valid_loader,
    ):

    model.train()

    for step, batch in enumerate(train_loader):
        batch_tokens = tokenizer(batch['whole_func_string'], return_tensors='pt', padding=True, truncation=True)

        input_ids = batch_tokens['input_ids']
        token_type_ids = batch_tokens['token_type_ids']
        attention_mask = batch_tokens['attention_mask']

        emb1 = model(input_ids, token_type_ids, attention_mask)
        emb2 = model(input_ids, token_type_ids, attention_mask)

        loss = -(similarity(emb1, emb2).mean())

        optim.zero_grad()
        loss.backward()
        optim.step()
        print(loss)

        step += 1
##
train_shard(
        model,
        tokenizer,
        optim,
        similarity,
        train_loader,
        valid_loader
        )
##
def k_nearest_neighbors(model, valid_loader, embedding_size, top_k, batch_size=batch_size):
    dataset_shape = (len(valid_loader), embedding_size)
    # allocate the dataset_embedding
    code_embedding = np.zeros(dataset_shape, dtype=np.float32)
    doc_embedding = np.zeros(dataset_shape, dtype=np.float32)

    model.eval()


    #build the faiss index
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):

            code_emb = model(input_ids=code_tokens['input_ids'],
                    token_type_ids=code_tokens['token_type_ids'],
                    attention_mask=code_tokens['attention_mask'])

            doc_emb = model(input_ids=code_tokens['input_ids'],
                    token_type_ids=code_tokens['token_type_ids'],
                    attention_mask=code_tokens['attention_mask'])

            code_embedding[i*batch_size:i*batch_size+batch_size] = code_emb
            doc_embedding[i*batch_size:i*batch_size+batch_size] = doc_emb


    index = faiss.index_factory(embedding_size, 'Flat')
    index.train(code_embedding)
    index.add(code_embedding)

    distances, neighbors = index.search(doc_embedding.astype(np.float32), top_k)
    return distances, neighbors
##
def validate(model, valid_loader, embedding_size=128, top_k=5, batch_size=batch_size):
    distances, neighbors = k_nearest_neighbors(model, valid_set, embedding_size, top_k, batch_size)
    valid_set.select(neighbors)


distances, neighbors = validate(model, valid_loader, 128, 5)
##
import faiss
import numpy as np
N = 6000000
d = 128
k = 5

