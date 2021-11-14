import logging

from dataclasses import dataclass
from typing import Union
from enum import Enum

from torch.utils.data import DataLoader

from datasets import load_dataset
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from model import RetrievalModel

from transformers import AutoModel, AutoModelForMaskedLM, AutoModelForPreTraining, AutoTokenizer
from transformers import PreTrainedModel
from tokenizers import Tokenizer

#from model import Model


@dataclass
class TrainConfig:
    batch_size: int
    gradient_accum: int
    lr: int


step_limit = 2

batch_size = 2
logdir = './output/logdir'

#model_name = 'huggingface/CodeBERTa-small-v1'
model_name = 'bert-base-cased'


device = torch.device('cpu')

dataset = load_dataset("code_search_net")
train_set = dataset['train'].select(range(step_limit))['whole_func_string']
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

valid_set = dataset['validation']
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)


logger = logging.getLogger(__name__)
writer = SummaryWriter()


def train_tokenizer():
    pass

def train_model():
    #writer.add_scalar("loss/multiple_negative_ranking", loss, epoch)

    writer.flush()
    writer.close()


train_tokenizer()
train_model()

##
# the bert models have dropout=0.1 by default
model = RetrievalModel(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.train()

similarity = torch.nn.CosineSimilarity(dim=-1)

optim = torch.optim.Adam(model.parameters())

##
batch = next(iter(train_loader))


# train unsupervisedly
for i in range(10000):
    #for batch in train_loader:

    batch_tokens = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)

    emb1 = model(**batch_tokens)
    emb2 = model(**batch_tokens)

    loss = -(similarity(emb1, emb2).mean())

    optim.zero_grad()
    loss.backward()
    optim.step()
    print(loss)

