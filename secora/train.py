import logging

from torch.utils.data import DataLoader

from datasets import load_dataset
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

#from model import Model

class TrainConfig:
    def __init__(self, base_model, tokenizer):
        pass

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

from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)
'''
tokenizer.set_truncation_and_padding(
        padding_strategy=transformers.file_utils.PaddingStrategy.MAX_LENGTH, 
        truncation_strategy=transformers.tokenization_utils_base.TruncationStrategy.LONGEST_FIRST, 
        max_length=32, 
        stride=1, 
        pad_to_multiple_of=1)
'''

model = AutoModel.from_pretrained(model_name)

##
def train_tokenizer():
    pass


def train_model():
    #writer.add_scalar("loss/multiple_negative_ranking", loss, epoch)

    writer.flush()
    writer.close()


train_tokenizer()
train_model()


# the bert models have dropout=0.1 by default
model.train()

similarity = torch.nn.CosineSimilarity(dim=-1)

optim = torch.optim.Adam(model.parameters())

##
batch = next(iter(train_loader))

# train unsupervisedly
for i in range(10000):
    #for batch in train_loader:

    batch_tokens = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)

    out = model(**batch_tokens)
    out2 = model(**batch_tokens)

    # use the cls output token as embedding vector
    # todo add the mlp ontop of the vector
    cls_v = out['last_hidden_state'][:,0,:]
    cls2_v = out2['last_hidden_state'][:,0,:]

    loss = -(similarity(cls_v, cls2_v).sum())

    optim.zero_grad()
    loss.backward()
    optim.step()
    print(loss)

