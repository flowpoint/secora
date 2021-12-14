import torch
from torch import tensor

from transformers import AutoModel, AutoModelForMaskedLM, AutoModelForPreTraining, AutoTokenizer
from transformers import PreTrainedModel
from tokenizers import Tokenizer


class EmbeddingModel(torch.nn.Module):
    ''' example:
        model_name = 'bert-base-cased'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model = EmbeddingModel(model_name)
        '''

    def __init__(self, pretrained_name, embedding_size=128):
        super().__init__()
        self.base_model = AutoModelForMaskedLM.from_pretrained(pretrained_name).base_model


        # use [cls] pooling like simcse, because of its effectiveness
        self.embsize = embedding_size 
        self.pooling = torch.nn.Linear(
                self.base_model.config.hidden_size,
                self.embsize
                )
        self.activation = torch.nn.Tanh()

    def forward(self, *args, **kwargs):
        x = self.base_model(*args, **kwargs).last_hidden_state
        x = x[:, 0, :]
        x = self.pooling(x)
        x = self.activation(x)
        return x
        


