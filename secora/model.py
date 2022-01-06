import torch
from torch import tensor

from transformers import AutoModel, AutoModelForMaskedLM, AutoModelForPreTraining, AutoTokenizer
from transformers import PreTrainedModel
from tokenizers import Tokenizer

from torch.cuda.amp import autocast


class EmbeddingModel(torch.nn.Module):
    ''' example:
        model_name = 'bert-base-cased'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model = EmbeddingModel(model_name)
        '''

    def __init__(self, config):
        super().__init__()

        self.base_model = AutoModelForMaskedLM.from_pretrained(config['model_name']).base_model

        # use [cls] pooling like simcse, because of its effectiveness
        self.embsize = config['embedding_size']
        self.precision = config['precision']

        self.pooling = torch.nn.Linear(
                self.base_model.config.hidden_size,
                self.embsize
                )
        self.activation = torch.nn.Tanh()

    def forward(self, *args, **kwargs):
        with autocast(enabled=self.precision == 'mixed'):
            x = self.base_model(*args, **kwargs).last_hidden_state
            x = x[:, 0, :]
            x = self.pooling(x)
            x = self.activation(x)
            return x
        


