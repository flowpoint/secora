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

        self.base_model = AutoModelForMaskedLM.from_pretrained(config['model_name'], hidden_dropout_prob=0.1).base_model

        # use [cls] pooling like simcse, because of its effectiveness
        self.embsize = config['embedding_size']
        self.precision = config['precision']

        self.pooling = torch.nn.Linear(
                self.base_model.config.hidden_size,
                self.embsize
                )
        self.activation = torch.nn.Tanh()

    def forward(self, input_ids, token_type_ids, attention_mask, *args, **kwargs):
        x = self.base_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, *args, **kwargs).last_hidden_state
        x = x[:, 0, :]
        x = self.pooling(x)
        x = self.activation(x)
        return x


class BiEmbeddingModel(torch.nn.Module):
    def __init__(self, config, dummy_inputs=None):
        super().__init__()
        self.precision = config['precision']
        self.is_graphed = False

        self.m = EmbeddingModel(config)

    def make_graphed(self, dummy_inputs):
        if self.is_graphed == True:
            return 

        self.is_graphed = True
        self.ungraphed_model = self.m
        torch.cuda.synchronize()

        self.m = torch.cuda.make_graphed_callables(self.m, dummy_inputs)
        torch.cuda.synchronize()

    def forward(self, input_ids, token_type_ids, attention_mask, *args, **kwargs):
        with autocast(enabled=self.precision == 'mixed'):
            x1 = self.m(input_ids, token_type_ids, attention_mask, *args, **kwargs)
            if self.training == True:
                x2 = self.m(input_ids, token_type_ids, attention_mask, *args, **kwargs)
                
                x = torch.cat([torch.unsqueeze(x1, dim=1), torch.unsqueeze(x2, dim=1)], dim=1)
                return x

            else:
                return torch.unsqueeze(x1, dim=1)

