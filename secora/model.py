import torch
from torch import tensor

from transformers import AutoModel, AutoModelForMaskedLM, AutoModelForPreTraining, AutoTokenizer
from transformers import PreTrainedModel
from tokenizers import Tokenizer

from torch.cuda.amp import autocast
from enum import Enum, auto
from .config import *

from collections import OrderedDict
'''
dropout = FloatSetting('dropout', lb=0., ub=1.)
basemodel = EnumSetting('model_name', BaseModel)
embedding_size = IntSetting('embedding_size', 1, 1024)
'''

class BaseModel(Enum):
    CODEBERT = 'microsoft/codebert-base'

#hidden_dropout_prob=config['dropout']
class EmbeddingModel(torch.nn.Module):
    ''' example:
        model_name = 'bert-base-cased'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model = EmbeddingModel(model_name)
        '''

    def __init__(self, basemodel: BaseModel, embsize: int, **kwargs):
        super().__init__()
        self.base_model = AutoModelForMaskedLM.from_pretrained(
                basemodel.value, **kwargs).base_model

        # use [cls] pooling like simcse, because of its effectiveness
        self.embsize = embsize
        self.pooling = torch.nn.Linear(
                self.base_model.config.hidden_size,
                self.embsize
                )
        self.activation = torch.nn.Tanh()

    @property
    def embedding_size(self):
        return self.embsize

    def forward(self, input_ids, token_type_ids, attention_mask, *args, **kwargs):
        x = self.base_model(
                input_ids=input_ids, 
                token_type_ids=token_type_ids, 
                attention_mask=attention_mask, 
                *args, 
                **kwargs).last_hidden_state
        x = x[:, 0, :]
        x = self.pooling(x)
        #x = self.activation(x)
        return x

class BiEmbeddingModel(torch.nn.Module):
    def __init__(self, basemodel: BaseModel, embsize: int, **kwargs):
        super().__init__()
        self.m = EmbeddingModel(basemodel, embsize, **kwargs)

    @property
    def embedding_size(self):
        return self.m.embedding_size

    def forward(self, input_ids, token_type_ids, attention_mask, *args, **kwargs):
        x1 = self.m(input_ids, token_type_ids, attention_mask, *args, **kwargs)
        if self.training == True:
            x2 = self.m(input_ids, token_type_ids, attention_mask, *args, **kwargs)
            
            x = torch.cat([torch.unsqueeze(x1, dim=1), torch.unsqueeze(x2, dim=1)], dim=1)
            return x

        else:
            return torch.unsqueeze(x1, dim=1)


class AMP(Enum):
    FP32 = 'fp32'
    FP16 = 'fp16'
    BF16 = 'bf16'
    DEFAULT = 'default'
    DISABLE = 'disable'

_precision_map = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16
        }

class BiEmbeddingModelCuda(torch.nn.Module):
    def __init__(self, basemodel: BaseModel, embsize: int, amp: AMP, **kwargs):
        super().__init__()
        self.m = BiEmbeddingModel(basemodel, embsize, **kwargs)
        self.amp = amp
        self.is_graphed = False

    @property
    def embedding_size(self):
        return self.m.embedding_size

    def make_graphed(self, dummy_inputs):
        if self.is_graphed == True:
            return 

        self.is_graphed = True
        self.ungraphed_model = self.m
        torch.cuda.synchronize()

        self.m = torch.cuda.make_graphed_callables(self.m, dummy_inputs)
        torch.cuda.synchronize()

    def forward(self, input_ids, token_type_ids, attention_mask, *args, **kwargs):
        if self.amp in [AMP.DISABLE, AMP.DEFAULT]:
            tp = {}
        else:
            tp = {'dtype': _precision_map[self.amp.value]}
        with autocast(enabled=(self.amp != AMP.DISABLE), **tp):
            return self.m(input_ids, token_type_ids, attention_mask, *args, **kwargs)


def get_model(checkpoint_path, config, device):
    st = torch.load(checkpoint_path, map_location=device)[0]
    model = BiEmbeddingModel(config).to(device)
    st2 = OrderedDict()
    for k in st.keys():
        v = st[k]
        #st2[k.removeprefix('module.')] = v
        st2[k.replace('module.', '', 1)] = v
        
    model.load_state_dict(st2)
    return model
