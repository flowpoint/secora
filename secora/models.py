import torch
from torch import tensor

from transformers import AutoModel, AutoModelForMaskedLM, AutoModelForPreTraining, AutoTokenizer
from transformers import PreTrainedModel
from tokenizers import Tokenizer

from torch.cuda.amp import autocast
from enum import Enum, auto
from .config import *

from collections import OrderedDict


class BaseModel(Enum):
    CODEBERT = 'microsoft/codebert-base'
    ROBERTA = 'roberta-base'
    DISTILROBERTA = 'distilroberta-base'


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

DROPOUT_SETTING = FloatSetting('dropout', lb=0., ub=1.)
BASEMODEL_SETTING = EnumSetting('model_name', BaseModel)
AMP_SETTING = EnumSetting('amp', AMP)


class EmbeddingModel(torch.nn.Module):
    def __init__(self, basemodel: BaseModel, embsize: int, **kwargs):
        super().__init__()
        self.base_model = AutoModelForMaskedLM.from_pretrained(
                basemodel.value, **kwargs).base_model

        # add a pooling mlp as some models don't have one
        self.embsize = embsize
        self.pooling = torch.nn.Linear(
                self.base_model.config.hidden_size,
                self.embsize
                )
        self.activation = torch.nn.Tanh()

    @property
    def embedding_size(self):
        return self.embsize

    def forward(self, input_ids, attention_mask, *args, **kwargs):
        x = self.base_model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                *args, 
                **kwargs)
        # [cls] pooling like simcse
        x = x.last_hidden_state[:, 0, :]
        x = self.pooling(x)
        x = self.activation(x)
        return x

class EmbeddingModelCuda(torch.nn.Module):
    def __init__(self, basemodel: BaseModel, embsize: int, amp: AMP, **kwargs):
        ''' wraps an Embeddingmodel to work faster with cuda acceleration
        '''
        super().__init__()
        self.m = EmbeddingModel(basemodel, embsize, **kwargs)
        self.amp = amp
        self.is_graphed = False

    @property
    def embedding_size(self):
        return self.m.embedding_size

    def make_graphed(self, dummy_inputs):
        ''' turns model into a cuda graphed model '''
        if self.is_graphed == True:
            return 

        self.is_graphed = True
        self.ungraphed_model = self.m
        torch.cuda.synchronize()

        self.m = torch.cuda.make_graphed_callables(self.m, dummy_inputs)
        torch.cuda.synchronize()

    def forward(self, input_ids, attention_mask, *args, **kwargs):
        if self.amp in [AMP.DISABLE, AMP.DEFAULT]:
            tp = {}
        else:
            tp = {'dtype': _precision_map[self.amp.value]}
        with autocast(enabled=(self.amp != AMP.DISABLE), **tp):
            return self.m(input_ids, attention_mask, *args, **kwargs)

def build_model(config, **kwargs):
    ''' unified function for building a model according to config '''
    logger = kwargs['logger']
    rank = kwargs['rank']
    logger.info('building model')

    model_args = (config['model_name'], config['embedding_size'], config['amp'])
    model_kwargs = {'hidden_dropout_prob': config['dropout']}
    if config['num_gpus'] > 0:
        m = EmbeddingModelCuda(*model_args, **model_kwargs).to(rank)
    else:
        m = EmbeddingModel(*model_args, **model_kwargs).to(rank)
    return m


def get_model(checkpoint_path, embsize, device):
    st = torch.load(checkpoint_path, map_location=device)[0]
    model = EmbeddingModel(BaseModel.CODEBERT, embsize).to(device)
    st2 = OrderedDict()
    for k in st.keys():
        v = st[k]
        #st2[k.removeprefix('module.')] = v
        st2[k.replace('module.m.', '', 1)] = v
        
    model.load_state_dict(st2)
    return model
