import torch
from torch import tensor

from transformers import AutoModelForPreTraining, AutoTokenizer
from tokenizers import Tokenizer


from abc import ABC


def get_model():
    #model = AutoModelForPreTraining.from_pretrained(model_name)
    tokenizer = AutoTokenizer(model_name)

    return model, tokenizer


class RetrievalModel(ABC):
    def __init__(self):
        pass








