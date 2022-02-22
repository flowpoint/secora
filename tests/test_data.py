import pytest
import os
from secora.data import *
from transformers import AutoTokenizer
import tempfile
from datasets import Dataset

#sample = {'func_documentation_tokens': inputa, 'func_code_tokens', inputb}

def get_mock_dataset(mlen=100):
    d = {"url": ['u']*mlen, 
        'func_documentation_tokens':['a']*mlen, 
        'func_code_tokens': ['b']*mlen,
        "language": ['python']*mlen}
    return Dataset.from_dict(d)

def get_mock_processed_dataset(mlen=100):
    d = {"url": ['u']*mlen, 
        'func_input_ids': [1]*mlen, 
        'func_attention_mask': [1]*mlen,
        'code_input_ids': [1]*mlen, 
        'code_attention_mask': [1]*mlen,
        "language": ['python']*mlen}
    return Dataset.from_dict(d)


def get_mock_train_split():
    ds = get_mock_dataset()
    conf = {'preprocess_cores': 1, 'languages': ['python'], 'max_input_tokens': 256, 'model_name': 'microsoft/codebert-base', 'preprocess_mode': 'concat'}
    return preprocess_split('train', conf)


@pytest.mark.slow
def test_fair_truncate_both_long():
    tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')

    for max_input_tokens in [10,100,512,1024, 2048]:
        doc = 'a'*1025
        code = 'b'*1025

        tokenizer(doc+tokenizer.sep_token+code)

        truncated_sample = fair_truncate(tokenizer, doc, code, max_input_tokens)
        halflen = max_input_tokens//2 - 2
        assert truncated_sample.startswith('a'*halflen)
        assert truncated_sample.endswith('b'*halflen)

@pytest.mark.slow
def test_fair_truncate_short_doc():
    tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')

    for max_input_tokens in [10,100,512,1024, 2048]:
        doc = 'a'*(max_input_tokens//5)
        code = 'b'*1025
    
        truncated_sample = fair_truncate(tokenizer, doc, code, max_input_tokens)
        assert truncated_sample.startswith(doc)

@pytest.mark.slow
def test_fair_truncate_short_code():
    tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')

    for max_input_tokens in [10,100,512,1024, 2048]:
        doc = 'a'*1025
        code = 'b'*(max_input_tokens//5)
    
        truncated_sample = fair_truncate(tokenizer, doc, code, max_input_tokens)
        assert truncated_sample.endswith(code)

@pytest.mark.slow
def test_fair_truncate_both_short():
    tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')

    for max_input_tokens in [10,100,512,1024, 2048]:
        doc = 'a'*(max_input_tokens//4)
        code = 'b'*(max_input_tokens//4)

        truncated_sample = fair_truncate(tokenizer, doc, code, max_input_tokens)
    # test with short doc
        assert truncated_sample.startswith(doc)
        assert truncated_sample.endswith(code)

