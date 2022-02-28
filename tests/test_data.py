import pytest
import os
from secora.data import *
from transformers import AutoTokenizer
import tempfile
from datasets import Dataset

#sample = {'func_documentation_tokens': inputa, 'func_code_tokens', inputb}


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

