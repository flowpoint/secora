from itertools import cycle

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from enum import Enum, auto
from .config import Setting

LANGUAGES = ['python',
    'java',
    'php',
    'javascript',
    'ruby',
    'go',
    'all',]

class LanguageSetting(Setting):
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def parse(self, s):
        return s

    @property
    def allowed_type(self):
        return list

    def check(self, val_list):
        return all([x in LANGUAGES for x in val_list])


def preproc_valid(sample):
    # delete docstring from code samples
    #return {'func_code_string': sample['func_code_string'].replace(sample['func_documentation_string'], '')}
    return {'func_code_string': " ".join(sample['func_code_tokens'])}


def fair_truncate_old(doc, code, max_length):
    ''' truncates two strings fairly '''
    dlen = len(doc)
    clen = len(code)
    if dlen < max_length//2 and clen < max_length//2:
        d = doc
        c = code
    elif not dlen < max_length//2 and not clen < max_length//2:
        d = doc[:max_length//2]
        c = code[:max_length//2]
    elif not clen < max_length//2:
        d = doc
        c = code[:max_length-dlen]
    else:
        d = doc[:max_length-clen]
        c = code

    return d, c

class PreprocessMode(Enum):
    CONCAT = 'concat'

def fair_truncate(tokenizer, doc, code, max_input_tokens):
    # optimize for fair length between doc/code tokens
    #grow sequence sizes fairly, until max_input_tokens is reached
    opt_steps = 0
    max_dlength = 1
    max_clength = 1
    # return the old sample that is still < max_input_tokens
    tr_sample = doc[:max_dlength] + tokenizer.sep_token + code[:max_clength]
    new_tr_sample = tr_sample

    while True:
        tr_sample = new_tr_sample
        new_tr_sample = doc[:max_dlength] + tokenizer.sep_token + code[:max_clength]
        tok = tokenizer(
                new_tr_sample,
                padding=False, 
                max_length=None, 
                truncation=False, 
                return_token_type_ids=True)


        opt_steps += 1
        if opt_steps > 1000:
            raise RuntimeError('too many steps taken during fair tokenization')
        
        if len(tok['input_ids']) >= max_input_tokens:
            return tr_sample
        elif max_dlength <= max_clength and max_dlength < len(doc):
            max_dlength = int(max_dlength*1.2)+1
        elif max_clength <= max_dlength and max_clength < len(code):
            max_clength = int(max_clength*1.2)+1
        # both doc and code fully fit
        else:
            return doc + tokenizer.sep_token + code

def tokenize_train_sample(tokenizer, sample, mode, max_input_tokens):
    ''' this is run in batch mode, so the features are batched '''
    doc = " ".join(sample['func_documentation_tokens'])
    code = " ".join(sample['func_code_tokens'])

    #if mode == V'joint':
    #    whole = batch['whole_func_string']

    if mode == PreprocessMode.CONCAT:
        trunc_sample = fair_truncate(tokenizer, doc, code, max_input_tokens)
        tokenized_sample = tokenizer(trunc_sample, padding='max_length', max_length=max_input_tokens, truncation=True, return_token_type_ids=True)
    else:
        raise RuntimeError(f"preprocess mode: {mode} is not supported")

    proc_batch = dict()

    for k, v in tokenized_sample.items():
        proc_batch[k] = v

    return proc_batch


def tokenize_valid_sample(tokenizer, batch, max_input_tokens):
    code = batch['func_code_string']
    doc = [x + tokenizer.sep_token for x in batch['func_documentation_string']]

    # call tokenizer twice instead of on a pair, so that its impossible to leak data between code and doc
    # instead of the joint tokenization
    tokenized_code = tokenizer(code, padding='max_length', max_length=max_input_tokens, truncation=True, return_token_type_ids=True)
    tokenized_doc = tokenizer(doc, padding='max_length', max_length=max_input_tokens, truncation=True, return_token_type_ids=True)

    proc_batch = dict()

    for k, v in tokenized_code.items():
        proc_batch['code_'+k] = v

    for k, v in tokenized_doc.items():
        proc_batch['doc_'+k] = v

    return proc_batch

class DataSplit(Enum):
    TRAIN = 'train'
    VALIDATION = 'validation'
    TEST = 'test'
    EVAL = 'eval'

def preprocess_split(split, config, limit_samples=None, **kwargs):
    if not isinstance(split, DataSplit):
        raise RuntimeError(f"invalid dataset split: {split}")

    datasets.set_progress_bar_enabled(kwargs['progress'])
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'].value)

    num_proc = config['preprocess_cores']

    if split == DataSplit.EVAL:
        dataset = load_dataset("code_search_net")['train']
    else:
        dataset = load_dataset("code_search_net")[split.value]

    if limit_samples is not None:
        if limit_samples < 1:
            raise RuntimeError('invalid limit_samples')
        dataset = dataset.select(range(limit_samples))

    if config['languages'] != 'all':
        dataset = dataset.filter(lambda x: x['language'] in config['languages'], num_proc=num_proc)

    if split != DataSplit.TRAIN:
        dataset = dataset.map(preproc_valid, batched=False, num_proc=num_proc)

    dataset = dataset.rename_column("func_code_url", "url")

    if split == DataSplit.TRAIN:
        def tokenize_fn(x): return tokenize_train_sample(tokenizer, x, config)
    else:
        def tokenize_fn(x): return tokenize_valid_sample(tokenizer, x, config['max_input_tokens'])

    dataset = dataset.map(
            tokenize_fn,
            remove_columns=set(dataset.column_names) - {'url', 'language'},
            batched=True,
            num_proc=num_proc)

    # cast dataset to torch tensors
    dataset.set_format(type='torch', columns=set(dataset.column_names) - {'url','language'}, output_all_columns=True)

    return dataset


def get_loader(dataset, config, **kwargs):
    sampler = DistributedSampler(dataset, drop_last=True, shuffle=False)
    loader = DataLoader(
            dataset, 
            batch_size=config['batch_size'], 
            shuffle=False,
            drop_last=True, 
            pin_memory=True, 
            # workers need to use the spawn or forkserver method in a distributed setting
            num_workers=6, 
            multiprocessing_context='spawn',
            persistent_workers=True, 
            sampler=sampler)

    return loader


def deviceloader(loader, device):
    for b in loader:
        for k, v in b.items():
            if isinstance(v, torch.Tensor):
                b[k] = v.to(device)
        yield b
