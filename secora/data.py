from enum import Enum, auto
from itertools import cycle

import numpy as np

from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import datasets
from datasets import load_dataset, NamedSplitAll
from transformers import AutoTokenizer

from secora.config import Setting


LANGUAGES = ['python',
    'java',
    'php',
    'javascript',
    'ruby',
    'go',
    ]

class LanguageSetting(Setting):
    ''' Set the CodeSearchNet languages to train over 
        python, java, php, javascript, ruby, go
        
        if all is in the list of languages, all languages are used
    '''
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def parse(self, s):
        return s

    @property
    def allowed_type(self):
        return list

    def check(self, val_list):
        return val_list == ['all'] or all([x in LANGUAGES for x in val_list])


class PreprocessMode(Enum):
    CONCAT = 'concat'


class DataSplit(Enum):
    TRAIN = 'train'
    VALIDATION = 'validation'
    TEST = 'test'
    EVAL = 'eval'


def _fair_truncate(tokenizer, doc, code, max_input_tokens):
    ''' truncate two samples to a fair length, 
    but include as much as possible
    '''
    # optimize for fair length between doc/code tokens
    # grow sequence sizes fairly, until max_input_tokens is reached
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
                truncation=False)


        opt_steps += 1
        if opt_steps > 1000:
            raise RuntimeError('too many steps taken during fair tokenization')
        
        if len(tok['input_ids']) >= max_input_tokens:
            return tr_sample
        elif max_dlength <= max_clength and max_dlength < len(doc):
            max_dlength = int(max_dlength*1.2)+1
        elif max_clength <= max_dlength and max_clength < len(code):
            max_clength = int(max_clength*1.2)+1
        else:
            # both doc and code fully fit
            return doc + tokenizer.sep_token + code


def tokenize_train_sample(tokenizer, sample, mode, max_input_tokens):
    doc = " ".join(sample['func_documentation_tokens'])
    code = " ".join(sample['func_code_tokens'])

    if mode == PreprocessMode.CONCAT:
        trunc_sample = _fair_truncate(tokenizer, doc, code, max_input_tokens)
        tokenized_sample = tokenizer(trunc_sample, padding='max_length', max_length=max_input_tokens, truncation=True)
    else:
        raise RuntimeError(f"preprocess mode: {mode} is not supported")

    proc_batch = dict()

    for k, v in tokenized_sample.items():
        proc_batch[k] = v

    return proc_batch


def tokenize_valid_sample(tokenizer, sample, mode, max_input_tokens):
    doc = " ".join(sample['func_documentation_tokens'])
    code = " ".join(sample['func_code_tokens']) + tokenizer.sep_token

    tokenized_code = tokenizer(code, padding='max_length', max_length=max_input_tokens, truncation=True)
    tokenized_doc = tokenizer(doc, padding='max_length', max_length=max_input_tokens, truncation=True)

    proc_sample = dict()

    for k, v in tokenized_code.items():
        proc_sample['code_'+k] = v

    for k, v in tokenized_doc.items():
        proc_sample['doc_'+k] = v

    return proc_sample


def preprocess_split(split, config, limit_samples=None, dataset=None, tokenizer=None, **kwargs):
    if not kwargs.get('progress', False) == True:
        datasets.disable_progress_bar()
    #datasets.set_progress_bar_enabled(kwargs.get('progress', False))

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'].value)

    num_proc = config['preprocess_cores']

    if split == DataSplit.EVAL:
        if dataset is None:
            dataset = load_dataset("code_search_net", split='all')
        else:
            dataset = dataset#['all']
    else:
        if dataset is None:
            dataset = load_dataset("code_search_net")[split.value]
        else:
            dataset = dataset[split.value]

    dataset.shuffle(0)

    if limit_samples is not None:
        if limit_samples < 1:
            raise RuntimeError('invalid limit_samples')
        dataset = dataset.select(np.random.choice(np.arange(len(dataset)), size=limit_samples, replace=False))

    languages = config['languages']
    if languages != ['all']:
        dataset = dataset.filter(lambda x: x['language'] in languages, num_proc=num_proc)

    dataset = dataset.rename_column("func_code_url", "url")

    if split == DataSplit.TRAIN:
        tfn = tokenize_train_sample
    else:
        tfn = tokenize_valid_sample
        
    def tokenize_fn(x): 
        return tfn(tokenizer, 
                x, 
                config['preprocess_mode'], 
                config['max_input_tokens'])

    dataset = dataset.map(
            tokenize_fn,
            remove_columns=set(dataset.column_names) - {'url', 'language'},
            batched=False,
            num_proc=num_proc)

    # cast dataset to torch tensors
    dataset.set_format(type='torch', columns=set(dataset.column_names) - {'url','language'}, output_all_columns=True)

    return dataset


def get_loader(dataset, batch_size, workers=0, dist=False, **kwargs):
    # convenience function 

    if dist == True:
        sampler = DistributedSampler(dataset, drop_last=True, shuffle=False)
    else:
        sampler=None

    loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            drop_last=True, 
            pin_memory=True, 
            num_workers=workers, 
            #multiprocessing_context='spawn',
            persistent_workers=workers > 0, 
            sampler=sampler)

    return loader


def deviceloader(loader, device):
    ''' wraps iterable to load samples to device'''
    for b in loader:
        for k, v in b.items():
            if isinstance(v, Tensor):
                b[k] = v.to(device)
        yield b
