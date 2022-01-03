from itertools import cycle

from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer



def preproc_valid(sample):
    # delete docstring from code samples
    return {'func_code_string': sample['func_code_string'].replace(sample['func_documentation_string'], '')}


def tokenize_train_sample(tokenizer, batch):
    #whole = batch['whole_func_string']
    # <CODESPLIT> is also used by codebert
    whole = [a+b+c for a,b,c in zip(batch['func_documentation_string'], cycle(['<CODESPLIT>']),  batch['func_code_string'])]
    tokenized_code = tokenizer(whole, padding='max_length', truncation=True, return_token_type_ids=True)
    url = batch['func_code_url']

    for k, v in tokenized_code.items():
        batch['proc_whole_'+k] = v

    batch['proc_url'] = url
    return batch


def tokenize_valid_sample(tokenizer, batch):
    code = batch['func_code_string']
    doc = batch['func_documentation_string']

    url = batch['func_code_url']

    # tokenizer is called twice, so that its impossible to leak data between code and doc
    # instead of the joint tokenization
    tokenized_code = tokenizer(code, padding='max_length', truncation=True, return_token_type_ids=True)
    tokenized_doc = tokenizer(doc, padding='max_length', truncation=True, return_token_type_ids=True)

    for k, v in tokenized_code.items():
        batch['proc_code_'+k] = v

    for k, v in tokenized_doc.items():
        batch['proc_doc_'+k] = v

    batch['proc_url'] = url

    return batch


def get_dataloaders(config):
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    dataset = load_dataset("code_search_net")
    if config['dryrun'] == True:
        train_set = dataset['train'].select(range(256))
        valid_set = dataset['validation'].select(range(256)).map(preproc_valid, batched=False)
    else:
        #train_set = dataset['train']
        #valid_set = dataset['validation'].map(preproc_valid, batched=False)
        train_set = dataset['train'].filter(lambda x: x['language'] == 'python')
        valid_set = dataset['validation'].filter(lambda x: x['language'] == 'python').map(preproc_valid, batched=False)

    # optional:
    # write a custom bucketing data collator to remove the need for padding and truncation
    # resulting in dynamic length sequences

    # preprocess tokenize the dataset once
    # by using batched, the tokenizer automatically pads and truncates to the same length
    train_set_tokenized = train_set.map(
            lambda x: tokenize_train_sample(tokenizer, x),
            remove_columns=train_set.column_names,
            batched=True)

    valid_set_tokenized = valid_set.map(
            lambda x: tokenize_valid_sample(tokenizer, x),
            remove_columns=valid_set.column_names,
            batched=True)
    ##
    # cast dataset to torch tensors
    train_set_tokenized.set_format(type='torch', columns=set(train_set_tokenized.column_names) - {'proc_url'})
    valid_set_tokenized.set_format(type='torch', columns=set(valid_set_tokenized.column_names) - {'proc_url'})
    ##

    # create the batched fast dataloader
    # (only possible with same length batches)
    train_loader = DataLoader(train_set_tokenized, batch_size=config['batch_size'], shuffle=True, drop_last=True)

    # don't shuffle validation set!
    valid_loader = DataLoader(valid_set_tokenized, batch_size=config['batch_size'], shuffle=False, drop_last=True)

    return train_loader, valid_loader

