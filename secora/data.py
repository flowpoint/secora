from itertools import cycle

from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

from torch.utils.data.distributed import DistributedSampler



def preproc_valid(sample):
    # delete docstring from code samples
    return {'func_code_string': sample['func_code_string'].replace(sample['func_documentation_string'], '')}


def tokenize_train_sample(tokenizer, batch, config):
    mode = config['preprocess_mode']
    if mode == 'joint':
        whole = batch['whole_func_string']
    elif mode == 'concat':
        # <CODESPLIT> is also used by codebert
        whole = [a+b+c for a,b,c in zip(batch['func_documentation_string'], cycle(['<CODESPLIT>']),  batch['func_code_string'])]
    else:
        raise RuntimeError(f"preprocess mode: {mode} is not supported")

    tokenized_code = tokenizer(whole, padding='max_length', max_length=config['max_input_tokens'], truncation=True, return_token_type_ids=True)

    proc_batch = dict()

    for k, v in tokenized_code.items():
        proc_batch[k] = v

    return proc_batch


def tokenize_valid_sample(tokenizer, batch, config):
    code = batch['func_code_string']
    doc = batch['func_documentation_string']

    # call tokenizer twice instead of on a pair, so that its impossible to leak data between code and doc
    # instead of the joint tokenization
    tokenized_code = tokenizer(code, padding='max_length', max_length=config['max_input_tokens'], truncation=True, return_token_type_ids=True)
    tokenized_doc = tokenizer(doc, padding='max_length', max_length=config['max_input_tokens'], truncation=True, return_token_type_ids=True)

    proc_batch = dict()

    for k, v in tokenized_code.items():
        proc_batch['code_'+k] = v

    for k, v in tokenized_doc.items():
        proc_batch['doc_'+k] = v

    return proc_batch


def preprocess_split(split, config, limit_samples=-1):
    if split not in ["train", "validation"]:
        raise RuntimeError(f"invalid dataset split: {split}")

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    dataset = load_dataset("code_search_net")[split]
    if limit_samples >= 1:
        dataset = dataset.select(range(limit_samples))

    dataset = dataset.filter(lambda x: x['language'] in config['languages'], num_proc=config['preprocess_cores'])

    if split == "validation":
        dataset = dataset.map(preproc_valid, batched=False, num_proc=config['preprocess_cores'])

    dataset = dataset.rename_column("func_code_url", "url")

    # optional:
    # write a custom bucketing data collator to remove the need for padding and truncation
    # resulting in dynamic length sequences

    # preprocess tokenize the dataset once
    # by using batched, the tokenizer automatically pads and truncates to the same length
    if split == "train":
        def tokenize_fn(x): return tokenize_train_sample(tokenizer, x, config)
    else:
        def tokenize_fn(x): return tokenize_valid_sample(tokenizer, x, config)

    dataset = dataset.map(
            tokenize_fn,
            remove_columns=set(dataset.column_names) - {'url'},
            batched=True,
            num_proc=config['preprocess_cores'])

    # cast dataset to torch tensors
    dataset.set_format(type='torch', columns=set(dataset.column_names) - {'url'}, output_all_columns=True)

    return dataset


def get_train_loader(shard, config, persistent=True):
    train_sampler = DistributedSampler(shard, drop_last=True, shuffle=False)
    train_loader = DataLoader(
            shard, 
            batch_size=config['batch_size'], 
            shuffle=False,
            drop_last=True, 
            pin_memory=True, 
            # workers need to use the spawn or forkserver method in a distributed setting
            num_workers=1, 
            multiprocessing_context='spawn',
            persistent_workers=persistent, 
            sampler=train_sampler)

    return train_loader
