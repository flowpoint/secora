import decimal
import torch
import yaml

import argparse

required_keys = '''
hostname
port
name
batch_size
seed
epochs
shards
grad_accum
warmup_batches
temp
top_k
checkpoint_dir
max_checkpoints
model_name
learning_rate
finetune_mode
languages
preprocess_cores
preprocess_mode
max_input_tokens
optimizer
precision
lr_schedule
dropout
'''

def check_config(config):
    if not isinstance(config, dict):
        raise ValueError("the passed object is not a dict")

    for k in required_keys.strip().split('\n'):
        if not k.strip() in config:
            raise ValueError(f'missing value in config: {k}')


def load_config(path):
    ''' this verifies and translates the config yaml file 
    to a valid training setup
    '''

    with open(path, 'r') as f:
        yconfig = yaml.safe_load(f)

    check_config(yconfig)

    config = dict()
    config.update(yconfig)
    config['learning_rate'] = float(decimal.Decimal(yconfig['learning_rate']))
    config['optimizer'] = yconfig['optimizer']

    if yconfig['num_gpus'] == 'auto':
        config['num_gpus'] = int(torch.cuda.device_count())
    elif torch.cuda.device_count() >= int(yconfig['num_gpus']) >= 0 and torch.cuda.is_available():
        pass
    else:
        raise ValueError('requested num_gpus not available')


    if not 'checkpoint_dir' in yconfig or yconfig['checkpoint_dir'] == "":
        raise ValueError('checkpoint dir must be specified')

    if not 'logdir' in yconfig or yconfig['logdir'] == "":
        raise ValueError('checkpoint dir must be specified')

    #if yconfig['precision'] == 'mixed' and int(config['num_gpus']) == 0:
    #    raise RuntimeError('cant use cuda amp mixed on cpu')

    if not isinstance(config['cuda_graphs'], bool):
        raise RuntimeError('cuda_graphs has to be bool')


    return config


def overwrite_config(args, config):
    if args.run_name is not None and args.run_name != "":
        config['name'] = args.run_name

    if args.batch_size is not None:
        config['batch_size'] = args.batch_size

    check_config(config)

    return config

def save_config(config, path):
    with open(path, 'w') as f:
        f.write(yaml.dump(config))


if __name__ == "__main__":
    desc = 'config utility, by default, checks config validity'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        yconfig = yaml.safe_load(f)

    check_config(yconfig)
    print('config seems valid')
