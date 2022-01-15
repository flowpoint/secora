import decimal
import torch
import yaml

import argparse

required_keys = '''
hostname
port
name
batch_size
infer_batch_size
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
'''

def check(config):
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

    check(yconfig)

    config = dict()
    config.update(yconfig)
    config['lr'] = float(decimal.Decimal(yconfig['learning_rate']))
    config['optim'] = yconfig['optimizer']

    if yconfig['num_gpus'] == 'auto':
        config['num_gpus'] = torch.cuda.device_count()
    elif torch.cuda.device_count() >= int(yconfig['num_gpus']) >= 0 and torch.cuda.is_available():
        pass
    else:
        raise ValueError('requested num_gpus not available')


    if not 'checkpoint_dir' in yconfig or yconfig['checkpoint_dir'] == "":
        raise ValueError('checkpoint dir must be specified')

    if not 'logdir' in yconfig or yconfig['logdir'] == "":
        raise ValueError('checkpoint dir must be specified')

    if yconfig['precision'] == 'mixed' and int(config['num_gpus']) == 0:
        raise RuntimeError('cant use cuda amp mixed on cpu')


    return config

if __name__ == "__main__":
    desc = 'config utility, by default, checks config validity'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        yconfig = yaml.safe_load(f)

    check(yconfig)
    print('config seems valid')

'''
config = {}
config['hostname'] = 'localhost'
config['port'] = '12355'

# experiment name
config['name'] = 'muli_gpu_profiling3'
config['batch_size'] = 8
config['infer_batch_size'] = 8

config['seed'] = 42

config['epochs'] = 1
config['shards'] = 20

config['grad_accum'] = grad_accum // config['batch_size']

# counted in batches, not in optimizer steps, because of grad_accum
config['warmup_batches'] = 10000
# temperature/ weighting of cosine sim
# taken from simcse
config['temp'] = 0.05

config['embedding_size'] = 128
config['top_k'] = 5


config['logdir'] = '~/secora_output'
config['checkpoint_dir'] = '~/secora_output'

config['max_checkpoints'] = 10

#config['model_name'] = 'huggingface/CodeBERTa-small-v1'
config['model_name'] = 'microsoft/codebert-base'
#config['model_name'] = 'roberta-large'
#model_name = 'bert-base-cased'
#model_name = 'bert-large-cased'

#config['lr'] = tune.sample_from(lambda spec: 10**(-10 * np.random.rand()))
config['lr'] = 1e-5
#config['momentum'] = tune.uniform(0.1,0.9)

#config['finetune_mode'] = 'pooling'
config['finetune_mode'] = 'all'

config['languages'] = ['python']

config['preprocess_cores'] = 10
config['preprocess_mode'] = 'concat'

config['max_input_tokens'] = 256

#config['run_type'] = 'debug'
#config['run_type'] = 'profile'
config['run_type'] = 'default'

config['optim'] = 'adam'

# set to No
#config['grad_clip'] = 1.0

config['precision'] = 'mixed'

'''
