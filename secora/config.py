import torch

config = {}

config['hostname'] = 'localhost'
config['port'] = '12355'

#config['num_gpus'] = 1
config['num_gpus'] = torch.cuda.device_count()

if config['num_gpus'] > 0 and not torch.cuda.is_available():
    raise RuntimeError('cuda is not available')

if config['num_gpus'] > torch.cuda.device_count():
    raise RuntimeError('num_gpus higher than number of available gpus')

# experiment name
config['name'] = 'muli_gpu_profiling1'
config['batch_size'] = 8
config['infer_batch_size'] = 8

config['seed'] = 42

config['epochs'] = 1
config['shards'] = 20
config['grad_accum'] = 64 // config['batch_size']

# counted in batches, not in optimizer steps, because of grad_accum
config['warmup_batches'] = 10000
# temperature/ weighting of cosine sim
# taken from simcse
config['temp'] = 0.05

config['embedding_size'] = 128
config['top_k'] = 5

#config['logdir'] = './output'
#config['checkpoint_dir'] = './output'

config['logdir'] = './output'
config['checkpoint_dir'] = './output'

config['max_checkpoints'] = 30

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

config['run_type'] = 'debug'
#config['run_type'] = 'profile'
#config['run_type'] = 'default'

config['optim'] = 'adam'

# set to No
#config['grad_clip'] = 1.0

config['precision'] = 'mixed'

if config['precision'] == 'mixed' and config['num_gpus'] == 0:
    raise RuntimeError('cant use cuda amp mixed on cpu')
