import torch

config = {}

if torch.cuda.is_available():
    config['device'] = torch.device('cuda')
else:
    config['device'] = torch.device('cpu')

# experiment name
config['name'] = 'profiling_1'
config['batch_size'] = 3
config['infer_batch_size'] = 3

config['epochs'] = 1
#config['shards'] = 10 
config['shards'] = 10
config['grad_accum'] = 64 // config['batch_size']

# counted in batches, not in optimizer steps, because of grad_accum
config['warmup_batches'] = 1
# temperature/ weighting of cosine sim
# taken from simcse
config['temp'] = 0.05

config['embedding_size'] = 128
config['top_k'] = 5

config['logdir'] = './output'
config['checkpoint_dir'] = './output'
config['max_checkpoints'] = 1

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

config['max_input_tokens'] = 512

config['run_type'] = 'debug'
#config['run_type'] = 'profile'
#config['run_type'] = 'default'

config['optim'] = 'adam'

# set to No
#config['grad_clip'] = 1.0

config['precision'] = 'mixed'
