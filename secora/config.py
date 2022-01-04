import torch

config = {}

if torch.cuda.is_available():
    config['device'] = torch.device('cuda')
else:
    config['device'] = torch.device('cpu')

# experiment name
config['name'] = 'test10'
config['dryrun'] = False
config['batch_size'] = 3
config['infer_batch_size'] = 3
#config['lr'] = 1e-5


config['epochs'] = 1 #4096*4 #2**16
config['shards'] = 1000 #4096*4 #2**16
config['grad_accum'] = 64 // config['batch_size']
# temperature/ weighting of cosine sim
# taken from simcse
config['temp'] = 0.05

config['embedding_size'] = 128
config['top_k'] = 5

config['checkpoint_dir'] = './output/runs'
config['max_checkpoints'] = 1

#config['model_name'] = 'huggingface/CodeBERTa-small-v1'
config['model_name'] = 'microsoft/codebert-base'
#model_name = 'bert-base-cased'
#model_name = 'bert-large-cased'

#config['lr'] = tune.sample_from(lambda spec: 10**(-10 * np.random.rand()))
config['lr'] = 1e-5
#config['momentum'] = tune.uniform(0.1,0.9)

#config['finetune_mode'] = 'pooling'
config['finetune_mode'] = 'all'

config['languages'] = ['python']

config['preprocess_cores'] = 24
config['preprocess_mode'] = 'concat'

config['max_input_tokens'] = 512

config['run_type'] = 'debug'
