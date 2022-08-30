import pytest
import torch

from secora.models import *
#from secora.train import *
from secora.train import build_config, parse_args, rng_init, training_worker, build_optimizer, build_scheduler
from secora.losses import contrastive_loss
from secora.tracking import *

from torch.cuda.amp import autocast, GradScaler

import tempfile
import yaml

example_config = '''
logdir: "/tmp/secora_output/"
checkpoint_dir: "/tmp/secora_output/"

# the name has to follow the naming scheme
# see secora/train.py
name: 'run_secora_t0_utc00'
seed: 42
max_checkpoints: 10

num_gpus: 1 #'auto'
amp: disable #default
cuda_graphs: False

preprocess_cores: 10
preprocess_mode: concat
max_input_tokens: 256
languages:
  - python

epochs: 2
shards: 2
warmup_batches: 10

finetune_mode: all

#model_name: 'microsoft/codebert-base'
model_name: 'distilroberta-base'
#model_name: 'testroberta'
optimizer: adam
lr_schedule: constant

learning_rate: 1e-6
batch_size: 2
grad_accum: 1 # counted in batches
grad_cache: 1 # counted in batches
temp: 0.05
dropout: 0.1
grad_clip: 0.0

embedding_size: 2
top_k:  2 #1000
'''

@pytest.fixture
def get_model_inputs():
    input_ids = torch.ones([1,512], dtype=torch.int64)
    attention_mask = torch.zeros([1,512], dtype=torch.int64)
    inputs  = input_ids, attention_mask
    return inputs


@pytest.fixture
def get_model_input_dict():
    input_ids = torch.ones([512], dtype=torch.int64)
    attention_mask = torch.zeros([512], dtype=torch.int64)

    inputs  = {'input_ids':input_ids, 'attention_mask': attention_mask}
    return inputs


@pytest.fixture
def get_mock_trainloader(get_model_input_dict):
    return torch.utils.data.DataLoader(
            [ get_model_input_dict ] * 100,
            batch_size=2
            )


@pytest.mark.slow
def test_embeddingmodel():
    embsize = 128

    model = EmbeddingModel(BaseModel.CODEBERT, embsize)
    input_ids = torch.ones([1,512], dtype=torch.int64)
    attention_mask = torch.zeros([1,512], dtype=torch.int64)
    outputs = model(input_ids, attention_mask)
    assert outputs.shape == torch.Size([1, embsize])
    # normalization of output vectors
    assert all(torch.mean(outputs, dim=-1) < 1.)


class BaseModel(Enum):
    CODEBERT = 'microsoft/codebert-base'
    ROBERTA = 'roberta-base'
    DISTILROBERTA = 'distilroberta-base'


@pytest.mark.slow
@pytest.mark.cuda
def test_train_shard_simple_step():
    step = 0
    rank = 0
    batch = {"input_ids": torch.ones([1,256], dtype=torch.int64, device=rank),
        "attention_mask": torch.ones([1,256], dtype=torch.int64, device=rank)}

    model = EmbeddingModel(BaseModel.DISTILROBERTA, 768, hidden_dropout_prob=0.5).to(rank)
    optim = torch.optim.Adam(model.parameters())

    forward_1 = model
    forward_2 = model
    loss_fn = lambda a,b: contrastive_loss(a, b, temp=0.05)

    model_inputs = batch['input_ids'], batch['attention_mask']
    emb1 = forward_1(*model_inputs)
    emb2 = forward_2(*model_inputs)
    #closs = contrastive_loss(emb1, emb2, temp=config['temp'])
    closs = loss_fn(emb1, emb2)
    loss = closs
    loss.backward()
    optim.step()
    optim.zero_grad()


class MockScheduler:
    def step(self,):
        pass
    def get_last_lr(self):
        return [0.1]


@pytest.mark.slow
@pytest.mark.cuda
@pytest.mark.skip('nan, under investigation')
def test_train_step(get_mock_trainloader):
    cache = GCache()
    shard_step = 0
    shard_done = False
    step = 0
    rank = 0

    model = EmbeddingModel(BaseModel.DISTILROBERTA, 768, hidden_dropout_prob=0.5).to(rank)
    optim = torch.optim.Adam(model.parameters())

    config = {'amp': AMP.DISABLE, 'temp': 0.1, 'grad_clip': 0, 'batch_size': 2, 'grad_cache':1}


    state_tracker = {
            'optimizer': optim, 
            'scheduler': MockScheduler(),
            'model': model,
            'training_progress': TrainingProgress,
            'scaler': GradScaler()
            }

    kwargs = {'rank': 0, 'distributed': False}

    forward_1 = model
    forward_2 = model
    loss_fn = lambda a,b: contrastive_loss(a, b, temp=0.05)

    loss, shard_done = train_step(
            deviceloader(get_mock_trainloader, rank),
            state_tracker,
            config,
            **kwargs)

    assert loss.to('cpu').detach().numpy() > 0
    assert shard_done == False


@pytest.mark.slow
@pytest.mark.cuda
def test_train_shard_grad():
    step = 0
    rank = 0
    batch = {"input_ids": torch.ones([1,256], dtype=torch.int64, device=rank),
        "attention_mask": torch.ones([1,256], dtype=torch.int64, device=rank)}

    model = EmbeddingModel(BaseModel.DISTILROBERTA, 768, hidden_dropout_prob=0.5).to(rank)
    optim = torch.optim.Adam(model.parameters())

    forward_1 = model
    forward_2 = model
    loss_fn = lambda a,b: contrastive_loss(a, b, temp=0.05)

    model_inputs = batch['input_ids'], batch['attention_mask']
    emb1 = forward_1(*model_inputs)
    emb2 = forward_2(*model_inputs)
    #closs = contrastive_loss(emb1, emb2, temp=config['temp'])
    closs = loss_fn(emb1, emb2)
    loss = closs
    loss.backward()
    optim.step()
    optim.zero_grad()


# just try to run the mainloop on a single gpu
@pytest.mark.slow
@pytest.mark.cuda
@pytest.mark.skip(reason='currently broken, creating tests to find error')
def test_train_main():
    example_config_yaml = yaml.safe_load(example_config)

    with tempfile.NamedTemporaryFile('w') as tmpconf:
        with tempfile.TemporaryDirectory() as tmpdirname:
            example_config_yaml['logdir'] = tmpdirname
            example_config_yaml['checkpoint_dir'] = tmpdirname
            yaml.safe_dump(example_config_yaml, tmpconf)

            testargs = ['/root/secora/secora/train.py', tmpconf.name, '--name=distilroberta', '--debug']
            main(testargs)


class MockLogger:
    def __init__(self):
        pass
    def info(self, *args, **kwargs):
        pass
    def warning(self, *args, **kwargs):
        pass
    def debug(self, *args, **kwargs):
        pass


def load_main_state(config, **kwargs):
    # unimportant value
    trainset_len = 1000

    # initialize state
    m = build_model(config, **kwargs)
    #model = distribute_model(m, **kwargs)
    model = m
    optim = build_optimizer(config, model, **kwargs)
    scheduler = build_scheduler(optim, config, trainset_len, **kwargs)
    scaler = GradScaler()
    training_progress = TrainingProgress()

    state_tracker = StateTracker(
            config['name'],
            config['logdir'],
            config['max_checkpoints'],
            MockLogger(),
            model=model,
            optimizer=optim,
            scheduler=scheduler,
            scaler=scaler,
            training_progress=training_progress)

    return state_tracker

def run_training(argv, *args, **kwargs):
    cli_args = parse_args(argv)
    timestamp = '1'
    config = build_config(config_id=timestamp, args=cli_args)
    rng_init(seed=config['seed'])
    torch.backends.cudnn.benchmark = True

    #mp.set_start_method('spawn')
    mp.spawn(training_worker, 
            args=(config, cli_args.progress, cli_args.debug, args, kwargs),
            nprocs = config['num_gpus'],
            join=True)
    state = load_main_state(config, rank=0, logger=MockLogger())

    return state


# just try to run the mainloop on a single gpu
@pytest.mark.slow
@pytest.mark.cuda
def test_train_main_determinism():
    with tempfile.NamedTemporaryFile('w') as tmpconf:
        with tempfile.TemporaryDirectory() as tmpdirname:
            example_config_yaml = yaml.safe_load(example_config)
            example_config_yaml['logdir'] = tmpdirname
            example_config_yaml['checkpoint_dir'] = tmpdirname
            yaml.safe_dump(example_config_yaml, tmpconf)

            testargs = ['/root/secora/secora/train.py', tmpconf.name, '--debug']
            state1 = run_training(testargs, logger=MockLogger())

    # start a exactly similarly configured run
    # but in a clean logdir
    with tempfile.NamedTemporaryFile('w') as tmpconf:
        with tempfile.TemporaryDirectory() as tmpdirname:
            example_config_yaml = yaml.safe_load(example_config)
            example_config_yaml['logdir'] = tmpdirname
            example_config_yaml['checkpoint_dir'] = tmpdirname
            yaml.safe_dump(example_config_yaml, tmpconf)

            testargs = ['/root/secora/secora/train.py', tmpconf.name, '--debug']
            state2 = run_training(testargs, logger=MockLogger())


    s1 = state1['model'].state_dict() 
    s2 = state2['model'].state_dict()

    torch.use_deterministic_algorithms(mode=True)
    assert all([torch.allclose(x,y) for x,y in zip(s1.values(), s2.values())])
    #assert all([a==b for a,b in zip(s1.items(), s2.items())])
