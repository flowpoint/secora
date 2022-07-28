import yaml

import pytest
import torch

from tempfile import TemporaryDirectory

from secora.models import *
from secora.training_tasks import *
from secora.train import *

import torch.distributed as dist
import torch.multiprocessing as mp

@pytest.fixture
def get_model_inputs():
    input_ids = torch.ones([1,512], dtype=torch.int64)
    attention_mask = torch.zeros([1,512], dtype=torch.int64)

    inputs  = input_ids, attention_mask
    return inputs

@pytest.fixture
def get_config():
    f = "configs/default.config"
    config_candidate = yaml.safe_load(f)

    config = TrainingConfig()
    config.parse_from_dict(config_candidate)
    config.check()
    config.final()
    return config


@pytest.mark.slow
def test_embeddingmodel():
    embsize = 128

    model = EmbeddingModel(BaseModel.CODEBERT, embsize)
    input_ids = torch.ones([1,512], dtype=torch.int64)
    attention_mask = torch.zeros([1,512], dtype=torch.int64)
    outputs = model(input_ids, attention_mask)
    print('hello')
    assert outputs.shape == torch.Size([1, embsize])
    # normalization of output vectors
    print(torch.mean(outputs))
    assert all(torch.mean(outputs, dim=-1) < 1.)


def run_train(rank, config, params, t_limit):
    logdir = os.path.join(config['logdir'], config['name'])
    checkdir = os.path.join(config['checkpoint_dir'], config['name'])
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(checkdir, exist_ok=True)

    TIMEOUT = datetime.timedelta(65)
    dist.init_process_group('nccl', rank=rank, world_size=config['num_gpus'], timeout=TIMEOUT)

    t_args = training_setup(config, t_limit=t_limit, progress=True, debug=False)
    training_plan(*t_args)

    tracker = t_args[0]
    if dist.get_rank() == 0:
        params.append(tracker['model'].to('cpu').parameters())
    
    del(t_args)
    del(config)
    del(tracker)
    torch.cuda.clear_cache()
    

def getconf(dir_):
    with open("configs/default.yml", 'r') as f:
        config_candidate = yaml.safe_load(f)

    config_candidate['logdir'] = dir_
    config_candidate['checkpoint_dir'] = dir_

    config = TrainingConfig()
    config.parse_from_dict(config_candidate)
    config.check()
    config.final()
    return config

def clean_run(config, params, t_limit):

    # train for 100 samples
    clean_init(seed=config['seed'])

    mp.set_start_method('spawn')
    mp.spawn(run_train, 
            args=(config, params, t_limit),
            nprocs = config['num_gpus'],
            join=True)


@pytest.mark.slow
def test_train_resume():
    ''' integration test '''


    params = []

    with TemporaryDirectory() as dir_:
        config = getconf(dir_)
        clean_run(config, params, 100)

        config = getconf(dir_)
        clean_run(config, params, 100)

    # 
    with TemporaryDirectory() as d:
        clean_run(config, params, 200)

    assert params == params2
