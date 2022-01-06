import torch
from torch import profiler
from torch.utils.data import DataLoader
import os

from model import EmbeddingModel
from data import preprocess_split
from losses import contrastive_loss



from SM3 import SM3

def test_step(model, optim, batch, config):
    loss = contrastive_loss(model, batch, config)
    loss.backward()

    s_loss = loss.detach().cpu().numpy()

    optim.step()
    optim.zero_grad(set_to_none=True)


def profile(config):
    model = EmbeddingModel(config)
    model = model.to(config['device'])

    if config['finetune_mode'] == 'all':
        params = model.parameters()
    elif config['finetune_mode'] == 'pooling':
        params = model.pooling.parameters()
    else:
        raise RuntimeError('finetune_mode has to be: all or pooling')

    if config['optim'] == 'adam':
        optim = torch.optim.Adam(params, lr=config['lr'])
    elif config['optim'] == 'sgd':
        optim = torch.optim.SGD(params, lr=config['lr'])
    elif config['optim'] == 'sm3':
        optim = SM3(params, lr=config['lr'])
    else:
        raise RuntimeError('config specifies and unsupported optimizer')

    train_set = preprocess_split('train', config)
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=False, drop_last=True, pin_memory=True, num_workers=4, persistent_workers=True, prefetch_factor=10)

    it = iter(train_loader)
    test_step(model, optim, next(it), config)

    tensorboard_run_path = os.path.join(config['logdir'], config['name'])
    trace_path = os.path.join(config['logdir'], config['name'], 'profile_trace.json')
    stacks_path = os.path.join(config['logdir'], config['name'], 'profile_stacks.txt')

    with profiler.profile(
            with_stack=True, 
            profile_memory=True, 
            record_shapes=True,
            on_trace_ready=profiler.tensorboard_trace_handler(tensorboard_run_path),
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ]) as p:
        test_step(model, optim, next(it), config)

    print(p.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))
