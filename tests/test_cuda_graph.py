import unittest
import os
import sys


import torch
from torch import tensor
import torch.nn.functional as F
import numpy as np

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from secora.config import load_config, overwrite_config
from secora.model import *

from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler

#from secora.infer import *

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def lossfn(model, dummies):
    #emb1 = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    #emb2 = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    #biemb = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    biemb = model(*dummies)
    emb1 = biemb[:,0]
    emb2 = biemb[:,1]

    # use the exact same loss from the simcse repository
    sim = F.cosine_similarity(emb1.unsqueeze(1), emb2.unsqueeze(0), dim=-1) / 0.05
    labels = torch.arange(sim.size(0), dtype=torch.int64, device=sim.get_device())
    loss = F.cross_entropy(sim, labels)

    return loss


def datagen():
    for i in range(1000000):
        dummy_inputs = (
            torch.randint(8,1000, [200], dtype=torch.int64),
            torch.randint(8, 1000, [200], dtype=torch.int64),
            torch.randint(8, 1000, [200], dtype=torch.int64)
            )

        yield dummy_inputs

def train(config):
    rank = dist.get_rank()

    dummy_inputs = (
        torch.zeros([8,200], dtype=torch.int64, device=rank),
        torch.zeros([8,200], dtype=torch.int64, device=rank),
        torch.zeros([8,200], dtype=torch.int64, device=rank)
        )

    m = BiEmbeddingModel(config).to(rank)
    #torch.cuda.make_graphed_callables(m, dummy_inputs)
    m.make_graphed(dummy_inputs)
    l = m(*dummy_inputs).sum()
    l.backward()

    model = DDP(m, device_ids=[rank])
    model.train()

    params = model.parameters()
    optim = torch.optim.Adam(params, lr=config['lr'])

    scaler = GradScaler()

    scheduler = get_linear_schedule_with_warmup(
            optim,
            num_warmup_steps=2,
            num_training_steps=98)

    dset = list(datagen())

    sampler = DistributedSampler(dset, drop_last=True, shuffle=False)
    loader = DataLoader(
            dset,
            batch_size=2000,
            shuffle=(sampler is None), 
            drop_last=True, 
            pin_memory=True, 
            # workers need to use the spawn or forkserver method in a distributed setting
            num_workers=1, 
            multiprocessing_context='spawn',
            persistent_workers=True,
            sampler=sampler)

    losscount = torch.zeros([1], device=rank)

    for b in loader:
        model.train()
        with model.no_sync():
            l2 = scaler.scale(lossfn(model, b).sum())
            l2.backward()
            scheduler.step()
            dist.all_reduce(l2)
            losscount += l2

        l2 = scaler.scale(lossfn(model, b).sum())
        l2.backward()
        scheduler.step()

        scaler.step(optim)
        scaler.update()
        #optim.step()

        dist.all_reduce(l2)
        losscount.add_(l2.detach())

        optim.zero_grad(set_to_none=True)

        if rank == 0:
            print(losscount.detach().cpu())

    '''
    for b in loader:
        with model.no_sync():
            with torch.no_grad():
                code_embedding = build_embedding_space(model, valid_loader, config, feature_prefix='code_', embedding_size=config['embedding_size'], device=rank)
    '''


def training_worker(rank, config):
    world_size = config['num_gpus']

    host_name = config['hostname']
    port = config['port']

    os.environ['MASTER_ADDR'] = host_name
    os.environ['MASTER_PORT'] = str(port)
    
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    
    # initialize the process group
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    train(config)


def main():
    config = load_config('configs/cluster.yml')

    np.random.seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.cuda.empty_cache()

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = True

    mp.set_start_method('spawn')
    mp.spawn(training_worker, 
            args=(config,),
            nprocs = config['num_gpus'],
            join=True)


class TestExample(unittest.TestCase):
    def test_example(self):
        main()
