import os
import sys
import time
from time import time
from enum import Enum, auto
from dataclasses import dataclass, field

from math import ceil
import argparse
import datetime
import re

from pdb import set_trace as bp

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
from torch.optim import lr_scheduler

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from secora.models import *
import secora.models as models

from secora.data import *
from secora import data
from secora.config import *
from secora.infer import validate
from secora.losses import contrastive_loss, mrr
import secora.losses as losses #.LOSS_TEMPERATURE
from secora.tracking import * # init_storage
from secora.display import Display
from secora.metrics import MetricLogger

import datasets
from SM3 import SM3

import grad_cache.functional

TIMEOUT = datetime.timedelta(65)

class FinetuneMode(Enum):
    ALL = 'all'
    POOLING = 'pooling'

class ScheduleEnum(Enum):
    LINEAR = 'linear'
    CONSTANT = 'constant'
    
class OptimizerEnum(Enum):
    ADAM = 'adam'
    ADAMW = 'adamw'
    SGD = 'sgd'
    SM3 = 'sm3'

_optimizer_mapping = {'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'sgd': torch.optim.SGD,
        'sm3': SM3
        }

class RunNameSetting(Setting):
    ''' enforces a naming scheme for training runs 
    prefix = run | test | debug | profile
    readable name = aname0withnumbers
    trial number = t0
    start timestamp rounded to upper second = utc1645359851
    example:
    name := prefix readable_name trial_number 
    run_aname0withnumbers_t0_utc1645359851
    '''

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.scheme = re.compile('(run|test|debug|profile)_[a-z][a-z0-9]*_t\d+_utc(\d)+\Z')

    @property
    def allowed_type(self):
        return str

    def check(self, val):
        matched = self.scheme.match(val) is not None
        return matched

class TrainingConfig(SimpleConfig):
    def __init__(self):
        super().__init__()

        setts = [
            IntSetting('batch_size', lb=1),
            IntSetting('seed', lb=0),
            IntSetting('epochs', lb=1),
            IntSetting('shards', lb=1),
            IntSetting('grad_accum', lb=1),
            IntSetting('grad_cache', lb=1),
            IntSetting('warmup_batches',lb=0),
            FloatSetting('temp', lb=0., ub=1.),
            IntSetting('top_k', lb=0),
            DirectorySetting('checkpoint_dir'),
            IntSetting('max_checkpoints', lb=0),
            models.BASEMODEL_SETTING,
            FloatSetting('learning_rate', lb=0.),
            EnumSetting('finetune_mode', FinetuneMode),
            data.LanguageSetting('languages'),
            IntSetting('preprocess_cores', lb=1),
            EnumSetting('preprocess_mode', data.PreprocessMode),
            IntSetting('max_input_tokens', lb=1),
            EnumSetting('optimizer', OptimizerEnum),
            models.AMP_SETTING,
            EnumSetting('lr_schedule', ScheduleEnum),
            models.DROPOUT_SETTING,
            RunNameSetting('name'),
            IntSetting('num_gpus', lb=0),
            BoolSetting('cuda_graphs'),
            IntSetting('embedding_size', lb=0),
            FloatSetting('grad_clip', lb=0.),
            DirectorySetting('logdir'),
            DirectorySetting('checkpoint_dir')
            ]

        for s in setts:
            self.add(s)

    def parse_from_dict(self, config_candidate):
        for k,v in config_candidate.items():
            # parse through the settings parsing function
            self[k] = self.settings[k].parse(v)

@dataclass
class GCache:
    cache_1: list = field(default_factory=list)
    cache_2: list = field(default_factory=list)
    closures_1: list = field(default_factory=list)
    closures_2: list = field(default_factory=list)


def gradcache_fns(forward1, forward2, loss_fn, optimize_fn, cache):
    ''' the cache is side effected '''
    def cfn1(*args, **kwargs):
        emb, closure = grad_cache.functional.cached(forward1)(*args, **kwargs)
        cache.cache_1.append(emb)
        cache.closures_1.append(closure)
        return emb

    def cfn2(*args, **kwargs):
        emb, closure = grad_cache.functional.cached(forward2)(*args, **kwargs)
        cache.cache_2.append(emb)
        cache.closures_2.append(closure)
        return emb

    def lfn(*args, **kwargs):
        loss = grad_cache.functional.cat_input_tensor(loss_fn)(*args, **kwargs)
        return loss

    def ofn(*args, **kwargs):
        for c, v in zip(cache.closures_1, cache.cache_1):
            c(v)
        for c, v in zip(cache.closures_2, cache.cache_2):
            c(v)

        res = optimize_fn(*args, **kwargs)

        cache.cache_1 = []
        cache.cache_2 = []
        cache.closures_1 = []
        cache.closures_2 = []

        return res

    return cfn1, cfn2, lfn, ofn 


def should_optimize(step, config):
    #cache_accum = 512 // config['batch_size']
    #return (step+1) % cache_accum == 0
    return True


def train_step(
        train_loader,
        state_tracker,
        config, 
        **kwargs):
    ''' the necessary accumulation/gradcache steps 
    until and including one optimizer step 

    note: the last values in a shard are omitted
    because they won't fit for a full optimizer/accumulation step/batch

    this can be alleviated by fitting the shard size or optimizer/accumulation,
    so that the accumulation divides the shard size
    '''

    optim = state_tracker['optimizer']
    scheduler = state_tracker['scheduler']
    training_progress = state_tracker['training_progress']
    scaler = state_tracker['scaler']
    model = state_tracker['model']
    rank = kwargs['rank']

    model.train()

    cache = GCache()
    forward_1 = forward_2 = model

    def loss_fn(a, b):
        return contrastive_loss(a, b, temp=config['temp'])

    def optimize():
        optim.step()
        optim.zero_grad(set_to_none=True)

    if config['grad_cache'] == True:
        fns = gradcache_fns(forward_1, forward_2, loss_fn, optimize, cache)
        forward_1, forward_2, loss_fn, optimize = fns

    step = 0
    step_loss = torch.tensor([0.], device=rank)

    while True:
        try:
            batch = next(deviceloader(train_loader, rank))
        except StopIteration as e:
            return step_loss, True

        model_inputs = batch['input_ids'], batch['attention_mask']
        with torch.no_grad():
            emb1 = forward_1(*model_inputs)
        emb2 = forward_2(*model_inputs)
        loss = loss_fn(emb1, emb2)

        if config['amp'] != AMP.DISABLE:
            loss = scaler.scale(loss)

        step_loss += loss
        loss.backward()
        step += 1
        
        if should_optimize(step, config) == True:
            break

    # only sync before optimizer step
    if kwargs.get('distributed', True) == True:
        torch.cuda.synchronize()
        dist.barrier()
    
    #gradient clipping
    if config['grad_clip'] != 0.:
        scaler.unscale_(optim)
        clip_grad_norm_(
                model.parameters(), 
                max_norm=config['grad_clip'])

    if config['amp'] != AMP.DISABLE:
        scaler.step(optim)
        scaler.update()
    else:
        optimize()

    scheduler.step()
    training_progress.optimizer_step += 1

    return step_loss, False


def train_shard(
        state_tracker,
        train_loader,
        config,
        grad_accum=1,
        **kwargs
        ):

    rank = kwargs.get('rank', 0)
    logger = kwargs['logger']
    scheduler = state_tracker['scheduler']
    training_progress = state_tracker['training_progress']
    display = kwargs['display']
    display.start_shard(len(train_loader))
    metriclogger = kwargs['metriclogger']

    shard_loss = torch.tensor([0.], device=rank, dtype=torch.float64, requires_grad=False)
    shard_step = 0
    shard_done = False

    while shard_done == False:
        loss, shard_done = train_step(
                deviceloader(train_loader, rank),
                state_tracker,
                config, 
                **kwargs)

        display.step_shard()
        shard_step += 1

        if rank == 0:
            metriclogger.add_scalar(
                    "learning_rate/train", 
                    scheduler.get_last_lr()[0], 
                    training_progress.optimizer_step)
            metriclogger.flush()

        shard_loss.add_(loss.detach())
        shard_step += 1

    dist.all_reduce(shard_loss)
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        avg_loss = shard_loss.cpu().numpy() / shard_step
        metriclogger.add_scalar("avg_loss/train", avg_loss, training_progress.optimizer_step)
        metriclogger.flush()

def hw_warmup(model, train_set, **kwargs):
    ''' 
    warmup hardware and accelerators
    collects data for lower level optimization algorithms
    done separately before training for reproducability with changing hw setups

    train_set needs to be without sideeffects when reading it
    '''

    logger = kwargs['logger']
    config = kwargs['config']
    rank = kwargs['rank']
    logger.info('warming up cuda benchmark')
    train_loader = get_loader(train_set, config['batch_size'], workers=0, dist=dist.is_initialized(), **kwargs)
    for step, batch in zip(range(12), deviceloader(train_loader, rank)):
        model_inputs = batch['input_ids'], batch['attention_mask']
        model(*model_inputs)
        torch.cuda.synchronize()
        dist.barrier()


def hw_optimize(model, **kwargs):
    '''
    use the warmup statistics to optimize performance before starting training
    '''

    config = kwargs['config']
    if config['cuda_graphs'] == True:
        logger.info('cuda_graphs is True: building the cuda graph')
        torch.cuda.synchronize()
        dist.barrier()
        model.make_graphed(model_inputs)
        torch.cuda.synchronize()
        dist.barrier()

def distribute_model(model, **kwargs):
    kwargs['logger'].info('building distributed model')
    return DDP(model, device_ids=[kwargs['rank']], find_unused_parameters=True)

def build_optimizer(config, model):
    mode = config['finetune_mode']
    if mode == FinetuneMode.ALL:
        params = model.parameters()
    elif mode == FinetuneMode.POOLING:
        params = model.pooling.parameters()
    else:
        raise RuntimeError(f'invalid finetune_mode {mode}')

    optim = _optimizer_mapping[config['optimizer'].value](params, lr=config['learning_rate'])
    return optim

def build_scheduler(optim, config, train_set_len):
    num_warmup_steps = ceil(config['warmup_batches'] / config['grad_accum'])
    num_training_steps_per_epoch = ceil(train_set_len / (config['batch_size'] * config['grad_accum']))
    num_training_steps = config['epochs'] * num_training_steps_per_epoch

    if config['lr_schedule'] == ScheduleEnum.LINEAR:
        scheduler = get_linear_schedule_with_warmup(
                optim,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps)
    elif config['lr_schedule'] == ScheduleEnum.CONSTANT:
        scheduler = get_constant_schedule_with_warmup(
                optim,
                num_warmup_steps=num_warmup_steps
                )

    return scheduler

def train(config, preempt_callback=None, **kwargs):
    rank = kwargs['rank']
    torch.autograd.set_detect_anomaly(True)
    #if rank == 0:
    #    writer = SummaryWriter(log_dir=os.path.join(config['logdir'], config['name']), flush_secs=30)

    logger = kwargs['logger']
    logger.info('started train function')

    if kwargs['debug'] == True:
        t_limit = 20*config['grad_accum']*config['batch_size']
        v_limit = 20*config['grad_accum']*config['batch_size']
    else:
        t_limit = 200000
        v_limit = 50000

    # training blocks
    train_set = preprocess_split(data.DataSplit.TRAIN, config, limit_samples=t_limit, **kwargs)
    valid_set = preprocess_split(data.DataSplit.VALIDATION, config, limit_samples=v_limit, **kwargs)

    # initialize state
    m = build_model(config, **kwargs)

    # this should soon be rather done after the training initialization
    hw_warmup(m, train_set, config=config, **kwargs)
    hw_optimize(m, config=config, **kwargs)

    model = distribute_model(m, **kwargs)
    optim = build_optimizer(config, model)
    scheduler = build_scheduler(optim, config, len(train_set))
    scaler = GradScaler()
    training_progress = TrainingProgress()
    state_tracker = StateTracker(
            config['name'],
            config['logdir'],
            config['max_checkpoints'],
            logger,
            model=model,
            optimizer=optim,
            scheduler=scheduler,
            scaler=scaler,
            training_progress=training_progress)

    if kwargs.get('debug', False) == True:
        num_epochs = 2
        num_shards = 2
        grad_accum = 1
    else:
        # load latest checkpoint 
        torch.cuda.synchronize()
        dist.barrier()
        state_tracker.load_latest()

        num_epochs = config['epochs']
        num_shards = config['shards']
        grad_accum = config['grad_accum']

    logger.info(f'shard_size: {len(train_set)//num_shards} samples')
    logger.info(f'validation set size: {len(valid_set)} samples')
    logger.info(f'starting training')

    # do one validation pass with the base model
    score = validate(state_tracker['model'], 
            valid_set, 
            config, 
            state_tracker['training_progress'],
            **kwargs)
            #writer, 

    while(training_progress.epoch < num_epochs):
        logger.info(f'starting epoch: {training_progress.epoch} of {num_epochs}')
        train_set.shuffle()

        while(training_progress.shard < num_shards):
            logger.info(f'training shard: {training_progress.shard}')

            shard = train_set.shard(num_shards, training_progress.shard, contiguous=True)
            train_loader = get_loader(shard, config['batch_size'])

            train_shard(
                state_tracker,
                train_loader,
                config,
                grad_accum=grad_accum,
                **kwargs
                )


            logger.info(f'validating shard {training_progress.shard}')

            score = validate(state_tracker['model'], 
                    valid_set, 
                    config, 
                    state_tracker['training_progress'],
                    **kwargs)
                    #writer, 

            training_progress.shard_done()

            torch.cuda.synchronize()
            dist.barrier()
            if rank == 0:
                state_tracker.save()

            if 'preempt_callback' in kwargs:
                kwargs['preempt_callback'](state_tracker, score, config, **kwargs)

        training_progress.epoch_done()

    if 'hparam_callback' in kwargs and rank == 0:
        kwargs['hparam_callback'](kwargs['metriclogger'], score)

    return score

def training_worker(rank, config, progress, debug):
    world_size = config['num_gpus']
    #os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    dist.init_process_group('nccl', rank=rank, world_size=world_size, timeout=TIMEOUT)
    torch.cuda.set_device(rank)

    init_storage(config, rank)
    logger = make_logger(config, debug=debug, rank=rank)
    display = Display(show_progress=progress, rank=rank)
    metriclogger = MetricLogger(config, rank)
    logger.info(f'starting run with name: {config["name"]}')
    train(config, display=display, metriclogger=metriclogger, progress=progress, debug=debug, logger=logger, rank=rank)

    torch.cuda.synchronize()
    dist.barrier(group=dist.group.WORLD)
    dist.destroy_process_group()


def rng_init(seed):
    torch.cuda.empty_cache()
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def build_config(config_id: str, args=None):
    ''' config id can be arbitrary, but must be unique in the config/training run store '''
    config = TrainingConfig()

    # this will need to be parallelism safe in future
    with args.config_file as f:
        config_candidate = yaml.safe_load(f)

    if args.name is not None:
        if args.debug == True:
            prefix = 'debug'
        else: 
            prefix = 'run'
            
        config_candidate['name'] = f'{prefix}_{args.name}_t0_utc{config_id}'

    if args.batch_size is not None:
        config_candidate['batch_size'] = args.batch_size

    if args.max_checkpoints is not None:
        config_candidate['max_checkpoints'] = args.max_checkpoints

    config.parse_from_dict(config_candidate)
    config.check()
    config.final()

    return config


def main(argv):
    args = parse_args(argv)

    timestamp = str(ceil(time()))
    config = build_config(config_id=timestamp, args=args)
    rng_init(seed=config['seed'])
    torch.backends.cudnn.benchmark = True

    mp.set_start_method('spawn')
    mp.spawn(training_worker, 
            args=(config, args.progress, args.debug),
            nprocs = config['num_gpus'],
            join=True)


def parse_args(argv):
    parser = argparse.ArgumentParser(description='manual training script.')
    parser.add_argument('config_file', type=argparse.FileType('r'))
    # these values override the config values if specified
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--max_checkpoints', type=int, default=None) 
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--progress', action='store_true', default=False)
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    main(sys.argv)
