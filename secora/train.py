import os
import time
from time import time
from enum import Enum, auto

from math import ceil
import argparse
import datetime
import re

import pdb

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
from torch.optim import lr_scheduler

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from .model import *

from .data import *
from . import data
from .config import *
from .infer import build_embedding_space, k_nearest_neighbors, validate
from .losses import contrastive_loss, mrr
from .tracking import *

from SM3 import SM3

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
        self.scheme = re.compile('(run|test|debug|profile)_\w\w*_t\d+_utc(\d)+\Z')

    @property
    def allowed_type(self):
        return str

    def check(self, val):
        matched = self.scheme.match(val) is not None
        return matched

class TrainingConfig(SimpleConfig):
    def __init__(self):
        super().__init__()

        setts = [IntSetting('batch_size', lb=1),
            IntSetting('seed', lb=0),
            IntSetting('epochs', lb=1),
            IntSetting('shards', lb=1),
            IntSetting('grad_accum', lb=1),
            IntSetting('warmup_batches',lb=0),
            FloatSetting('temp', lb=0., ub=1.),
            IntSetting('top_k', lb=0),
            DirectorySetting('checkpoint_dir'),
            IntSetting('max_checkpoints', lb=0),
            EnumSetting('model_name',BaseModel),
            FloatSetting('learning_rate', lb=0.),
            EnumSetting('finetune_mode', FinetuneMode),
            data.LanguageSetting('languages'),
            IntSetting('preprocess_cores', lb=1),
            EnumSetting('preprocess_mode', data.PreprocessMode),
            IntSetting('max_input_tokens', lb=1),
            EnumSetting('optimizer', OptimizerEnum),
            EnumSetting('amp', AMP),
            EnumSetting('lr_schedule', ScheduleEnum),
            FloatSetting('dropout', lb=0., ub=1),
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


def train_shard(
        state_tracker,
        train_loader,
        config,
        writer=None,
        **kwargs
        ):

    rank = dist.get_rank()

    logger = kwargs['logger']

    optim = state_tracker['optimizer']
    scheduler = state_tracker['scheduler']
    training_progress = state_tracker['training_progress']
    scaler = state_tracker['scaler']
    model = state_tracker['model']
    model.train()

    shard_loss = torch.tensor([0.], device=rank, dtype=torch.float64, requires_grad=False)

    if rank == 0:
        bar = tqdm(
            total=len(train_loader), 
            unit=' batch', 
            desc='train_shard', 
            smoothing=0.03,
            disable=not kwargs['progress'])

    heartbeat = time()

    for step, batch in enumerate(deviceloader(train_loader, rank)):
        model_inputs = batch['input_ids'], batch['token_type_ids'], batch['attention_mask']

        biemb = model(*model_inputs)
        closs = contrastive_loss(biemb[:,0], biemb[:,1], config['temp'])
        if config['amp'] != AMP.DISABLE:
            loss = scaler.scale(closs)
        else:
            loss = closs

        shard_loss.add_(loss.detach())
        bar.update(n=1)

        if rank == 0 and time() - heartbeat > 60:
            logger.info(f"heartbeat: training: epoch: {training_progress.epoch} shard: {training_progress.shard} step: {step}/{len(train_loader)}")
            heartbeat = time()

        # only sync before optimizer step
        if (step+1) % kwargs['grad_accum']!= 0:
            logger.debug('unsynced loss.backward')
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()

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
                optim.step()

            scheduler.step()
            training_progress.optimizer_step += 1
            if rank == 0:
                writer.add_scalar("learning_rate/train", scheduler.get_last_lr()[0], training_progress.optimizer_step)
                writer.flush()

            optim.zero_grad(set_to_none=True)

    dist.all_reduce(shard_loss)
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        avg_loss = shard_loss.cpu().numpy()/step
        writer.add_scalar("avg_loss/train", avg_loss, training_progress.optimizer_step)
        writer.flush()


def train(config, preempt_callback=None, **kwargs):
    #check_config(config)
    rank = dist.get_rank()
    if rank == 0:
        path = os.path.join(config['logdir'], config['name'], 'config.yml')
        save_config(config.to_dict(), path)
        writer = SummaryWriter(log_dir=os.path.join(config['logdir'], config['name']), flush_secs=30)

    logger = kwargs['logger']
    logger.info('started train function')

    if kwargs['debug'] == True:
        limit = 10*config['grad_accum']*config['batch_size']
    else:
        limit = None

    train_set = preprocess_split(data.DataSplit.TRAIN, config, limit_samples=limit, **kwargs)
    valid_set = preprocess_split(data.DataSplit.VALIDATION, config, limit_samples=limit, **kwargs)

    # both sets arent shuffled, shuffle train set every epoch manually
    train_loader = get_loader(train_set, config, **kwargs)
    valid_loader = get_loader(valid_set, config, **kwargs)

    logger.info('building model')
    if config['num_gpus'] > 0:
        m = BiEmbeddingModelCuda(config['model_name'], config['embedding_size'], config['amp'], hidden_dropout_prob=config['dropout']).to(rank)
    else:
        m = BiEmbeddingModel(config['model_name'], config['embedding_size'], config['amp'], hidden_dropout_prob=config['dropout']).to(rank)

    logger.info('warming up cuda benchmark')
    for step, batch in zip(range(12), deviceloader(train_loader, rank)):
        model_inputs = batch['input_ids'], batch['token_type_ids'], batch['attention_mask']
        m(*model_inputs)
        torch.cuda.synchronize()
        dist.barrier()

    if config['cuda_graphs'] == True:
        logger.info('cuda_graphs is True: building the cuda graph')
        torch.cuda.synchronize()
        dist.barrier()
        m.make_graphed(model_inputs)
        torch.cuda.synchronize()
        dist.barrier()

    logger.info('building DDP model')
    model = DDP(m, device_ids=[rank], find_unused_parameters=True)
    model.embedding_size = m.embedding_size

    if config['finetune_mode'] == FinetuneMode.ALL:
        params = model.parameters()
    elif config['finetune_mode'] == FinetuneMode.POOLING:
        params = model.pooling.parameters()

    optim = _optimizer_mapping[config['optimizer'].value](params, lr=config['learning_rate'])

    num_warmup_steps = ceil(config['warmup_batches'] / config['grad_accum'])
    num_training_steps_per_epoch = ceil(len(train_set) / (config['batch_size'] * config['grad_accum']))
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

    if kwargs['debug'] == True:
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

    logger.info(f'shard_size: {len(train_set)/num_shards} samples')
    logger.info(f'validation set size: {len(valid_set)} samples')

    rank = dist.get_rank()

    logger.info(f'starting training')
    while(training_progress.epoch < num_epochs):
        logger.info(f'starting epoch: {training_progress.epoch} of {num_epochs}')
        train_set.shuffle()

        while(training_progress.shard < num_shards):
            logger.info(f'training shard: {training_progress.shard}')

            shard = train_set.shard(num_shards, training_progress.shard, contiguous=True)
            train_loader = get_loader(shard, config)

            train_shard(
                state_tracker,
                train_loader,
                config,
                writer,
                grad_accum=grad_accum,
                **kwargs
                )


            logger.info(f'validating shard {training_progress.shard}')

            score = validate(state_tracker['model'], 
                    valid_loader, 
                    config, 
                    writer, 
                    state_tracker['training_progress'],
                    **kwargs)

            training_progress.shard_done()

            torch.cuda.synchronize()
            dist.barrier()
            if rank == 0:
                state_tracker.save()

            if 'preempt_callback' in kwargs:
                kwargs['preempt_callback'](state_tracker, score, config, **kwargs)

        training_progress.epoch_done()

    if 'hparam_callback' in kwargs and rank == 0:
        kwargs['hparam_callback'](writer, score)

    return score


def training_worker(rank, config, progress, debug):
    world_size = config['num_gpus']

    #os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    dist.init_process_group('nccl', rank=rank, world_size=world_size, timeout=TIMEOUT)
    logger = make_logger(config, debug=debug, rank=rank)
    torch.cuda.set_device(rank)
    train(config, progress=progress, debug=debug, logger=logger)

    torch.cuda.synchronize()
    dist.barrier(group=dist.group.WORLD)
    dist.destroy_process_group()


def clean_init():
    torch.cuda.empty_cache()
    np.random.seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.use_deterministic_algorithms(True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual training script.')
    parser.add_argument('config_file', type=argparse.FileType('r'))
    # these values override the config values if specified
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    # 
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--progress', action='store_true', default=False)
    args = parser.parse_args()

    config = TrainingConfig()

    with args.config_file as f:
        config_candidate = yaml.safe_load(f)

    timestamp = str(ceil(time()))
    if args.name is not None:
        if args.debug == True:
            prefix = 'debug'
        else: 
            prefix = 'run'
            
        config_candidate['name'] = f'{prefix}_{args.name}_t0_utc{timestamp}'

    if args.batch_size is not None:
        config_candidate['batch_size'] = args.batch_size

    for k,v in config_candidate.items():
        # parse through the settings parsing function
        config[k] = config.settings[k].parse(v)

    config.check()
    config.final()

    logdir = os.path.join(config['logdir'], config['name'])
    checkdir = logdir #os.path.join(config['checkpoint_dir'], config['name'])
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(checkdir, exist_ok=True)

    clean_init()
    torch.backends.cudnn.benchmark = True

    mp.set_start_method('spawn')
    mp.spawn(training_worker, 
            args=(config, args.progress, args.debug),
            nprocs = config['num_gpus'],
            join=True)

