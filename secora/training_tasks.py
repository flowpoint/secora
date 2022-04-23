import os
from math import ceil
from time import time
from enum import Enum, auto
from abc import ABC
import re

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from SM3 import SM3

from secora.models import *
import secora.models as models

from secora import data
from secora.tracking import *
from secora.config import *
from secora.infer import build_embedding_space, k_nearest_neighbors, validate
from secora.train_utils import GCache, ProgressDisplay
from secora.losses import contrastive_loss

DEVICE_BATCHSIZE = 6

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


class TrainingPlan:
    def __init__(self):
        self.tasks = []

    def add_task(self, task):
        #self.tasks.append(task)
        pass

    def run(self):
        for t in self.tasks:
            t.run()


def train_step(model_inputs, cache, shard_loss, state_tracker, config, writer, **kwargs):
    rank = dist.get_rank()

    scaler = state_tracker['scaler']
    optim = state_tracker['optimizer']
    scheduler = state_tracker['scheduler']
    model = state_tracker['model']

    training_progress = state_tracker['training_progress']
    step = training_progress.batch
    
    cache_accum = config['batch_size'] // DEVICE_BATCHSIZE
    grad_accum = config['grad_accum']

    emb1, c1 = cache.call_model(model, model_inputs)
    emb2, c2 = cache.call_model(model, model_inputs)
    cache.cache_1.append(emb1)
    cache.cache_2.append(emb2)
    cache.closures_1.append(c1)
    cache.closures_2.append(c2)

    # only sync before optimizer step
    if (step+1) % cache_accum == 0:

        closs = cache.loss_fn(cache.cache_1, cache.cache_2)
        loss = scaler.scale(closs)
        loss.backward()
        shard_loss.add_(loss.detach())
        
        for c, e in zip(cache.closures_1, cache.cache_1):
            c(e)
        for c, e in zip(cache.closures_2, cache.cache_2):
            c(e)

        cache.reset()

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
        training_progress.batch += 1
        optim.zero_grad(set_to_none=True)

        if rank == 0:
            writer.add_scalar(
                    "learning_rate/train", 
                    scheduler.get_last_lr()[0], 
                    training_progress.optimizer_step)
            writer.flush()


def train_shard(
        state_tracker,
        train_loader,
        config,
        writer=None,
        **kwargs
        ):


    rank = dist.get_rank()
    logger = logging.getLogger('secora')
    training_progress = state_tracker['training_progress']
    state_tracker['model'].train()

    cache = GCache(config['temp'], contrastive_loss)

    shard_loss = torch.tensor(
            [0.],
            device=rank,
            dtype=torch.float64,
            requires_grad=False)

    for batch in data.deviceloader(train_loader, rank):
        model_inputs = batch['input_ids'], batch['attention_mask']
        train_step(
                model_inputs, 
                cache,
                shard_loss,
                state_tracker,
                config,
                writer,
                **kwargs
                )
        kwargs['display'].update()

    kwargs['display'].reset()

    dist.all_reduce(shard_loss)
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        avg_loss = shard_loss.cpu().numpy() / len(train_loader)
        writer.add_scalar("avg_loss/train", avg_loss, training_progress.optimizer_step)
        writer.flush()


def setup_model(config, rank, train_set, **kwargs):
    logger = logging.getLogger('secora')

    logger.info('building model')
    if config['num_gpus'] > 0:
        mclass = EmbeddingModelCuda
    else:
        mclass = EmbeddingModel

    m = mclass(
            config['model_name'], 
            config['embedding_size'], 
            config['amp'], 
            hidden_dropout_prob=config['dropout']).to(rank)

    logger.info('warming up cuda benchmark')
    train_loader = data.get_loader(train_set, DEVICE_BATCHSIZE, workers=0, dist=dist.is_initialized(), **kwargs)
    for step, batch in zip(range(12), data.deviceloader(train_loader, rank)):
        model_inputs = batch['input_ids'], batch['attention_mask']
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
    return model


def training_plan(state_tracker, train_set, valid_set, config, writer, display, **kwargs):
    rank = dist.get_rank()
    logger = logging.getLogger('secora')
    logger.info(f'starting training')

    training_progress = state_tracker['training_progress']

    num_epochs = config['epochs']
    num_shards = config['shards']
    grad_accum = config['grad_accum']

    # do one validation pass with the base model
    score = validate(state_tracker['model'], 
            valid_set, 
            config, 
            writer, 
            state_tracker['training_progress'],
            **kwargs)

    while(training_progress.epoch < num_epochs):
        logger.info(f'starting epoch: {training_progress.epoch} of {num_epochs}')
        train_set.shuffle()

        while(training_progress.shard < num_shards):
            logger.info(f'training shard: {training_progress.shard}')
            shard = train_set.shard(num_shards, training_progress.shard, contiguous=True)
            train_loader = data.get_loader(shard, config['batch_size'])

            train_shard(
                state_tracker,
                train_loader,
                config,
                writer,
                display=display,
                **kwargs
                )

            logger.info(f'validating shard {training_progress.shard}')

            score = validate(state_tracker['model'], 
                    valid_set, 
                    config, 
                    writer, 
                    state_tracker['training_progress'],
                    **kwargs)

            training_progress.shard_done()
            torch.cuda.synchronize()
            dist.barrier()

            checkpoint_id = ceil(time())
            if rank == 0:
                state_tracker.save(checkpoint_id)

        training_progress.epoch_done()


def training_setup(config, **kwargs):
    rank = dist.get_rank()
    if rank == 0:
        path = os.path.join(config['logdir'], config['name'], 'config.yml')
        with open(path, 'w') as f:
            f.write(yaml.dump(config.to_dict()))

        writer = SummaryWriter(log_dir=os.path.join(config['logdir'], config['name']), flush_secs=30)

    logger = logging.getLogger('secora')
    logger.info('started train function')

    if kwargs['debug'] == True:
        t_limit = 20*config['batch_size']
        v_limit = 20*3000#DEVICE_BATCHSIZE

    train_set = data.preprocess_split(data.DataSplit.TRAIN, config, limit_samples=t_limit, **kwargs)
    valid_set = data.preprocess_split(data.DataSplit.VALIDATION, config, limit_samples=v_limit, **kwargs)
    
    model = setup_model(config, rank, train_set, **kwargs)

    if config['finetune_mode'] == FinetuneMode.ALL:
        params = model.parameters()
    elif config['finetune_mode'] == FinetuneMode.POOLING:
        params = model.pooling.parameters()

    optim = _optimizer_mapping[config['optimizer'].value](params, lr=config['learning_rate'])

    num_warmup_steps = config['warmup_batches']
    num_training_steps_per_epoch = ceil(len(train_set) / config['batch_size'])
    num_steps_per_shard = num_training_steps_per_epoch // config['shards']
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
    display = ProgressDisplay(training_progress, num_steps_per_shard, enabled=(rank == 0), **kwargs)

    state_tracker = StateTracker(
            config['name'],
            config['logdir'],
            config['max_checkpoints'],
            model=model,
            optimizer=optim,
            scheduler=scheduler,
            scaler=scaler,
            training_progress=training_progress)

    # load latest checkpoint 
    torch.cuda.synchronize()
    dist.barrier()
    state_tracker.load_latest()

    num_epochs = config['epochs']
    num_shards = config['shards']

    logger.info(f'shard_size: {len(train_set)/num_shards} samples')
    logger.info(f'validation set size: {len(valid_set)} samples')

    return state_tracker, train_set, valid_set, config, writer, display


def train(config, **kwargs):
    t_args = training_setup(config, **kwargs)
    training_plan(*t_args, **kwargs)

