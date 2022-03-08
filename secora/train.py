import os
import time
from time import time
from enum import Enum, auto

from math import ceil
import argparse
import datetime
import re
from contextlib import contextmanager

from abc import ABC

from pdb import set_trace as bp

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

from secora.models import *
import secora.models as models

from secora.data import *
from secora import data
from secora.config import *
from secora.infer import build_embedding_space, k_nearest_neighbors, validate
from secora.losses import contrastive_loss, mrr
import secora.losses as losses #.LOSS_TEMPERATURE
from secora.tracking import *

import datasets
from SM3 import SM3

from grad_cache.functional import cached, cat_input_tensor

DEVICE_BATCHSIZE = 6

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
        pass

    def run(self):
        for t in self.tasks:
            t.run()



def get_heartbeat():
    heartbeat = time()
    rank = dist.get_rank()

    @contextmanager
    def heartbeat():
        yield

    return heartbeat

def train_step(model_inputs, cache, shard_loss, state_tracker, config, writer, **kwargs):
    rank = dist.get_rank()

    scaler = state_tracker['scaler']
    optim = state_tracker['optimizer']
    scheduler = state_tracker['scheduler']
    model = state_tracker['model']
    model.train()

    training_progress = state_tracker['training_progress']
    step = training_progress.batch
    
    cache_accum = config['batch_size'] // DEVICE_BATCHSIZE

    grad_cache = True
    if grad_cache:
        grad_accum=1

    if not grad_cache:
        emb1 = model(*model_inputs)
        emb2 = model(*model_inputs)
        closs = contrastive_loss(emb1, emb2, temp=config['temp'])
        if config['amp'] == AMP.DISABLE:
            loss = closs
        else:
            loss = scaler.scale(closs)

        shard_loss.add_(loss.detach())
    else:
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

        if rank == 0:
            writer.add_scalar("learning_rate/train", scheduler.get_last_lr()[0], training_progress.optimizer_step)
            writer.flush()

        optim.zero_grad(set_to_none=True)


class GCache:
    def __init__(self, temp):
        self.temp = temp

        self.cache_1: list = []
        self.cache_2: list = []
        self.closures_1: list = []
        self.closures_2: list = []

    def reset(self):
        self.cache_1 = []
        self.cache_2 = []
        self.closures_1 = []
        self.closures_2 = []

    @cached
    def call_model(self, model, model_inputs):
        return model(*model_inputs)

    @cat_input_tensor
    def loss_fn(self, x, y):
        return contrastive_loss(x, y, temp=self.temp)

class ProgressDisplay:
    ''' logs the livelyness and progress of a training task '''

    def __init__(self, training_progress, len_, **kwargs):
        self.rank = dist.get_rank()
        self.heartbeat = time()
        self.training_progress = training_progress

        show_bar = kwargs.get('progress', False) and self.rank == 0
        print('--------')
        print(len_)
        print('--------')

        self.bar = tqdm(
            total=len_,
            unit=' batch', 
            desc='train_shard', 
            smoothing=0.03,
            disable=not show_bar)


    def update(self):
        if self.rank == 0 and time() - self.heartbeat > 10:
            self.bar.clear()
            e = self.training_progress.epoch
            s = self.training_progress.shard
            #b = training_progress.shard
            batch = self.training_progress.batch
            logger = logging.getLogger('secora')
            logger.info(f"heartbeat: training: epoch: {e} shard: {s} batch: {batch}")
            self.heartbeat = time()

        self.bar.update(n=1)

    def reset(self):
        self.bar.reset()


def train_shard(
        state_tracker,
        train_loader,
        config,
        writer=None,
        grad_accum=1,
        **kwargs
        ):

    rank = dist.get_rank()
    logger = logging.getLogger('secora')
    training_progress = state_tracker['training_progress']

    cache = GCache(config['temp'])

    shard_loss = torch.tensor(
            [0.], 
            device=rank, 
            dtype=torch.float64, 
            requires_grad=False)

    for batch in deviceloader(train_loader, rank):
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


def train(config, preempt_callback=None, **kwargs):
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
    else:
        t_limit = 200000
        v_limit = 50000

    train_set = preprocess_split(data.DataSplit.TRAIN, config, limit_samples=t_limit, **kwargs)
    valid_set = preprocess_split(data.DataSplit.VALIDATION, config, limit_samples=v_limit, **kwargs)

    logger.info('building model')
    if config['num_gpus'] > 0:
        m = EmbeddingModelCuda(config['model_name'], config['embedding_size'], config['amp'], hidden_dropout_prob=config['dropout']).to(rank)
    else:
        m = EmbeddingModel(config['model_name'], config['embedding_size'], config['amp'], hidden_dropout_prob=config['dropout']).to(rank)

    logger.info('warming up cuda benchmark')
    train_loader = get_loader(train_set, config['batch_size'], workers=0, dist=dist.is_initialized(), **kwargs)
    for step, batch in zip(range(12), deviceloader(train_loader, rank)):
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
    display = ProgressDisplay(training_progress, num_steps_per_shard, **kwargs)

    state_tracker = StateTracker(
            config['name'],
            config['logdir'],
            config['max_checkpoints'],
            model=model,
            optimizer=optim,
            scheduler=scheduler,
            scaler=scaler,
            training_progress=training_progress)

    if kwargs.get('debug',False) == True:
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
            train_loader = get_loader(shard, config['batch_size'])

            train_shard(
                state_tracker,
                train_loader,
                config,
                writer,
                grad_accum=grad_accum,
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
    make_logger(config, debug=debug, rank=rank)
    logger = logging.getLogger('secora')

    torch.cuda.set_device(rank)
    train(config, progress=progress, debug=debug, logger=logger)

    torch.cuda.synchronize()
    dist.barrier(group=dist.group.WORLD)
    dist.destroy_process_group()


def clean_init(seed):
    torch.cuda.empty_cache()
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual training script.')
    parser.add_argument('config_file', type=argparse.FileType('r'))
    # these values override the config values if specified
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--max_checkpoints', type=int, default=None) 
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

    if args.max_checkpoints is not None:
        config_candidate['max_checkpoints'] = args.max_checkpoints

    config.parse_from_dict(config_candidate)
    config.check()
    config.final()

    logdir = os.path.join(config['logdir'], config['name'])
    checkdir = logdir #os.path.join(config['checkpoint_dir'], config['name'])
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(checkdir, exist_ok=True)

    clean_init(seed=config['seed'])
    torch.backends.cudnn.benchmark = True

    datasets.set_progress_bar_enabled(args.progress)

    mp.set_start_method('spawn')
    mp.spawn(training_worker, 
            args=(config, args.progress, args.debug),
            nprocs = config['num_gpus'],
            join=True)
