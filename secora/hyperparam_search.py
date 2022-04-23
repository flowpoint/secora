import optuna
import random
import argparse
import os
import logging
from enum import Enum, auto

import numpy as np
import datetime

from copy import deepcopy
from functools import partial

from optuna.trial import TrialState
from optuna.samplers import RandomSampler

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from secora.train import *
from secora.training_tasks import *
from secora.tracking import make_logger
from secora.data import get_loader


def clean_init(seed):
    torch.cuda.empty_cache()
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def build_trial_config(default_config, single_trial):
    ''' we set the hyperparameters here instead of inside the training code,
    to decouple slighly from the optuna/hparam framework dependency.
    also we want trials to fail fast if the configuration is wrong, 
    because training big models is costly
    this also makes unittesting easier
    '''

    dist.barrier()
    rank = dist.get_rank()
    trial = optuna.integration.TorchDistributedTrial(single_trial, rank)

    # overwrite config values with search hyperparameters
    t = trial.number
    config = deepcopy(default_config)
    config['name'] = default_config['name'].replace('xxx', f"{t}")

    dist.barrier()

    if dist.get_rank() == 0:
        logdir = os.path.join(config['logdir'], config['name'])
        checkdir = os.path.join(config['checkpoint_dir'], config['name'])
        os.makedirs(logdir, exist_ok=True)
        os.makedirs(checkdir, exist_ok=True)
    dist.barrier()

    config['learning_rate'] = trial.suggest_uniform('learning_rate', 1e-6, 1e-4)
    config['dropout'] = trial.suggest_uniform('dropout', 0.05, 0.3)
    config['temp'] = trial.suggest_uniform('temp', 0.02, 0.1)
    accums = [x//config['batch_size'] for x in [32, 64, 128, 256, 512]]
    config['grad_accum'] = trial.suggest_categorical('grad_accum', accums)
    optimizers = ['adam', 'adamw', 'sm3', 'sgd']
    config['optimizer'] = trial.suggest_categorical('optimizer', optimizers)

    hparams = {x:config[x] for x in ['learning_rate', 'dropout', 'temp', 'grad_accum', 'optimizer']}

    training_config = TrainingConfig()
    
    training_config.parse_from_dict(config)
    training_config.check()
    training_config.final()

    return hparams, training_config


class Objective:
    def __init__(
            self,
            default_config, 
            progress, 
            debug, 
            study):

        self.progress = progress
        self.debug = debug
        self.study = study
        self.default_config = default_config

    def save_callback(self, checkpoint_id):
        logdir = os.path.join(self.default_config['logdir'], self.default_config['name'])
        study_path = os.path.join(logdir, f'study_{str(checkpoint_id)}.pickle')
        with open(study_path, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(study, f, pickle.HIGHEST_PROTOCOL)

    def hparam_callback(self, writer, score):
        if dist.get_rank() == 0:
            writer.add_hparams(
                self.hparams, 
                {'mrr/validation':score}, 
                hparam_domain_discrete={'optimizer':optimizers},
                run_name=config['name'])
            writer.flush()

    def training_plan(self, state_tracker, train_set, valid_set, config, writer, display, **kwargs):
        rank = dist.get_rank()
        logger = logging.getLogger('secora')
        logger.info(f'starting training')

        num_epochs = config['epochs']
        num_shards = config['shards']

        training_progress = state_tracker['training_progress']

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

                checkpoint_id = ceil(time.time())
                if rank == 0:
                    state_tracker.save(checkpoint_id)
                    self.preempt_callback(checkpoint_id, state_tracker, score, config, **kwargs)

            training_progress.epoch_done()

        if rank == 0:
            self.hparam_callback(writer, score)

    def train(self, config, **kwargs):
        t_args = training_setup(config, **kwargs)
        self.training_plan(*t_args, **kwargs)

    def preempt_callback(self, checkpoint_id, state_tracker, score, config, **kwargs):
        epoch = state_tracker['training_progress'].epoch
        shard = state_tracker['training_progress'].shard
        self.trial = kwargs['trial']
        self.trial.report(score, epoch*config['shards'] + shard)

        self.save_callback(checkpoint_id)

        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    def __call__(self, trial):
        self.hparams, self.config = build_trial_config(self.default_config, trial)
        score = self.train(
                self.config, 
                trial=trial,
                progress=self.progress, 
                debug=self.debug)
        return score


def _list_checkpoints(dir_):
    return list(filter(lambda x: x.startswith("study"), os.listdir(dir_)))


def load_latest(default_config):
    logdir = os.path.join(default_config['logdir'], default_config['name'])
    logger = logging.getLogger('secora')
    logger.info('trying to load latest study checkpoint')

    study_name = default_config['name']
    study = None

    # save study
    storage_name = None
    checkpoints = _list_checkpoints(logdir)

    if checkpoints != []:
        with open(sorted(checkpoints)[-1], 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            study = pickle.load(f)

    return study, storage_name


def hyperopt_worker(rank, default_config, progress, debug):
    n_trials = 8

    world_size = default_config['num_gpus']

    dist.init_process_group('nccl', rank=rank, world_size=world_size, timeout=datetime.timedelta(65))
    make_logger(default_config, debug=debug, rank=rank)

    logger = logging.getLogger('secora')
    torch.cuda.set_device(rank)

    study = None
    if rank == 0:
        study, storage_name = load_latest(default_config)

    if rank == 0 and study is None:
        study_name = default_config['name']

        study = optuna.create_study(direction='maximize',
                study_name=study_name,
                storage=storage_name,
                load_if_exists=True,
                sampler= RandomSampler(seed=default_config['seed']),
                # note, if warmup steps are more than steps in a shard, min_resource can mean that 
                # trials are incorrectly discarded, as they only warmed up and didn't train
                pruner=optuna.pruners.HyperbandPruner(
                    min_resource=1,
                    max_resource=default_config['epochs']*default_config['shards'], 
                    reduction_factor=3
                    ),
                )

    objectives = []

    if rank == 0:
        for t in range(n_trials):
            objective = Objective(
                    default_config,
                    progress,
                    debug,
                    study)
            objectives.append(objective)

        logger.info('optimizing study')
        study.optimize(
                objective, 
                gc_after_trial=True, 
                n_trials=n_trials, 
                callbacks=[])

    else:
        for t in range(n_trials):
            try:
                logger.debug(f'trying objective fun {t}')
                objectives[t](None)
            except optuna.TrialPruned:
                logger.debug(f'except optuna TrialPruned in {t}')

    if rank == 0:
        assert study is not None
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print(f"finished trials:{len(study.trials)}")
        print(f"finished trials:{len(pruned_trials)}")
        print(f"finished trials:{len(complete_trials)}")
        print(f"best_trial:{study.best_trial}")
        trial = study.best_trial
        print(f"trial_value:{trial.value}")

    torch.cuda.synchronize()
    dist.barrier(group=dist.group.WORLD)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='hyperparam search')
    parser.add_argument('config_file', type=argparse.FileType('r'))
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--max_checkpoints', type=int, default=None) 
    parser.add_argument('--progress', action='store_true', default=False)

    args = parser.parse_args()

    with args.config_file as f:
        config_candidate = yaml.safe_load(f)

    timestamp = str(ceil(time()))
    #if args.name is not None:

    if args.debug == True:
        prefix = 'debug'
    else: 
        prefix = 'run'


    if args.batch_size is not None:
        config_candidate['batch_size'] = args.batch_size

    if args.max_checkpoints is not None:
        config_candidate['max_checkpoints'] = args.max_checkpoints

    nam = 'hparasearch' if args.name is None else args.name
    config_candidate['name'] = f'{prefix}_{nam}_txxx_utc{timestamp}'

    clean_init(config_candidate['seed'])
    torch.backends.cudnn.benchmark = True

    mp.set_start_method('spawn')
    mp.spawn(hyperopt_worker, 
            args=(config_candidate, args.progress, args.debug),
            nprocs = config_candidate['num_gpus'],
            join=True)
