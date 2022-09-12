import optuna
from optuna.trial import TrialState
from optuna.samplers import RandomSampler
from optuna.integration.tensorboard import TensorBoardCallback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import random
import os

from secora.tracking import make_logger
import numpy as np
import datetime
from secora.train import *
from secora.metrics import MetricLogger

from copy import deepcopy

from functools import partial

import optuna


def should_preempt(trial, state_tracker, score, config, **kwargs):
    epoch = state_tracker['training_progress'].epoch
    shard = state_tracker['training_progress'].shard

    trial.report(score, epoch*config['shards'] + shard)

    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

def build_hparam_config(config_candidate, trial, *args, **kwargs):
    t = trial.number
    config_candidate = deepcopy(config_candidate)
    config_candidate['training_run_id'] = config_candidate['training_run_id'] + f"_{t}"

    config_candidate['learning_rate'] = trial.suggest_uniform('learning_rate', 1e-6, 1e-4)
    config_candidate['dropout'] = trial.suggest_uniform('dropout', 0.05, 0.3)
    config_candidate['temp'] = trial.suggest_uniform('temp', 0.02, 0.1)

    #accums = [1]#[x//config_candidate['batch_size'] for x in [32, 64, 128, 256, 512]]
    #config_candidate['grad_accum'] = trial.suggest_categorical('grad_accum', accums)
    config_candidate['grad_accum'] = trial.suggest_int('grad_accum', 1, 1)
    optimizers = ['adam', 'adamw', 'sm3', 'sgd']
    config_candidate['optimizer'] = trial.suggest_categorical('optimizer', optimizers)

    hparams = {x:config_candidate[x] for x in ['learning_rate', 'dropout', 'temp', 'grad_accum', 'optimizer']}

    config = TrainingConfig()
    config.parse_from_dict(config_candidate)
    config.check()
    config.final()
    return config, hparams


def hparam_callback(hparams, metric_logger, score, rank, config, *args, **kwargs):
    if rank == 0:
        metric_logger.add_hparams(
            hparams, 
            {'mrr/hparam_search':score}, 
            hparam_domain_discrete={'grad_accum':[config['grad_accum']]},
            run_name=config['training_run_id'])
        metric_logger.flush()


#, 'optimizer':[config['optimizer']]},

def objective(search_args, config_candidate_, trial, *args, **kwargs):
    config_candidate=deepcopy(config_candidate_)
    config_candidate['training_run_id'] = f"training_run_id-trial{trial.number}"

    kwargs['debug'] = search_args.debug

    hparam_config, hparams = build_hparam_config(config_candidate, trial)


    results = run_workers(
            hparam_config, 
            resume=False, 
            *args, 
            preempt_callback=partial(should_preempt, trial),
            hparam_callback=partial(hparam_callback, hparams),
            **kwargs)

    score = results[0]['result']

    return score


def hyperparam_search(search_args, *args, **kwargs):
    max_hparam_shards = search_args.max_hparam_shards

    with search_args.config_file as f:
        config_candidate = yaml.safe_load(f)

    torch.cuda.set_device(search_args.device)

    study = optuna.create_study(direction='maximize',
            study_name=search_args.search_name,
            storage=search_args.db,
            load_if_exists=True,
            sampler=RandomSampler(seed=search_args.seed),
            # note, if warmup steps are more than steps in a shard, min_resource can mean that 
            # trials are incorrectly discarded, as they only warmed up and didn't train
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=1,
                max_resource=max_hparam_shards,
                reduction_factor=3
                ),
            )

    study.optimize(partial(objective, search_args, config_candidate), n_trials=search_args.trials)


    assert study is not None
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print(f"finished trials:{len(study.trials)}")
    print(f"finished trials:{len(pruned_trials)}")
    print(f"finished trials:{len(complete_trials)}")
    print(f"best_trial:{study.best_trial}")
    trial = study.best_trial
    print(f"trial_value:{trial.value}")
