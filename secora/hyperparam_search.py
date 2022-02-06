import optuna
from optuna.trial import TrialState
from optuna.samplers import RandomSampler
from optuna.integration.tensorboard import TensorBoardCallback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import random
import argparse
import os

from tracking import make_logger
from config import load_config, overwrite_config
import numpy as np
import datetime
from train import train

from copy import deepcopy

from functools import partial


def callback(state_tracker, score, config, **kwargs):
    epoch = state_tracker['training_progress'].epoch
    shard = state_tracker['training_progress'].shard
    trial = kwargs['trial']
    trial.report(score, epoch*config['shards'] + shard)

    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()


def hyperopt_worker(rank, default_config, progress, debug, master_port):
    n_trials = 8

    world_size = default_config['num_gpus']
    host_name = default_config['hostname']

    os.environ['MASTER_ADDR'] = host_name
    os.environ['MASTER_PORT'] = master_port

    dist.init_process_group('nccl', rank=rank, world_size=world_size, timeout=datetime.timedelta(65))
    logger = make_logger(default_config, debug=debug, rank=rank)
    torch.cuda.set_device(rank)

    def objective(default_config, progress, debug, logger, rank, single_trial):
        dist.barrier()
        trial = optuna.integration.TorchDistributedTrial(single_trial, rank)

        #overwrite config values with search hyperparameters
        t = trial.number
        config = deepcopy(default_config)
        config['name'] = default_config['name'] + f"_{t}"

        config['learning_rate'] = trial.suggest_uniform('learning_rate', 1e-6, 1e-4)
        config['dropout'] = trial.suggest_uniform('dropout', 0.05, 0.3)
        config['temp'] = trial.suggest_uniform('temp', 0.02, 0.1)
        accums = [x//config['batch_size'] for x in [32, 64, 128, 256, 512]]
        config['grad_accum'] = trial.suggest_categorical('grad_accum', accums)
        optimizers = ['adam', 'adamw', 'sm3', 'sgd']
        config['optimizer'] = trial.suggest_categorical('optimizer', optimizers)

        if rank == 0:
            logdir = os.path.join(config['logdir'], config['name'])
            checkdir = os.path.join(config['checkpoint_dir'], config['name'])
            os.makedirs(logdir, exist_ok=True)
            os.makedirs(checkdir, exist_ok=True)
        dist.barrier()

        hparams = {x:config[x] for x in ['learning_rate', 'dropout', 'temp', 'grad_accum', 'optimizer']}

        def hparam_callback(writer, score):
            if rank == 0:
                writer.add_hparams(
                    hparams, 
                    {'mrr/validation':score}, 
                    hparam_domain_discrete={'grad_accum':accums, 'optimizer':optimizers},
                    run_name=config['name'])
                writer.flush()

        score = train(config, preempt_callback=callback, hparam_callback=hparam_callback, trial=trial, progress=progress, debug=debug, logger=logger)
        return score


    def obj_(single_trial): return objective(default_config, progress, debug, logger, rank, single_trial) 

    study = None

    if rank == 0:
        logdir = os.path.join(default_config['logdir'], default_config['name'])
        logger.info('creating study')

        sql_url = default_config['sql_url']

        study_name = default_config['name']
        storage_name = f"{sql_url}/{study_name}"

        study = optuna.create_study(direction='maximize',
                study_name=study_name,
                storage=storage_name,
                load_if_exists=True,
                sampler= RandomSampler(seed=default_config['seed']),
                # note, if warmup steps are more than steps in a shard, min_resource can mean that 
                # trials are incorrectly discarded, as they only warmed up and didn't train
                pruner=optuna.pruners.HyperbandPruner(
                    min_resource=1, max_resource=default_config['epochs']*default_config['shards'], reduction_factor=3
                    ),
                )
        logger.info('optimizing study')
        study.optimize(obj_, gc_after_trial=True, n_trials=n_trials)
    else:
        for t in range(n_trials):
            try:
                #torch.cuda.empty_cache()
                logger.debug(f'trying objective fun {t}')
                obj_(None)
            except optuna.TrialPruned:
                logger.debug(f'except optuna TrialPruned in {t}')
                pass

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
    parser.add_argument('config_path', type=str)
    parser.add_argument('sql_url', type=str)
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--progress', action='store_true', default=False)
    parser.add_argument('--port', type=int, default=random.randint(10000, 15000))

    args = parser.parse_args()
    master_port = str(args.port)

    config = load_config(args.config_path)
    config = overwrite_config(args, config)

    config['sql_url'] = args.sql_url

    np.random.seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.cuda.empty_cache()

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = True

    mp.set_start_method('spawn')
    mp.spawn(hyperopt_worker, 
            args=(config, args.progress, args.debug, master_port),
            nprocs = config['num_gpus'],
            join=True)

