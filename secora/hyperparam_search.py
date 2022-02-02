import optuna
from optuna.trial import TrialState

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
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

def callback(epoch, shard, score, config, **kwargs):
    trial = kwargs['trial']
    trial.report(score, epoch*config['shards'] + shard)

    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()


def hyperopt_worker(rank, config, progress, debug, master_port):
    n_trials = 3

    world_size = config['num_gpus']
    host_name = config['hostname']

    os.environ['MASTER_ADDR'] = host_name
    os.environ['MASTER_PORT'] = master_port

    dist.init_process_group('nccl', rank=rank, world_size=world_size, timeout=datetime.timedelta(65))
    logger = make_logger(config, debug=debug, rank=rank)
    torch.cuda.set_device(rank)

    def objective(default_config, progress, debug, logger, rank, single_trial):
        dist.barrier()
        trial = optuna.integration.TorchDistributedTrial(single_trial, rank)

        #overwrite config values with search hyperparameters
        t = trial.number
        config = deepcopy(default_config)
        config['name'] = config['name'] + f"_{t}"
        config['lr'] = trial.suggest_float('lr', 1e-6, 1e-4, log=True)

        logdir = os.path.join(config['logdir'], config['name'])
        checkdir = os.path.join(config['checkpoint_dir'], config['name'])
        os.makedirs(logdir, exist_ok=True)
        os.makedirs(checkdir, exist_ok=True)

        return train(config, preempt_callback=callback, trial=trial, progress=progress, debug=debug, logger=logger)

    def obj_(single_trial): return objective(config, progress, debug, logger, rank, single_trial) 

    study = None

    if rank == 0:
        logger.info('creating study')
        study = optuna.create_study(direction='maximize')
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
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--progress', action='store_true', default=False)
    parser.add_argument('--port', type=int, default=random.randint(10000, 15000))

    args = parser.parse_args()
    master_port = str(args.port)

    config = load_config(args.config_path)
    config = overwrite_config(args, config)

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

