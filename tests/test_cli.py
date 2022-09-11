import pytest

import os
import tempfile
import yaml

import torch

from tests.test_train import example_config, MockLogger

from secora.train import build_model, build_optimizer, build_scheduler, build_config, GradScaler, TrainingProgress, StateTracker
from secora.cli import main

def load_main_state(config, **kwargs):
    # unimportant value
    trainset_len = 1000

    # initialize state
    kwargs['logger'] = MockLogger()
    m = build_model(config, **kwargs)
    #model = distribute_model(m, **kwargs)
    model = m
    optim = build_optimizer(config, model, **kwargs)
    scheduler = build_scheduler(optim, config, trainset_len, **kwargs)
    scaler = GradScaler()
    training_progress = TrainingProgress()

    state_tracker = StateTracker(
            config['training_run_id'],
            config['logdir'],
            config['max_checkpoints'],
            MockLogger(),
            model=model,
            optimizer=optim,
            scheduler=scheduler,
            scaler=scaler,
            training_progress=training_progress)

    return state_tracker


@pytest.mark.slow
@pytest.mark.cuda
def test_train_main_progress():
    # wether trainable params have actually changed
    with tempfile.NamedTemporaryFile('w') as tmpconf:
        with tempfile.TemporaryDirectory() as tmpdirname:
            example_config_yaml = yaml.safe_load(example_config)
            example_config_yaml['logdir'] = tmpdirname
            example_config_yaml['checkpoint_dir'] = tmpdirname
            yaml.safe_dump(example_config_yaml, tmpconf)

            testargs = ['/root/secora/secora/cli.py', 'train', 'start' ,tmpconf.name, '--debug', '--deterministic']
            state_tracker0  = load_main_state(build_config(example_config_yaml))
            main(testargs)
            state_tracker1  = load_main_state(build_config(example_config_yaml))

    s0 = state_tracker0['model'].state_dict() 
    s1 = state_tracker1['model'].state_dict() 

    for (x, xv), (y, yv) in zip(s0.items(), s1.items()):
        if xv.requires_grad == True:
            assert not torch.equal(xv, yv), f"model param werent updated during training: {x} {y} : {xv} {yv}"


@pytest.mark.slow
@pytest.mark.cuda
def test_train_main_determinism():
    with tempfile.NamedTemporaryFile('w') as tmpconf:
        with tempfile.TemporaryDirectory() as tmpdirname:
            example_config_yaml = yaml.safe_load(example_config)
            example_config_yaml['logdir'] = tmpdirname
            example_config_yaml['checkpoint_dir'] = tmpdirname
            yaml.safe_dump(example_config_yaml, tmpconf)

            testargs = ['/root/secora/secora/cli.py', 'train', 'start' ,tmpconf.name, '--debug', '--deterministic']
            main(testargs)
            state_tracker1  = load_main_state(build_config(example_config_yaml))

    # start a exactly similarly configured run
    # but in a clean logdir
    with tempfile.NamedTemporaryFile('w') as tmpconf:
        with tempfile.TemporaryDirectory() as tmpdirname:
            example_config_yaml = yaml.safe_load(example_config)
            example_config_yaml['logdir'] = tmpdirname
            example_config_yaml['checkpoint_dir'] = tmpdirname
            yaml.safe_dump(example_config_yaml, tmpconf)

            testargs = ['/root/secora/secora/cli.py', 'train', 'start', tmpconf.name, '--debug', '--deterministic']
            main(testargs)
            state_tracker2  = load_main_state(build_config(example_config_yaml))


    s1 = state_tracker1['model'].state_dict() 
    s2 = state_tracker2['model'].state_dict()

    for (x, xv), (y, yv) in zip(s1.items(), s2.items()):
        assert torch.equal(xv, yv), f"model param differ: {y} {z} : {yv} {zv}"

    
@pytest.mark.slow
@pytest.mark.cuda
def test_train_main_resume():
    with tempfile.NamedTemporaryFile('w') as tmpconf:
        with tempfile.TemporaryDirectory() as tmpdirname:
            example_config_yaml = yaml.safe_load(example_config)
            example_config_yaml['logdir'] = tmpdirname
            example_config_yaml['checkpoint_dir'] = tmpdirname
            example_config_yaml['epochs'] = 4
            yaml.safe_dump(example_config_yaml, tmpconf)

            testargs = ['/root/secora/secora/cli.py', 'train', 'start' ,tmpconf.name, '--debug', '--deterministic']
            main(testargs)
            state_tracker1  = load_main_state(build_config(example_config_yaml))

    # start a exactly similarly configured run
    # but in a clean logdir
    with tempfile.NamedTemporaryFile('w') as tmpconf:
        with tempfile.TemporaryDirectory() as tmpdirname:
            example_config_yaml = yaml.safe_load(example_config)
            example_config_yaml['logdir'] = tmpdirname
            example_config_yaml['checkpoint_dir'] = tmpdirname
            example_config_yaml['epochs'] = 2
            yaml.safe_dump(example_config_yaml, tmpconf)

            testargs = ['/root/secora/secora/cli.py', 'train', 'start' ,tmpconf.name, '--debug', '--deterministic']
            main(testargs)

            trainid = sorted(os.listdir(tmpdirname))[0]
            testargs = ['/root/secora/secora/cli.py', 'train', 'resume', trainid, '--debug', '--storage_path', tmpdirname]
            main(testargs)
            state_tracker2  = load_main_state(build_config(example_config_yaml))


    s1 = state_tracker1['model'].state_dict() 
    s2 = state_tracker2['model'].state_dict()

    for (x, xv), (y, yv), in zip(s1.items(), s2.items()):
        assert torch.equal(xv,yv), f"model param differ: {y} {z} : {yv} {zv}"
