import pytest
import tempfile
import yaml

from tests.test_train import example_config, MockLogger

from secora.train import *
from secora.train_cli import parse_args

def load_main_state(config, **kwargs):
    # unimportant value
    trainset_len = 1000

    # initialize state
    m = build_model(config, **kwargs)
    #model = distribute_model(m, **kwargs)
    model = m
    optim = build_optimizer(config, model, **kwargs)
    scheduler = build_scheduler(optim, config, trainset_len, **kwargs)
    scaler = GradScaler()
    training_progress = TrainingProgress()

    state_tracker = StateTracker(
            config['name'],
            config['logdir'],
            config['max_checkpoints'],
            MockLogger(),
            model=model,
            optimizer=optim,
            scheduler=scheduler,
            scaler=scaler,
            training_progress=training_progress)

    return state_tracker


def run_training(argv, *args, **kwargs):
    cli_args = parse_args(argv)
    timestamp = '1'
    config = build_config(config_id=timestamp, args=cli_args)
    rng_init(seed=config['seed'], deterministic=True)

    mp.spawn(training_worker, 
            args=(config, cli_args.progress, cli_args.debug, args, kwargs),
            nprocs = config['num_gpus'],
            join=True)
    state = load_main_state(config, rank=0, logger=MockLogger())

    return state


# just try to run the mainloop on a single gpu
@pytest.mark.slow
@pytest.mark.cuda
def test_train_main_determinism():
    with tempfile.NamedTemporaryFile('w') as tmpconf:
        with tempfile.TemporaryDirectory() as tmpdirname:
            example_config_yaml = yaml.safe_load(example_config)
            example_config_yaml['logdir'] = tmpdirname
            example_config_yaml['checkpoint_dir'] = tmpdirname
            yaml.safe_dump(example_config_yaml, tmpconf)

            testargs = ['/root/secora/secora/train.py', tmpconf.name, '--debug']
            state1 = run_training(testargs, logger=MockLogger())

    # start a exactly similarly configured run
    # but in a clean logdir
    with tempfile.NamedTemporaryFile('w') as tmpconf:
        with tempfile.TemporaryDirectory() as tmpdirname:
            example_config_yaml = yaml.safe_load(example_config)
            example_config_yaml['logdir'] = tmpdirname
            example_config_yaml['checkpoint_dir'] = tmpdirname
            yaml.safe_dump(example_config_yaml, tmpconf)

            testargs = ['/root/secora/secora/train.py', tmpconf.name, '--debug']
            state2 = run_training(testargs, logger=MockLogger())


    s1 = state1['model'].state_dict() 
    s2 = state2['model'].state_dict()

    torch.use_deterministic_algorithms(mode=True)
    assert all([torch.all(torch.eq(x,y)) for x,y in zip(s1.values(), s2.values())])
