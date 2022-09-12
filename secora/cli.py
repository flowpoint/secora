import sys
import argparse

from secora.train import train_start, train_resume
from secora.hyperparam_search import hyperparam_search


def main(argv):
    secora_parser = argparse.ArgumentParser(description='secora training library')
    secora_subparsers = secora_parser.add_subparsers(help='hyperparam_search or train or evaluate or infer', required=True)


    hyperparam_search_parser = secora_subparsers.add_parser('hyperparam_search', help='hyperparam_search')
    hyperparam_search_parser.add_argument('config_file', type=argparse.FileType('r'))
    hyperparam_search_parser.add_argument('--db', type=str, default=None)
    hyperparam_search_parser.add_argument('--search_name', type=str, default=None)
    hyperparam_search_parser.add_argument('--trials', type=int, default=10)
    hyperparam_search_parser.add_argument('--max_hparam_shards', type=int, default=32)
    hyperparam_search_parser.add_argument('--debug', action='store_true', default=False)
    hyperparam_search_parser.add_argument('--device', type=int, default=0)
    hyperparam_search_parser.add_argument('--seed', type=int, default=22)
    hyperparam_search_parser.set_defaults(func=hyperparam_search)


    train_parser = secora_subparsers.add_parser('train', help='train')
    train_subparsers = train_parser.add_subparsers(help='start or resume training', required=True)


    start_parser = train_subparsers.add_parser('start', help='start')
    start_parser.add_argument('config_file', type=argparse.FileType('r'))
    # these values override the config values if specified
    start_parser.add_argument('--batch_size', type=int, default=None)
    start_parser.add_argument('--max_checkpoints', type=int, default=None) 
    start_parser.add_argument('--debug', action='store_true', default=False)
    start_parser.add_argument('--progress', action='store_true', default=False)
    start_parser.add_argument('--deterministic', action='store_true', default=False)

    # set the training main function to call
    start_parser.set_defaults(func=train_start)


    resume_parser = train_subparsers.add_parser('resume', help='resume')
    resume_parser.add_argument('training_run_id', type=str, default=None)
    resume_parser.add_argument('--debug', action='store_true', default=False)
    resume_parser.add_argument('--storage_path', type=str, default=None)

    resume_parser.set_defaults(func=train_resume)


    cli_args = secora_parser.parse_args(argv[1:])
    cli_args.func(cli_args)


if __name__ == "__main__":
    main(sys.argv)