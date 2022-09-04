from secora.train import *

def parse_args(argv):
    parser = argparse.ArgumentParser(description='manual training script.')
    parser.add_argument('config_file', type=argparse.FileType('r'))
    # these values override the config values if specified
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--max_checkpoints', type=int, default=None) 
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--progress', action='store_true', default=False)
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    main(sys.argv)
