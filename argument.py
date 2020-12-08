import argparse


def parser():
    parser = argparse.ArgumentParser(description='Some hyperparameters')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--test', type=int, default=0,
                        help='test or train')
    parser.add_argument('--workers', type=int, default=12,
                        help='Total Processes')
    parser.add_argument('--step_episode', type=int, default=20,
                        help='num of steps in a episode')
    parser.add_argument('--gpu_ids', type=int, default=-1, nargs='+',
                        help='GPUs to use [-1 CPU only] (default: -1)')
    parser.add_argument('--max_episode_length', type=int, default=10000,
                        help='max length of a episode')
    parser.add_argument('--time', type=float, default=0.5,
                        help='training time')
    parser.add_argument('--max_branch', type=int, default=10,
                        help='max # of branch')
    parser.add_argument('--random', type=int, default=0,
                        help='Random Policy')

    args = parser.parse_args()
    return args
