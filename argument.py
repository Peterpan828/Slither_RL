import argparse

def parser():
    parser = argparse.ArgumentParser(description='Some hyperparameters')
    

    parser.add_argument('--episode', type=int, default=100,
                        help='total episode')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
 
    args = parser.parse_args()
    return args