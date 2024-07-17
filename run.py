import sys
import numpy as np
import random
from pdb import set_trace
from dataset import get_dataset
import torch
import argparse

print('PYTORCH VERSION\n', torch.__version__)
print('PYTHON VERSION\n', sys.version)
print('CUDA VERSION\n', torch.version.cuda)


def main(args):
    raw_tr, raw_te = get_dataset(args.dataset)
    breakpoint()
    pass


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
arg = parser.add_argument_group()
arg.add_argument('--seed', type=int, default=123) 
arg.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'FashionMNIST', 'SVHN', 'CIFAR10'])
arg.add_argument('--dt_dataset', type=str, default='CIFAR10')
arg.add_argument('--dt_labels', type=list, default=[5, 6, 7, 8, 9])

if __name__ == '__main__':
    args = parser.parse_args()
    fix_seed(args.seed)
    main(args)