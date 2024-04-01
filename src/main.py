import argparse

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import random
import torch

from engine import train
from data_setup import *
from modeling import *

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_name',
                        default='google-bert/bert-base-uncased',
                        type=str,
                        required=True)
    parser.add_argument('--data_dir',
                        default='../data/ViRes',
                        type=str,
                        required=True)
    parser.add_argument('output_dir',
                        default='../data/ViRes/output',
                        type=str,
                        required=True)
    
    # --- TRAINING ARGS ---
    parser.add_argument('--max_seq_length',
                        default=128,
                        type=int,
                        required=True)
    parser.add_argument('--train_batch_size',
                        default=32,
                        type=int)
    parser.add_argument('--test_batch_size',
                        default=32,
                        type=int)
    parser.add_argument('--lr',
                        default=2e-5,
                        type=float)
    parser.add_argument('--epochs',
                        default=5,
                        type=int)
    parser.add_argument('--warmup_proportion',
                        default=0.1,
                        type=float)
    parser.add_argument('--accumulate_gradients',
                        default=1,
                        type=int)
    parser.add_argument('--gradient_accumulation_steps',
                        default=1,
                        type=int)
    parser.add_argument('--seed',
                        default=42,
                        type=int)

    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

if __name__ == '__main__':
    main()