import os
import logging
import argparse

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import random
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from engine import train
from data_utils import *
from data_setup import *
from modeling import *

from transformers import AutoTokenizer, AutoModel, AutoConfig, AdamW, get_scheduler

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
					datefmt = '%m/%d/%Y %H:%M:%S',
					level = logging.INFO)

def main():
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_name',
                        default='google-bert/bert-base-uncased',
                        type=str,
                        required=True)
    parser.add_argument('--data_dir',
                        default='../data/ViRes',
                        type=str)
    parser.add_argument('--output_dir',
                        default='../data/ViRes/output',
                        type=str)
    parser.add_argument('--experiment_name',
                        default='ACOS_BERT_01',
                        type=str)
    parser.add_argument('--end_token',
                        default='[SEP]',
                        type=str)
    
    # --- TRAINING ARGS ---
    parser.add_argument('--max_seq_length',
                        default=128,
                        type=int)
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    processor = DataProcessor(args.data_dir)
    label_set = LabelSet(['Target', 'Opinion'])
    compose_set = get_acs(os.path.join(args.data_dir, 'aspect_category_set.txt'), os.path.join(args.data_dir, 'sentiment_set.txt'))
    

    os.makedirs(args.output_dir, exist_ok=True)

    train_examples = processor.train_examples
    dev_examples = processor.dev_examples
    test_examples = processor.test_examples

    # -------- Dataset --------
    train_dataset = SupervisedDataset(train_examples, label_set, compose_set, tokenizer, args.max_seq_length, end_token=args.end_token)
    dev_dataset = SupervisedDataset(dev_examples, label_set, compose_set, tokenizer, args.max_seq_length, end_token=args.end_token)
    test_dataset = SupervisedDataset(test_examples, label_set, compose_set, tokenizer, args.max_seq_length, end_token=args.end_token)

    num_train_steps = int(len(train_dataset) / args.train_batch_size * args.epochs)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    # -------- DataLoader --------
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=os.cpu_count())
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.test_batch_size, num_workers=os.cpu_count())
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=os.cpu_count())

    # -------- Setup Training --------

    model = BertForTABSAJoint_CRF(args.model_name, config, 2, len(label_set.labels_to_id))

    no_decay = ['bias', 'gamma', 'beta']
    optimizer_parameters = [
		 {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
		 {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
		 ]

    optimizer = AdamW(params=optimizer_parameters,
					  lr=args.lr)
    
    scheduler = get_scheduler(
		name='linear',
		optimizer=optimizer,
		num_warmup_steps=int(args.warmup_proportion * num_train_steps),
		num_training_steps=num_train_steps
							 )

    hparams = {
        'gradient_accumulation_steps': args.gradient_accumulation_steps
    }
    writer = SummaryWriter(os.path.join(args.output_dir, 'runs', args.experiment_name))

    # ------ Training ------
    results = train(model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    epochs=args.epochs,
                    device=device,
                    hparams=hparams,
                    writer=writer)
    
    pd.DataFrame(results).to_csv(os.path.join(args.output_dir, 'results.csv'), index=False)

    os.makedirs('saved_models', exist_ok=True)
    torch.save(model, 'saved_models/{args.experiment_name}.bin')
    
if __name__ == '__main__':
    main()