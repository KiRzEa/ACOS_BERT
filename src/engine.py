from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import PreTrainedModel, PreTrainedTokenizerm, AdamW, get_scheduler

from sklearn.metrics import *

def train_step(model: PreTrainedModel, 
               optimizer: torch.optim, 
               loss_fn: torch.nn.Module, 
               train_dataloader: DataLoader):
    raise NotImplementedError

def test_step(model: PreTrainedModel,
              test_dataloader: DataLoader):
    raise NotImplementedError

def train(model: PreTrainedModel,
          optimizer: torch.optim,
          loss_fn: torch.nn.Module,
          train_dataloader: DataLoader,
          test_dataloader: DataLoader):
    raise NotImplementedError