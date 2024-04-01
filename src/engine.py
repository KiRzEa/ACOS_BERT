from tqdm.auto import tqdm
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import PreTrainedModel, PreTrainedTokenizerm, get_scheduler, SchedulerType
from modeling import BertForTABSAJoint_CRF

from sklearn.metrics import *

def train_step(model: BertForTABSAJoint_CRF, 
               optimizer: torch.optim.Optimizer, 
               loss_fn: torch.nn.Module, 
               scheduler: SchedulerType,
               dataloader: DataLoader,
               device: torch.device,
               hparams: Dict):

    model.train()
    train_loss = 0.
    train_ner_loss = 0.

    progress_bar = tqdm(
        enumerate(dataloader),
        desc='Training',
        total=len(dataloader)
    )
    for step, (_, batch) in progress_bar:
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, ner_mask, acs_labels, ner_labels = batch
        loss, ner_loss, _, _ = model(input_ids=input_ids, 
                                     attention_mask=attention_mask, 
                                     ner_mask=ner_mask,
                                     acs_labels=acs_labels,
                                     ner_labels=ner_labels)
        if hparams['n_gpu'] > 1:
            loss = loss.mean()
            ner_loss = ner_loss.mean()
        if hparams['gradient_accumulation_steps'] > 1:
            loss = loss / hparams['gradient_accumulation_steps']
            ner_loss = ner_loss / hparams['gradient_accumulation_steps']
        loss.backward(retain_graph=True)
        ner_loss.backward()

        train_loss += loss
        train_ner_loss += ner_loss
        if (step + 1) % hparams['gradient_accumulation_steps'] == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        
        progress_bar.set_postfix(
            {
                'train_loss': train_loss.detach().cpu().numpy(),
                'train_ner_loss': train_ner_loss.detach().cpu().numpy()
            }
        )

    return train_loss, train_ner_loss


def test_step(model: BertForTABSAJoint_CRF,
              dataloader: DataLoader,
              device: torch.device,
              hparams: Dict):
    
    model.eval()

    test_loss = 0.
    test_ner_loss = 0.
    test_acc = 0.

    progress_batch = tqdm(
        dataloader,
        desc='Evaluating',
        total=len(dataloader)
    )

    with torch.inference_mode():
        for (example_id, batch) in progress_batch:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, ner_mask, acs_labels, ner_labels = batch
            loss, ner_loss, logits, ner_predict = model(input_ids=input_ids,
                                                        attention_mask=attention_mask,
                                                        ner_mask=ner_mask,
                                                        acs_labels=acs_labels,
                                                        ner_labels=ner_labels)
            
            acs_predictions = torch.softmax(logits, dim=-1).argmax(dim=-1).detach().cpu().numpy()
            acs_labels.to('cpu').numpy()
            ner_labels.to('cpu').numpy()

def train(model: BertForTABSAJoint_CRF,
          optimizer: torch.optim,
          loss_fn: torch.nn.Module,
          scheduler: SchedulerType,
          train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          epochs: int,
          device: torch.device,
          hparams: Dict):
    
    model.to(device)
    raise NotImplementedError