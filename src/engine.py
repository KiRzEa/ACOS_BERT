from tqdm.auto import tqdm
from typing import Dict, List, Optional
from pathlib import Path

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from transformers import PreTrainedModel, PreTrainedTokenizer, get_scheduler, SchedulerType
from modeling import BertForTABSAJoint_CRF

from sklearn.metrics import *

def train_step(model: BertForTABSAJoint_CRF, 
               optimizer: torch.optim.Optimizer, 
               scheduler: SchedulerType,
               dataloader: DataLoader,
               device: torch.device,
               hparams: Dict):

    model.train()
    train_loss = 0.
    train_ner_loss = 0.
    train_acc = 0.

    progress_bar = tqdm(
        enumerate(dataloader),
        desc='Training',
        total=len(dataloader)
    )
    for step, (_, _, _, batch) in progress_bar:
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, ner_mask, acs_labels, ner_labels = batch
        loss, ner_loss, _, _ = model(input_ids=input_ids, 
                                     attention_mask=attention_mask, 
                                     ner_mask=ner_mask,
                                     acs_labels=acs_labels,
                                     ner_labels=ner_labels)

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
                'train_loss': train_loss.detach().cpu().numpy() / (step + 1),
                'train_ner_loss': train_ner_loss.detach().cpu().numpy() / (step + 1),
            }
        )
    train_loss /= len(dataloader)
    train_ner_loss /= len(dataloader)

    return train_loss, train_ner_loss


def test_step(model: BertForTABSAJoint_CRF,
              dataloader: DataLoader,
              device: torch.device,
              hparams: Dict,
              output_dir: str,
              epoch: int):
    
    model.eval()

    test_loss = 0.
    test_ner_loss = 0.
    test_acc = 0.

    progress_bar = tqdm(
        enumerate(dataloader),
        desc='Evaluating',
        total=len(dataloader)
    )
    with open(output_dir, f'test_pre_epoch_{epoch+1}.txt', 'w') as f:
        f.write('example_id\ttext\tacs\tacs_label\tacs_predict\tner_labels\tner_predictions')
    with torch.inference_mode():
        for step, (example_ids, texts, acs, batch) in progress_bar:
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

            for idx in range(len(example_ids)):
                f.write(f'{example_ids[idx]}\t \
                        {texts[idx]}\t \
                        {acs[idx]}\t \
                        {acs_labels[idx]}\t \
                        {acs_predictions[idx]}\t \
                        {ner_labels[idx]}\t \
                        {ner_predict[idx]}\n')
            test_acc += (acs_predictions == acs_labels).sum() / len(acs_predictions)
            test_loss += loss
            test_ner_loss += ner_loss

            progress_bar.set_postfix(
                {
                    'test_loss': test_loss.detach().cpu().numpy() / (step + 1),
                    'test_ner_loss': test_ner_loss.detach().cpu().numpy() / (step + 1),
                    'test_acc': test_acc / (step + 1)
                }
            )
        
        test_loss /= len(dataloader)
        test_ner_loss /= len(dataloader)
        test_acc /= len(dataloader)

def train(model: BertForTABSAJoint_CRF,
          optimizer: torch.optim,
          scheduler: SchedulerType,
          train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          epochs: int,
          device: torch.device,
          hparams: Dict,
          writer: SummaryWriter):
    
    results = {
        'train_loss': [],
        'train_ner_loss': [],
        'test_loss': [],
        'test_ner_loss': [],
        'test_acc': []
    }

    model.to(device)
    
    for epoch in tqdm(epochs, desc='Iteration'):
        train_loss, train_ner_loss = train_step(model=model,
                                                optimizer=optimizer,
                                                scheduler=scheduler,
                                                dataloader=train_dataloader,
                                                device=device,
                                                hparams=hparams)
        test_loss, test_ner_loss, test_acc = test_step(model=model,
                                                       optimizer=optimizer,
                                                       scheduler=scheduler,
                                                       dataloader=test_dataloader,
                                                       device=device,
                                                       hparams=hparams,
                                                       epoch=epoch)
        
        writer.add_scalars(main_tag="Loss",
                                    tag_scalar_dict={"train_loss": train_loss,
                                                    "test_loss": test_loss},
                                    global_step=epoch)
        writer.add_scalars(main_tag="Accuracy",
                            tag_scalar_dict={"test_acc": test_acc},
                            global_step=epoch)
        
        results["train_loss"].append(train_loss.detach().cpu().numpy())
        results["train_ner_loss"].append(train_ner_loss.detach().cpu().numpy())
        results["test_loss"].append(test_loss.detach().cpu().numpy())
        results["test_ner_loss"].append(test_ner_loss.detach().cpu().numpy())
        results["test_acc"].append(test_acc)

    writer.close()
    return results