import itertools
from dataclasses import dataclass

from typing import List

import torch
from torch.utils.data import DataLoader, Dataset

from transformers import PreTrainedTokenizerFast, BatchEncoding
from tokenizers import Encoding

def extract_spans_and_sentence(text_with_annotations):
   raise NotImplementedError

def align_tokens_and_annotations_bilou(tokenized: Encoding, annotations):
  tokens = tokenized.tokens
  aligned_labels =  ['O'] * len(tokens)
  for anno in annotations:
    annotation_token_idx_set = set()
    for char_idx in range(anno['start'], anno['end']):
      token_idx = tokenized.char_to_token(char_idx)
      if token_idx is not None:
        annotation_token_idx_set.add(token_idx)

      
    for num, token_idx in enumerate(sorted(annotation_token_idx_set)):
        if num == 0:
            prefix = 'B'
        else:
            prefix = 'I'
        aligned_labels[token_idx] = f"{prefix}-{anno['label']}"

  return aligned_labels

class LabelSet:
  def __init__(self, labels: List[str]):
    self.labels_to_id = {}
    self.ids_to_label = {}

    
    self.labels_to_id["[PAD]"] = 0
    self.labels_to_id["[CLS]"] = 1
    self.labels_to_id["O"] = 2

    self.ids_to_label[0] = "[PAD]"
    self.ids_to_label[1] = "[CLS]"
    self.ids_to_label[2] = "O"

    num = 2
    for _num, (label, s) in enumerate(itertools.product(labels, "BI")):
      num += 1
      l = f"{s}-{label}"
      print(num, l)
      self.labels_to_id[l] = num
      self.ids_to_label[num] = l

  def get_aligned_label_ids_from_annotations(self, tokenized_text, annotations):
    raw_labels = align_tokens_and_annotations_bilou(tokenized_text, annotations)
    return list(map(self.labels_to_id.get, raw_labels))
  
@dataclass
class InputExample:
    input_ids: List[int]
    token_type_ids: List[int]
    attention_mask: List[int]
    ner_mask: List[int]
    as_label: int = None
    ner_labels: List[int] = None

class TrainingDataset(Dataset):
    def __init__(self,
                 data,
                 label_set: LabelSet,
                 tokenizer: PreTrainedTokenizerFast,
                 ):
        self.label_set = label_set
        self.tokenizer = tokenizer

        raise NotImplementedError