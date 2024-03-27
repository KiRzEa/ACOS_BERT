import itertools
from dataclasses import dataclass

from typing import List

import torch
from torch.utils.data import DataLoader, Dataset

from transformers import PreTrainedTokenizerFast, BatchEncoding
from tokenizers import Encoding

def extract_spans_and_sentence(text_with_annotations):
   


def align_tokens_and_annotations_bilou(tokenized: Encoding, annotations):
  tokens = tokenized.tokens
  aligned_labels =  ['O'] * len(tokens)
  for anno in annotations:
    annotation_token_idx_set = set()
    for char_idx in range(anno['start'], anno['end']):
      token_idx = tokenized.char_to_token(char_idx)
      if token_idx is not None:
        annotation_token_idx_set.add(token_idx)
    if len(annotation_token_idx_set) == 1:
      token_idx  = annotation_token_idx_set.pop()
      prefix = (
          "U"
      )
      aligned_labels[token_idx] = f"{prefix}-{anno['label']}"
    else:
      last_token_in_anno_idx = len(annotation_token_idx_set) - 1
      for num, token_idx in enumerate(sorted(annotation_token_idx_set)):
        if num == 0:
          prefix = 'B'
        elif num == last_token_in_anno_idx:
          prefix = 'L'
        else:
          prefix = 'I'
        aligned_labels[token_idx] = f"{prefix}-{anno['label']}"
  return aligned_labels

class LabelSet:
  def __init__(self, labels: List[str], scheme="BIO"):
    self.labels_to_id = {}
    self.ids_to_label = {}
    self.labels_to_id["O"] = 0
    self.ids_to_label[0] = "O"

    match scheme:
        case "TO":
            self.scheme = "T"
        case "BIO":
            self.scheme = "BI"
        case "BILOU":
            self.scheme = "BILU"

    num = 0
    for _num, (label, s) in enumerate(itertools.product(labels, self.scheme)):
      num = _num + 1
      l = f"{s}-{label}"
      self.labels_to_id[l] = num
      self.ids_to_label[num] = l

  def get_aligned_label_ids_from_annotations(self, tokenized_text, annotations):
    raw_labels = align_tokens_and_annotations_bilou(tokenized_text, annotations)
    return list(map(self.labels_to_id.get, raw_labels))
  
@dataclass
class InputExample:
    input_ids: IntList
    token_type_ids: IntList
    attention_mask: IntList
    ner_mask: IntList
    as_label: Int = None
    ner_labels: IntList = None

class TrainingDataset(Dataset):
    def __init__(self,
                 data,
                 label_set: LabelSet,
                 tokenizer: PreTrainedTokenizerFast,
                 ):
        self.label_set = label_set
        self.tokenizer = tokenizer

        raise NotImplementedError