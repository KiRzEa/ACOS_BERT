import itertools
from dataclasses import dataclass
from typing import List, Dict

from utils import align_tokens_and_annotations_bio, process_label

import torch
from torch.utils.data import DataLoader, Dataset

from transformers import PreTrainedTokenizerFast, BatchEncoding


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
    raw_labels = align_tokens_and_annotations_bio(tokenized_text, annotations)
    return list(map(self.labels_to_id.get, raw_labels))

@dataclass
class RawExample:
    id: str
    text: str
    labels: str
    
@dataclass
class ProcessedExample:
    id: str
    text: str
    aspect_category: str
    sentiment: str
    target: Dict
    opinion: Dict


class DataProcessor:
    def __init__(self, data_dir):
        self.raw_examples = self.read_data(data_dir)
    
    def read_data(self, data_dir):

        def strip(text):
            return text.strip()
        
        raw_examples = []
        with open(data_dir) as f:
            content = f.read()

            for example in content.split('\n\n'):
                id, text, labels = map(strip, example.split('\n'))
                raw_examples.append(
                   RawExample(id, text, labels)
                )               
        return raw_examples

    def extract_spans_and_sentence(self, examples):
        processed_examples = []
        for example in examples:
            annotations = process_label(example.text, example.labels)
            for idx, anno in enumerate(annotations):
                processed_examples.append(
                    ProcessedExample(
                        id=f'{id}:{idx}',
                        text=example.text,
                        aspect_category=annotations['aspect_category'],
                        sentiment=annotations['sentiment'],
                        target=annotations['target'],
                        opinion=annotations['opinion']
                    )
                )
        return processed_examples
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
