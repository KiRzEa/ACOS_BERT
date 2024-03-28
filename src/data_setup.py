import itertools
from dataclasses import dataclass
from typing import List, Dict

from utils import align_tokens_and_annotations_bio, process_label, get_ner_mask, normalize_label

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
    print(raw_labels)
    print(list(map(self.labels_to_id.get, raw_labels)))
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
        self.DATA_DIR = data_dir
        self.train_examples = self.process_data(data_dir + '/Train.txt')
        self.dev_examples = self.process_data(data_dir + '/Dev.txt')
        self.test_examples = self.process_data(data_dir + '/Test.txt')

    def read_data(self, data_dir):

        def strip(text):
            return text.strip()
        print(data_dir)
        raw_examples = []
        with open(data_dir) as f:
            content = f.read()

            for example in content.split('\n\n'):
                id, text, labels = map(strip, example.split('\n'))
                raw_examples.append(
                   RawExample(id, text, labels)
                )               
        return raw_examples

    def process_data(self, path):
        examples = self.read_data(path)
        processed_examples = []
        for example in examples:
            annotations = process_label(example.text, example.labels)
            for idx, anno in enumerate(annotations):
                processed_examples.append(
                    ProcessedExample(
                        id=f'{example.id}:{idx}',
                        text=example.text,
                        aspect_category=anno['aspect_category'],
                        sentiment=anno['sentiment'],
                        target=anno['target'],
                        opinion=anno['opinion']
                    )
                )
        return processed_examples
    
@dataclass
class InputExample:
    _id: str
    input_ids: List[int]
    token_type_ids: List[int]
    attention_mask: List[int]
    ner_mask: List[int]
    acs_label: int = None
    ner_labels: List[int] = None

class TrainingDataset(Dataset):
    def __init__(self,
                 examples: List[ProcessedExample],
                 label_set: LabelSet,
                 compose_set: List[str],
                 tokenizer: PreTrainedTokenizerFast,
                 max_seq_length: int
                 ):
        self.label_set = label_set
        self.tokenizer = tokenizer
        self.compose_set = compose_set
        self.examples = self.process(examples)
        self.max_seq_length = max_seq_length
        self.compose_to_id = {compose: idx for idx, compose in enumerate(compose_set.items())}
        self.id_to_compose = {idx: compose for idx, compose in enumerate(compose_set.items())}
        

    def process(self, examples: List[ProcessedExample]):
        pre_id = "##"
        for example in examples:
            if pre_id != example.id:
                for ac_s in self.compose_set:
                    example_id = example.id.split(':')
                    tokenized = self.tokenizer(example.text, ac_s, padding='max_length', max_length=self.max_seq_length)
                    ner_mask = get_ner_mask(tokenized, self.max_seq_length)
                    asc_label = 0
                    ner_labels = ['O'] * sum(ner_mask)

                    
            else:
                # acs_label = normalize_label(example.aspect_category, example.sentiment)
                pass

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def main():
  processor = DataProcessor('/content/ACOS_BERT/data/ViRes')

if __name__ == "__main__":
  main()
