import itertools
from dataclasses import dataclass
from typing import List, Dict
from tqdm.auto import tqdm
from ftfy import fix_text

from utils import align_tokens_and_annotations_bio, process_label, get_ner_mask, normalize_label, get_acs

import torch
from torch.utils.data import DataLoader, Dataset

from transformers import PreTrainedTokenizerFast, BatchEncoding
from tokenizers import Encoding

@dataclass
class RawExample:
    id: str
    text: str
    labels: str

@dataclass
class LabelDict:
    aspect_category: str
    sentiment: str
    target: Dict
    opinion: Dict

@dataclass
class ProcessedExample:
    id: str
    text: str
    labels: List[LabelDict]

@dataclass
class InputExample:
    example_id: str
    input_ids: List[int]
    token_type_ids: List[int]
    attention_mask: List[int]
    ner_mask: List[int]
    acs_label: int = None
    ner_labels: List[int] = None

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

  def get_aligned_label_ids_from_annotations(self, tokenized_text: Encoding, annotations: List[Dict]):
    raw_labels = align_tokens_and_annotations_bio(tokenized_text, annotations)
    return list(map(self.labels_to_id.get, raw_labels))



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
                id = fix_text(id)
                text = fix_text(text)
                labels = fix_text(labels)
                raw_examples.append(
                   RawExample(id, text, labels)
                )               
        return raw_examples

    def process_data(self, path):
        examples = self.read_data(path)
        processed_examples: List[ProcessedExample] = []
        for example in tqdm(examples, desc='[INFO] Loading data...'):
            annotations = process_label(example.text, example.labels)
            labels: LabelDict = []
            for idx, anno in enumerate(annotations):
                labels.append(
                    LabelDict(
                        aspect_category=anno['aspect_category'],
                        sentiment=anno['sentiment'],
                        target=anno['target'],
                        opinion=anno['opinion']
                    )
                )
            processed_examples.append(
                ProcessedExample(
                    id=example.id,
                    text=example.text,
                    labels=labels
                )
            )
        return processed_examples
    


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
        self.max_seq_length = max_seq_length
        self.compose_to_id = {compose: idx for idx, compose in enumerate(compose_set)}
        self.id_to_compose = {idx: compose for idx, compose in enumerate(compose_set)}
        self.examples = self.process(examples)

    def process(self, examples: List[ProcessedExample]):
        pre_id = "##"
        is_first = True
        total_examples: List[InputExample] = []
        input_examples: List[InputExample] = []   
        for example in tqdm(examples, desc='[INFO] Processing data...'):
            for ac_s in self.compose_set:
                example_id = example.id.split(':')[0] + f':{self.compose_to_id[ac_s]}'
                tokenized: Encoding = self.tokenizer(example.text, ac_s, padding='max_length', max_length=self.max_seq_length)[0]
                ner_mask = get_ner_mask(tokenized, self.max_seq_length)
                if ac_s == normalize_label(example.aspect_category, example.sentiment):
                    acs_label = 1
                    ner_labels = self.label_set.get_aligned_label_ids_from_annotations(tokenized, [example.target, example.opinion])
                else:
                    acs_label = 0
                    ner_labels = self.label_set.labels_to_id['O'] * sum(ner_mask)

                input_examples.append(
                    InputExample(
                        example_id=example_id,
                        input_ids=tokenized.ids,
                        token_type_ids=tokenized.type_ids,
                        attention_mask=tokenized.attention_mask,
                        ner_mask=ner_mask,
                        acs_label=acs_label,
                        ner_labels=ner_labels
                    )
                )
        return total_examples
            

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def main():
    processor = DataProcessor('/workspaces/ACOS_BERT/data/ViRes')
    label_set = LabelSet(['Target', 'Opinion'])
    compose_set = get_acs('/workspaces/ACOS_BERT/data/ViRes/aspect_category_set.txt', '/workspaces/ACOS_BERT/data/ViRes/sentiment_set.txt')
    for example in processor.train_examples:
        if example.opinion['text'] == 'negative':
            print(example)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('trituenhantaoio/bert-base-vietnamese-uncased')
    training_dataset = TrainingDataset(processor.train_examples, label_set, compose_set, tokenizer, 128)
    training_dataset = TrainingDataset(processor.dev_examples, label_set, compose_set, tokenizer, 128)
    training_dataset = TrainingDataset(processor.test_examples, label_set, compose_set, tokenizer, 128)
if __name__ == "__main__":
  main()
