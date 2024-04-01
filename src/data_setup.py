import os
import itertools
from typing import List, Dict
from tqdm.auto import tqdm
from ftfy import fix_text

from data_utils import *

import torch
from torch.utils.data import DataLoader, Dataset

from transformers import PreTrainedTokenizerFast, BatchEncoding
from tokenizers import Encoding



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

  def get_aligned_label_ids_from_annotations(self, tokenized_text: Encoding, ner_labels, annotations: LabelDict):
    raw_labels = align_tokens_and_annotations_bio(tokenized_text, ner_labels, annotations, self.ids_to_label)
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
        self.compose_to_id = {compose: str(idx) for idx, compose in enumerate(compose_set)}
        self.id_to_compose = {str(idx): compose for idx, compose in enumerate(compose_set)}
        self.sep_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        self.examples = self.process(examples)

    def process(self, examples: List[ProcessedExample]):
        total_examples: List[InputExample] = []
        for example in tqdm(examples, desc='[INFO] Processing data...'):
            total_examples.extend(self.generate_acs_examples(example))
        return total_examples

    def generate_acs_examples(self, example: ProcessedExample):
        acs_examples: List[InputExample] = []
        for acs in self.compose_set:
            tokenized: Encoding = self.tokenizer(example.text, acs, padding='max_length', max_length=self.max_seq_length)[0]
            example_id = f'{example.id}:{self.compose_to_id[acs]}'
            ner_mask = get_ner_mask(tokenized, self.max_seq_length, self.sep_id)
            acs_label = 0
            ner_labels = [self.label_set.labels_to_id['O']] * sum(ner_mask) + [self.label_set.labels_to_id['[PAD]']] * (self.max_seq_length - sum(ner_mask))
            ner_labels[0] = self.label_set.labels_to_id['[CLS]']
            acs_examples.append(
                InputExample(
                    example_id=example_id,
                    input_ids=tokenized.ids,
                    attention_mask=tokenized.attention_mask,
                    ner_mask=ner_mask,
                    acs_label=acs_label,
                    ner_labels=ner_labels
                )
            )
        for label in example.labels:
            tokenized = self.tokenizer(example.text, acs, padding='max_length', max_length=self.max_seq_length)[0]
            acs_label = normalize_label(label.aspect_category, label.sentiment)
            acs_id = self.compose_to_id[acs_label]
            for acs_example in acs_examples:
                if acs_id == acs_example.example_id.split(':')[-1]:
                    acs_example.acs_label = 1
                    acs_example.ner_labels = self.label_set.get_aligned_label_ids_from_annotations(tokenized, ner_labels, label)
                    break

        return acs_examples 

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

class TrainingBatch:
    def __init__(self, examples: List[InputExample]):
        self.input_ids = torch.LongTensor([example.input_ids for example in examples])
        self.attention_mask = torch.LongTensor([example.attention_mask for example in examples])
        self.ner_mask = torch.LongTensor([example.ner_mask for example in examples])
        self.acs_label = torch.LongTensor([example.acs_label for example in examples])
        self.ner_labels = torch.LongTensor([example.ner_labels for example in examples])

    def __getitem__(self, item):
        return getattr(self, item)

def main():
    import warnings
    warnings.filterwarnings('ignore')
    processor = DataProcessor('../data/ViRes')
    label_set = LabelSet(['Target', 'Opinion'])
    compose_set = get_acs('../data/ViRes/aspect_category_set.txt', '../data/ViRes/sentiment_set.txt')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('trituenhantaoio/bert-base-vietnamese-uncased')
    training_dataset = TrainingDataset(processor.dev_examples, label_set, compose_set, tokenizer, 128)
    train_dataloader = DataLoader(training_dataset, collate_fn=TrainingBatch, batch_size=len(compose_set), num_workers=os.cpu_count())
    for batch in train_dataloader:
        print(batch.input_ids.shape)
        break
    # training_dataset = TrainingDataset(processor.dev_examples, label_set, compose_set, tokenizer, 128)
    # # training_dataset = TrainingDataset(processor.test_examples, label_set, compose_set, tokenizer, 128)
    # for example in tqdm(processor.dev_examples):
    #     if example.id == '#214':
    #         print(example.labels)
    # for example in tqdm(training_dataset.examples):
    #     if example.example_id.split(':')[0] == '#214':
    #         print('=' * 50)
    #         print(training_dataset.id_to_compose[example.example_id.split(':')[-1]])
    #         print(list(map(label_set.ids_to_label.get, example.ner_labels)))
    #         print('='*50)
if __name__ == "__main__":
  main()
