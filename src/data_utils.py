import re
import itertools
import argparse

from dataclasses import dataclass
from typing import List, Dict
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
    attention_mask: List[int]
    ner_mask: List[int]
    acs_label: int = None
    ner_labels: List[int] = None

def process_label(text, labels):
    annotations = []
    for label in labels.split('; '):
        aspect_category, target, sentiment, opinion = re.sub(r'\{|\}', '', label).split(',')
        
        target_span = extract_span(text, target)
        opinion_span = extract_span(text, opinion
                                    )
        annotations.append(
           {
              'aspect_category': aspect_category,
              'sentiment': sentiment,
              'opinion': {'text': opinion, 'start': opinion_span[0], 'end': opinion_span[1], 'label': 'Opinion'},
              'target':  {'text': target, 'start': target_span[0], 'end': target_span[1], 'label': 'Target'}
           }
        )
    return annotations


def extract_span(text, unit):
    if unit == 'null':
        return (-1, -1)
    start = text.find(unit)
    end = start + len(unit)
    return (start, end)


def align_tokens_and_annotations_bio(tokenized: Encoding, ner_labels, annotations: LabelDict, ids_to_label):
    aligned_labels = list(map(ids_to_label.get, ner_labels))
    annotations = [annotations.target, annotations.opinion]
    for anno in annotations:
        if anno['text'] == 'null':
            continue
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

def get_ner_mask(tokenized: Encoding, max_seq_length: int, sep_id: int):
    try:
        sep_idx = tokenized.ids.index(sep_id)
        ner_mask = [1] * sep_idx + [0] * (max_seq_length - sep_idx)
    except:
        print(sep_id)
        print(tokenized.ids)
        exit()
    return ner_mask


def normalize_label(aspect_category, sentiment):
    aspect_category = re.sub(r'#', ' ', aspect_category.lower())
    aspect_category = re.sub(r'&', '_', aspect_category)
    return aspect_category + ' ' + sentiment

def get_acs(aspect_category_path, sentiment_path):

    with open(aspect_category_path) as f:
        aspect_category_set = f.read().split('\n')
    with open(sentiment_path) as f:
        sentiment_set = f.read().split('\n')

    compose_set = []
    for (aspect_category, sentiment) in itertools.product(aspect_category_set, sentiment_set):
        compose_set.append(normalize_label(aspect_category, sentiment))

    return compose_set
     


def main():
    print("[INFO] Testing...")
    print(get_acs('/workspaces/ACOS_BERT/data/ViRes/aspect_category_set.txt', '/workspaces/ACOS_BERT/data/ViRes/sentiment_set.txt'))

if __name__ == "__main__":
   main()