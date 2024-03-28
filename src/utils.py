import re
import itertools
import argparse

from tokenizers import Encoding

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


def align_tokens_and_annotations_bio(tokenized: Encoding, annotations):
    tokens = tokenized.tokens
    aligned_labels =  ['O'] * len(tokens)
    aligned_labels[0] = "[CLS]"

    for anno in annotations:
        if anno['text'] == 'null':
          continue
        annotation_token_idx_set = set()
        for char_idx in range(anno['start'], anno['end']):
            try:
                token_idx = tokenized.char_to_token(char_idx)
            except:
                print(tokenized.tokens)
                print(anno)
                print(char_idx)
                print(anno['start'])
                print(anno['end'])
                exit()
            if token_idx is not None:
                annotation_token_idx_set.add(token_idx)

      
        for num, token_idx in enumerate(sorted(annotation_token_idx_set)):
            if num == 0:
                prefix = 'B'
            else:
                prefix = 'I'
            aligned_labels[token_idx] = f"{prefix}-{anno['label']}"

    return aligned_labels

def get_ner_mask(tokenized: Encoding, max_seq_length: int):
    segment_ids = tokenized.type_ids
    sep_idx = segment_ids.index(1) - 1
    ner_mask = [1] * sep_idx + [0] * (max_seq_length - sep_idx)

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