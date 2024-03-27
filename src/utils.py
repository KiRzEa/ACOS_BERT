import re

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
              'opinion': {'text': opinion, 'start': opinion_span[0], 'end': opinion_span[1]},
              'target':  {'text': target, 'start': target_span[0], 'end': target_span[1]}
           }
        )
    return annotations

def extract_span(text, unit):
    if unit == 'null':
        return (-1, -1)
    start = text.find(unit)
    end = start + len(unit)
    return (start, end)

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

    
def main():
   print("[INFO] Testing...")
if __name__ == "__main__":
   main()