import argparse
from pathlib import Path
import json
from typing import List
from collections import Counter

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

def extract_sequence_ids(sequence_labels: List[str]) -> List[List[int]]:
    all_indexs = []
    sequence_indexs = []
    same_target = False
    for index, label in enumerate(sequence_labels):
        if label == 'B':
            if same_target == True:
                all_indexs.append(sequence_indexs)
                sequence_indexs = []
                same_target = False

            same_target = True
            sequence_indexs.append(index)
        elif label == 'I':
            sequence_indexs.append(index)
        elif label == 'O':
            if same_target:
                all_indexs.append(sequence_indexs)
                sequence_indexs = []
                same_target = False
        else:
            raise ValueError('Sequence labels should only be `B`, `I`, or'
                             f' `O` and not {label} out of {sequence_labels}')
    if sequence_indexs != []:
        all_indexs.append(sequence_indexs)
    return all_indexs
                

if __name__=='__main__':

    target_dataset_help = 'The dataset that contains the targets, '\
                          'confidence scores, tokens and text'
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dataset", type=parse_path, 
                        help=target_dataset_help)
    args = parser.parse_args()

    extracted_targets = Counter()
    with args.target_dataset.open('r') as dataset_file:
        count = 0
        for line in dataset_file:
            targets = []
            if line:
                data = json.loads(line)
                tokens = data['tokens']
                text = data['text']
                confidence = data['label_confidence']
                labels = data['predicted_sequence_labels']
                sequence_indexs = extract_sequence_ids(labels)
                for index_list in sequence_indexs:
                    target_tokens = []
                    for index in index_list:
                        target_tokens.append(tokens[index])
                    target = ' '.join(target_tokens)
                    targets.append(target)
            extracted_targets.update(targets)
            count += 1
    print(sum(extracted_targets.values()))
