import argparse
from pathlib import Path
import json
from typing import List
from collections import Counter
import statistics

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
    min_confidence_help = 'Minimum confidence level to determine it is a target'
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dataset", type=parse_path, 
                        help=target_dataset_help)
    parser.add_argument("min_confidence", type=float, 
                        help=min_confidence_help)
    parser.add_argument("train_dataset", type=parse_path, help='TDSA Training data')
    parser.add_argument("test_dataset", type=parse_path, help='TDSA Test data')
    args = parser.parse_args()

    min_confidence = args.min_confidence
    extracted_targets = Counter()
    average_target_confidences = []
    with args.target_dataset.open('r') as dataset_file:
        count = 0
        for line in dataset_file:
            targets = []
            if line:
                data = json.loads(line)
                tokens = data['tokens']
                text = data['text']
                confidences = data['label_confidence']
                labels = data['predicted_sequence_labels']
                sequence_indexs = extract_sequence_ids(labels)
                for index_list in sequence_indexs:
                    target_confidences = []
                    target_tokens = []
                    pass_confidence_threshold = True
                    for index in index_list:
                        target_tokens.append(tokens[index])
                        confidence = confidences[index]
                        target_confidences.append(confidence)
                        if not confidence > min_confidence:
                            pass_confidence_threshold = False
                    if pass_confidence_threshold:
                        target = ' '.join(target_tokens)
                        targets.append(target)
                    average_confidence = statistics.mean(target_confidences)
                    average_target_confidences.append(average_confidence)
            extracted_targets.update(targets)
            count += 1
    print(len(extracted_targets))
    print(sum(extracted_targets.values()))
    lower_targets = set()
    for target in extracted_targets:
        lower_targets.add(target.lower())
    print(len(lower_targets))
