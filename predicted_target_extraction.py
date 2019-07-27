import argparse
import json
from pathlib import Path
from typing import List

from target_extraction.data_types import TargetText, TargetTextCollection
from target_extraction.dataset_parsers import semeval_2014, wang_2017_election_twitter_test, wang_2017_election_twitter_train

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':
    train_fp_help = 'File path to the original training data that is '\
                     'associated to the predicted targets'
    test_fp_help = 'File path to the original test data that is '\
                   'associated to the predicted targets'
    confidence_score_help = "The confidence that the predicted target is a "\
                            "target based on the model's confidence scores"
    parser = argparse.ArgumentParser()
    parser.add_argument("predicted_target_data_fp", type=parse_path, 
                        help='File path to the predicted targets')
    parser.add_argument("train_fp", type=parse_path, 
                        help=train_fp_help,
                        default='./results/baseline')
    parser.add_argument("test_fp", type=parse_path, 
                        help=test_fp_help)
    parser.add_argument('confidence_score', type=float, 
                        help=confidence_score_help)
    parser.add_argument('--election', action='store_true', 
                        help='Whether the dataset is the Election Twitter dataset')
    args = parser.parse_args()

    # Setting the data up
    predicted_fp = args.predicted_target_data_fp
    if not args.election:
        train_fp = args.train_fp
        test_fp = args.test_fp
        train_data = semeval_2014(train_fp, False)
        test_data = semeval_2014(test_fp, False)
    else:
        temp_dir = Path('/tmp', 'election twitter dataset')
        train_data = wang_2017_election_twitter_train(temp_dir)
        test_data = wang_2017_election_twitter_test(temp_dir)
    train_targets = list(train_data.target_count(lower=True).keys())
    test_targets = list(test_data.target_count(lower=True).keys())

    acceptable_confidence = args.confidence_score
    all_targets: List[TargetText] = []
    with predicted_fp.open('r') as predicted_file:
        for index, line in enumerate(predicted_file):
            target_data = json.loads(line)
            target_id = str(index)
            target_data_dict = {'text': target_data['text'], 'text_id': target_id,
                                'confidences': target_data['confidence'], 
                                'sequence_labels': target_data['sequence_labels'], 
                                'tokenized_text': target_data['tokens']}
            target_data = TargetText.target_text_from_prediction(**target_data_dict, 
                                                                confidence=acceptable_confidence)
            if target_data['targets']:
                all_targets.append(target_data)
    all_targets: TargetTextCollection = TargetTextCollection(all_targets)
    pred_target_dict = all_targets.target_count(lower=True)
    pred_targets = set(list(pred_target_dict.keys()))
    train_count = 0
    for train_target in train_targets:
        if train_target in pred_targets:
            train_count += 1
    percentage_train_targets = (train_count / len(train_targets)) * 100
    print(f'Percentage of targets that have been predicted that are in train: {percentage_train_targets}')

    test_count = 0
    for test_target in test_targets:
        if test_target in pred_targets:
            test_count += 1
    percentage_test_targets = (test_count / len(test_targets)) * 100
    print(f'Percentage of targets that have been predicted that are in test: {percentage_test_targets}')

    train_test = TargetTextCollection.combine(train_data, test_data)
    train_test_in_count = 0
    train_test_out_count = 0
    train_test_targets = train_test.target_count(lower=True)
    for train_test_target in train_test_targets:
        if train_test_target in pred_targets:
            train_test_in_count += 1
        else:
            train_test_out_count += 1
    print(f'Number of new predicted targets that are in the whole gold datasets: '
        f'{train_test_in_count} compared to that are not: {train_test_out_count}')

    train_and_pred = TargetTextCollection.combine(train_data, all_targets)
    train_and_pred_targets = set(list(train_and_pred.target_count(lower=True).keys()))
    test_in_count = 0
    test_in_pred_count = 0
    test_in_train_count = 0
    test_out_count = 0
    test_out_pred_count = 0
    test_out_train_count = 0
    for test_target in test_targets:
        if test_target in train_and_pred_targets:
            test_in_count += 1
        else:
            test_out_count += 1
        if test_target in pred_targets:
            test_in_pred_count += 1
        else:
            test_out_pred_count += 1
        if test_target in train_targets:
            test_in_train_count += 1
        else:
            test_out_train_count += 1
    print(f'Number of new predicted and training targets that are in the test datasets: '
        f'{test_in_count} compared to that are not: {test_out_count}')
    print(f'Number of new predicted targets that are in the test datasets: '
        f'{test_in_pred_count} compared to that are not: {test_out_pred_count}')
    print(f'Number of new training targets that are in the test datasets: '
        f'{test_in_train_count} compared to that are not: {test_out_train_count}')
    print(f'Total number of predicted targets: {len(pred_targets)}')
    print(f'Number of targets in train: {len(train_targets)}')
    print(f'Number of targets in test: {len(test_targets)}')
    print(f'Number of targets in train and test: {len(train_test_targets)}')