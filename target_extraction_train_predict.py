import argparse
from pathlib import Path
import json
from typing import Iterable
import tempfile
import random

from allennlp.models import Model
from sklearn.model_selection import train_test_split
import target_extraction
from target_extraction.data_types import TargetTextCollection
from target_extraction.dataset_parsers import semeval_2014, wang_2017_election_twitter_test, wang_2017_election_twitter_train
from target_extraction.tokenizers import spacy_tokenizer, ark_twokenize
from target_extraction.allen import AllenNLPModel


def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

def text_to_json(text_fp: Path) -> Iterable[str]:
    with text_fp.open('r') as text_file:
        for line in text_file:
            line = line.strip()
            if line:
                tokens = line.split()
                yield {'text': line, 'tokens': tokens}

def predict_on_file(input_fp: Path, output_fp: Path, model: Model, batch_size: int) -> None:
    first = True
    with output_fp.open('w+') as output_data_file:
        for prediction in model.predict_sequences(text_to_json(input_fp), 
                                                  batch_size=batch_size):
            prediction_str = json.dumps(prediction)
            if first:
                first = False
            else:
                prediction_str = f'\n{prediction_str}'
            output_data_file.write(prediction_str)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_fp", type=parse_path,
                        help='File path to the train data')
    parser.add_argument("--test_fp", type=parse_path,
                        help='File path to the test data')
    parser.add_argument("--number_to_predict_on", type=int, 
                        help='Sub sample the data until this number of samples are left')
    parser.add_argument("--batch_size", type=int, default=64,
                        help='Batch size. Higher this is the more memory you need')
    parser.add_argument('dataset_name', type=str, 
                        choices=['semeval_2014', 'election_twitter'],
                        help='dataset that is to be trained and predicted')
    parser.add_argument('model_config', type=parse_path,
                        help='File Path to the Model configuration file')
    parser.add_argument('model_save_dir', type=parse_path, 
                        help='Directory to save the trained model')
    parser.add_argument('data_fp', type=parse_path, 
                        help='File Path to the data to predict on')
    parser.add_argument('output_data_fp', type=parse_path, 
                        help='File Path to the output predictions')
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    model_name = f'{dataset_name} model'
    model = AllenNLPModel(model_name, args.model_config, 'target-tagger', 
                          args.model_save_dir)

    if not args.model_save_dir.is_dir():

        if dataset_name == 'semeval_2014':
            if not args.train_fp or not args.test_fp:
                raise ValueError('If training and predicting for the SemEval '
                                'datasets the training and test file paths must '
                                'be given')
            train_data = semeval_2014(args.train_fp, False)
            test_data = semeval_2014(args.test_fp, False)
        else:
            temp_election_directory = Path('/tmp/election_dataset_dir')
            train_data = wang_2017_election_twitter_train(temp_election_directory)
            test_data = wang_2017_election_twitter_test(temp_election_directory)
        # Use the same size validation as the test data
        test_size = len(test_data)
        # Create the train and validation splits
        train_data = list(train_data.values())
        train_data, val_data = train_test_split(train_data, test_size=test_size)
        train_data = TargetTextCollection(train_data)
        val_data = TargetTextCollection(val_data)
        # Tokenize the data
        datasets = [train_data, val_data, test_data]
        tokenizer = spacy_tokenizer()

        sizes = []
        for dataset in datasets:
            dataset.tokenize(tokenizer)
            returned_errors = dataset.sequence_labels(return_errors=True)
            if returned_errors:
                for error in returned_errors:
                    error_id = error['text_id']
                    del dataset[error_id]
            returned_errors = dataset.sequence_labels(return_errors=True)
            if returned_errors:
                raise ValueError('Sequence label errors are still persisting')
            sizes.append(len(dataset))
        print(f'Lengths Train: {sizes[0]}, Validation: {sizes[1]}, Test: {sizes[2]}')
        print('Fitting model')
        model.fit(train_data, val_data, test_data)
        print('Finished fitting model\nNow Evaluating model:')
        test_iter = iter(test_data.values())
        for test_pred in model.predict_sequences(test_data.values(), batch_size=256):
            relevant_test = next(test_iter)
            relevant_test['predicted_sequence_labels'] = test_pred['sequence_labels']
        test_scores = test_data.exact_match_score('predicted_sequence_labels')
        print(f'Test F1 scores: {test_scores[2]}')
    else:
        model.load(cuda_device=0)
    first = True
    data_fp = args.data_fp
    from time import time
    t = time() 
    if args.number_to_predict_on:
        data_count = 0
        with data_fp.open('r') as data_file:
            for line in data_file:
                data_count += 1
        if data_count <= args.number_to_predict_on:
            raise ValueError(f'Number of lines in the data file {data_count} '
                             'to predict on is less than or equal to the number'
                             f' of lines to sub-sample {args.number_to_predict_on}')
        lines_numbers_to_subsample = random.sample(range(data_count), 
                                                   k=args.number_to_predict_on)
        lines_numbers_to_subsample = set(lines_numbers_to_subsample)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_fp = Path(temp_dir, 'temp_input_file.txt')
            with temp_fp.open('w+') as temp_file:
                with data_fp.open('r') as data_file:
                    for index, line in enumerate(data_file):
                        if index in lines_numbers_to_subsample:
                            temp_file.write(line)
            print(f'subsampled data {args.number_to_predict_on} lines')
            predict_on_file(temp_fp, args.output_data_fp, model, args.batch_size)
    else:
        predict_on_file(data_fp, args.output_data_fp, model, args.batch_size)
    print(f'Done took {time() - t}')
