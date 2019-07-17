import argparse
from pathlib import Path
import json

from sklearn.model_selection import train_test_split
import target_extraction
from target_extraction.data_types import TargetTextCollection
from target_extraction.dataset_parsers import semeval_2014, wang_2017_election_twitter_test, wang_2017_election_twitter_train
from target_extraction.tokenizers import spacy_tokenizer, ark_twokenize
from target_extraction.allen import AllenNLPModel


def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_fp", type=parse_path,
                        help='File path to the train data')
    parser.add_argument("--test_fp", type=parse_path,
                        help='File path to the test data')
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
    if dataset_name == 'election_twitter':
        tokenizer = ark_twokenize
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

    model_name = f'{dataset_name} model'
    model = AllenNLPModel(model_name, args.model_config, 'target-tagger', 
                          args.model_save_dir)
    print('Fitting model')
    model.fit(train_data, val_data, test_data)
    print('Finished fitting model')
    first = True
    with args.output_data_fp.open('w+') as output_data_file:
        with args.data_fp.open('r') as data_file:
            for predictions in model.predict_sequences(data_file):
                for prediction in predictions:
                    prediction_str = json.dumps(prediction)
                    if first:
                        first = False
                    else:
                        prediction_str = f'\n{prediction_str}'
                    output_data_file.write(prediction_str)
