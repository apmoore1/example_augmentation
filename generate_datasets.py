import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union
import json

from bella.parsers import semeval_14, election_test, election_train
from bella.data_types import Target, TargetCollection

def generate_stats(data_path: Path) -> Dict[str, Union[int, float]]:
    target_data = []
    with data_path.open('r') as data_lines:
        for line in data_lines:
            line = json.loads(line)
            line['spans'] = [tuple(span) for span in line['spans']]
            target_data.append(Target(**line))
    target_data = TargetCollection(target_data)
    target_stats = defaultdict(lambda: 0)
    data_size = len(target_data)
    target_stats['size'] = data_size
    for i in range(1, 3):
        target_stats[f'Distinct sentiment {i}'] = len(target_data.subset_by_sentiment(i))
    for data in target_data.data_dict():
        target_stats[data['sentiment']] += 1
    for key, value in target_stats.items():
        if key == 'size':
            continue
        target_stats[key] = value / data_size
    return target_stats

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    data_dir_help = 'File path to the directory that currently '\
                    'stores the data'
    data_splits_help = 'File path to the directory to store the dataset splits'
    parser.add_argument("data_dir", help=data_dir_help, type=parse_path)
    parser.add_argument("data_splits_dir", help=data_splits_help, 
                        type=parse_path)
    args = parser.parse_args()


    data_dir = args.data_dir
    rest_train_fp = Path(data_dir, 'Restaurants_Train_v2.xml')
    rest_test_fp = Path(data_dir, 'Restaurants_Test_Gold.xml')
    laptop_train_fp = Path(data_dir, 'Laptop_Train_v2.xml')
    laptop_test_fp = Path(data_dir, 'Laptops_Test_Gold.xml')
    election_dir = Path(data_dir, 'election')


    rest_train_data = semeval_14(rest_train_fp, name='Restaurant')
    rest_test_data = semeval_14(rest_test_fp, name='Restaurant Test')
    laptop_train_data = semeval_14(laptop_train_fp, name='Laptop')
    laptop_test_data = semeval_14(laptop_test_fp, name='Laptop Test')
    election_train = election_train(election_dir, name='Election')
    election_test = election_test(election_dir, name='Election Test')

    data_splits_dir: Path = args.data_splits_dir
    data_splits_dir.mkdir(parents=True, exist_ok=True)
    all_test_data = [rest_test_data, laptop_test_data, election_test]
    print('Test data')
    for test_data in all_test_data:
        dataset_name = test_data.name
        dataset_path = str(Path(data_splits_dir, dataset_name))
        test_data.to_json_file(dataset_path, cache=False)
        print(f'{dataset_name}')
        for key, value in generate_stats(Path(data_splits_dir, dataset_name)).items():
            print(f'{key}: {value}')

    all_train_data = [rest_train_data, laptop_train_data, election_train]

    for train_data in all_train_data:
        dataset_name = train_data.name
        train_name = f'{dataset_name} Train'
        train_path = str(Path(data_splits_dir, train_name))
        val_name = f'{dataset_name} Val'
        val_path = str(Path(data_splits_dir, val_name))

        train_data.to_json_file([train_path, val_path], split=0.2,
                                cache=False)
        for dataset_type in [train_name, val_name]:
            print(f'{dataset_type}')
            for key, value in generate_stats(Path(data_splits_dir, dataset_type)).items():
                print(f'{key}: {value}')
    #
    # Create the dataset where the target data 
    #
    for test_data in all_test_data:
        dataset_name = test_data.name + 'Group'
        dataset_path = str(Path(data_splits_dir, dataset_name))
        test_data.to_json_file(dataset_path, cache=False, 
                               group_by_sentence=True)

    for train_data in all_train_data:
        dataset_name = train_data.name
        train_name = f'{dataset_name} Train Group'
        train_path = str(Path(data_splits_dir, train_name))
        val_name = f'{dataset_name} Val Group'
        val_path = str(Path(data_splits_dir, val_name))

        train_data.to_json_file([train_path, val_path], 
                                split=0.2, cache=False, group_by_sentence=True)