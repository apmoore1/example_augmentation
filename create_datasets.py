import argparse
import json
from pathlib import Path

from bella.data_types import TargetCollection, Target
import numpy as np

from augmentation_helper import exchange_targets

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("expanded_dataset_fp", type=parse_path, 
                        help='File Path to the expanded dataset')
    parser.add_argument("dataset_save_fp", type=parse_path, 
                        help='File path to save the newly created dataset')
    parser.add_argument("k_similar", type=int, 
                        help='Number of similar alternative targets')
    parser.add_argument("--lm", action="store_true", 
                        help="language model expanded dataset")
    parser.add_argument("--embedding", action="store_true", 
                        help="embedding expanded dataset")
    parser.add_argument("--threshold", type=float, 
                        help="Threshold value")
    args = parser.parse_args()

    threshold = args.threshold
    if threshold is None:
        threshold = False
    is_language_model = True
    similarity_field_name = 'alternative_perplexity'
    if args.lm:
        if threshold:
            threshold = 1
    elif args.embedding:
        similarity_field_name = 'alternative_similarity'
        is_language_model = False
    else:
        raise ValueError('Has to be either a language model or embedding '
                         'expanded dataset, you have not stated an option')
    
    k_similar = args.k_similar
    new_target_dataset = []

    with args.expanded_dataset_fp.open('r') as expanded_dataset_file:
        for line in expanded_dataset_file:
            target_data = json.loads(line)
            target_id = target_data['target_id']
            # Filtering based on similarity and threshold
            temp_k = k_similar
            similarity_field = np.array(target_data[similarity_field_name])
            if is_language_model and threshold:
                similarity_field = target_data['original_perplexity'] - similarity_field
                above_original_perplexity_index = np.argmax((similarity_field <= 0) + 0) 
                similarity_field = similarity_field[:above_original_perplexity_index]
                if len(similarity_field) < k_similar:
                    temp_k = len(similarity_field)
            elif (not is_language_model) and threshold:
                above_threshold_index = np.argmin((similarity_field >= threshold) + 0)
                similarity_field = similarity_field[:above_threshold_index]
                if len(similarity_field) < k_similar:
                    temp_k = len(similarity_field)
            # For each of the filtered alternative targets it creates a json 
            # like object that will be used to store it in a collection to then 
            # save to a json file
            alternative_targets = target_data['alternative_targets'][:temp_k]
            for index, alternative_target in enumerate(alternative_targets):
                new_target_id = f'{target_id}_{index}'
                new_target_data = exchange_targets(target_data, alternative_target, 
                                                   new_target_id)
                # sanitizing the target dataset.
                new_target_data.pop('alternative_targets')
                new_target_data.pop(similarity_field_name)
                if is_language_model:
                    new_target_data.pop('original_perplexity')
                new_target_dataset.append(Target(**new_target_data))
    print(f'Size of the expanded dataset {len(new_target_dataset)}')
    new_target_dataset = TargetCollection(new_target_dataset)
    new_target_dataset.to_json_file(str(args.dataset_save_fp),cache=False)           
            
