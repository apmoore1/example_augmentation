import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Callable

from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from bella.data_types import TargetCollection
from gensim.models.word2vec import Word2Vec

from augmentation_helper import word_embedding_augmentation

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

def allen_spacy_tokeniser(text: str) -> Callable[[str], List[str]]:
    '''
    Returns the allennlp English spacy tokeniser as a callable function which 
    takes a String and returns a List of tokens/Strings.
    '''
    splitter = SpacyWordSplitter()
    return [token.text for token in splitter.split_words(text)]

def augmented_dataset(target_related_words_sim: Dict[str, List[Tuple[str, float]]],
                      dataset: TargetCollection, save_fp: Path, lower: bool
                      ) -> None:
    '''
    Given a dictionary of target words from the training dataset and the 
    values being all of the related words with their similarity score associated 
    to the target key, TDSA training dataset it will for each sample in the 
    training set check if the sample's target exists as a key in the given 
    dictionary and if so write the sample to the save file along with the 
    related targets and similarity scores under the following keys; 
    `alternative_targets` and `alternative_similarity`
    '''
    training_targets_in_embeddings = set(list(target_related_words_sim.keys()))
    with save_fp.open('w+') as save_file:
        count = 0
        for target_dict in dataset.data_dict():
            original_target = target_dict['target']
            if lower:
                original_target = original_target.lower()
            if original_target in training_targets_in_embeddings:
                alt_targets_similarity = target_related_words_sim[original_target]
                alt_targets_similarity = sorted(alt_targets_similarity, 
                                                key=lambda x: x[1], reverse=True)
                different_targets = [target for target, _ in alt_targets_similarity]
                alternative_similarity = [similarity for _, similarity in alt_targets_similarity]
                target_dict['alternative_targets'] = different_targets
                target_dict['alternative_similarity'] = alternative_similarity
                target_dict['epoch_number'] = list(target_dict['epoch_number'])
                json_target_dict = json.dumps(target_dict)
                if count != 0:
                    json_target_dict = f'\n{json_target_dict}'
                count += 1
                save_file.write(json_target_dict)

if __name__=='__main__':
    '''
    This will create a file which will be saved at the location stated within 
    the `augmented_dataset_fp` argument that is json data where each line is an 
    original target however it will have two extra values within the usual 
    target dictionary:
    1. `alternative_targets` -- A list of alternative targets that can be 
       used instead of the one given
    2. `alternative_similarity` -- A list of cosine similarity scores for the 
       alternative targets. The simialrity is of the original target and the 
       alternative targets.
    
    The first and second lists are indexed the same i.e. 2nd target 
    corresponds to the second similarity score. Also the lists have been 
    ordered by highest similarity score first.

    NOTE: Unlike the language model the similarity between the alternative 
    target the original is the same non matter the sentence it came from which 
    is not True for the language model therefore the similarity scores are not 
    sentence dependent compared to the language model perplexity score. 
    Therefore in theory we only need unique training targets and there 
    alternatives but we want the output to be as similar to the output of 
    `./augment_transformer.py`.
    NOTE: Unlike the transformer language model note every training target 
    will have alternatives targets as they might not exist in the embedding.
    '''
    tokeniser_choices = ['spacy']
    parser = argparse.ArgumentParser()
    augmented_dataset_help = "File Path to save the augmented dataset where "\
                             "each new line will contain a json dictionary "\
                             "that will have the standard Target data from "\
                             "the original dataset but will also include two "\
                             "additional fields: 1. `alternative_targets` 2. "\
                             "`alternative_similarity` "
    parser.add_argument("train_fp", help="File path to the training data", 
                        type=parse_path)
    parser.add_argument("embedding_fp", help="File path to the embedding",
                        type=parse_path)
    parser.add_argument("additional_targets_fp", type=parse_path,
                        help='File Path to additional targets')
    parser.add_argument("augmented_dataset_fp", type=parse_path, 
                        help=augmented_dataset_help)
    parser.add_argument("tokeniser", type=str, choices=tokeniser_choices)
    parser.add_argument("--lower", action="store_true")
    args = parser.parse_args()

    # Load tokeniser
    if args.tokeniser == 'spacy':
        tokeniser = allen_spacy_tokeniser
    else:
        raise ValueError(f'Tokeniser has to be one of the following {tokeniser_choices}')

    training_data = TargetCollection.load_from_json(args.train_fp)
    embedding = Word2Vec.load(str(args.embedding_fp))
    target_related_words_sim: Dict[str, List[Tuple[str, float]]]
    target_related_words_sim = word_embedding_augmentation(training_data, embedding,
                                                           lower=args.lower, 
                                                           k_nearest=-1,
                                                           tokeniser=tokeniser)
    augmented_dataset(target_related_words_sim, training_data, 
                      args.augmented_dataset_fp, lower=args.lower)