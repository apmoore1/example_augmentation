import copy
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Tuple
import random

from bella.data_types import TargetCollection, Target
from bella.contexts import context
from bella.tokenisers import stanford
from gensim.models.word2vec import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def exchange_targets(target_data: Dict[str, Any], 
                     target: str, new_target_id: str) -> Dict[str, Any]:
    '''
    Given a single Target data point it will replace the target 
    text within it as well as the target field with the given 
    target.
    
    It also adds a field `augmented` = True so that it is known
    that this data point is an augmented data point.
    
    :param target_data: The Target data dict that is to have its 
                        text, spans, and target field with the 
                        given target.
    :param target: The target string to replace the exisiting 
                   target.
    :param new_target_id: A unique target identifier, this is required so that 
                          the TargetCollection that is created for data 
                          augmentation has new identifiers for each target.
                          However the original target_id that created this 
                          target will be accessible through the 
                          'original_target_id' field/key.
    :returns: The Target data dict with the target and its related 
              data replaced with the given target.
    '''
    data_copy = copy.deepcopy(target_data)
    left_context = context(data_copy, 'left')[0]
    right_context = context(data_copy, 'right')[0]
    alternative_text = left_context + target + right_context
    # Finding the span of the new target within the new text
    start = len(left_context)
    end = len(target) + start
    alternative_span = [(start, end)]
    # Changing the fields
    original_target_id = data_copy['target_id']
    # Required when we are augmenting on top of augmentation
    if 'original_target_id' in data_copy:
        original_target_id = data_copy['original_target_id']
        
    data_copy['original_target_id'] = original_target_id
    data_copy['text'] = alternative_text
    data_copy['target'] = target
    data_copy['spans'] = alternative_span
    data_copy['augmented'] = True
    data_copy['target_id'] = new_target_id
    return data_copy






def multi_word_targets(dataset: TargetCollection, 
                       tokeniser: Callable[[str], List[str]],
                       lower: bool = True) -> List[str]:
    '''
    Given a dataset it will return all of the targets 
    tokenised and then re-combined with `_` to show 
    the multiple words that make up the target if it 
    contains multiple words.
    
    :param dataset: The dataset that contains all of the 
                    targets.
    :param tokeniser: The tokeniser to define if a target 
                      is made up of multiple words
    :param lower: if to lower case the target words.
    :returns: A list of targets where if they are made up of 
              multiple words the will now have an `_` between 
              them e.g. `tesco supermarket` would be 
              `tesco_supermarket`
    '''
    targets = dataset.target_set(lower=lower)
    tokenised_targets = [tokeniser(target) for target in targets]
    multi_word_targets = ['_'.join(target) for target in tokenised_targets]
    return multi_word_targets

def target_similarity_matrix(targets: List[str], word_embedding: Word2Vec
                             ) -> np.ndarray:
    '''
    Returns a matrix of shape [n_targets, n_targets] which represents 
    the similarity between targets, the similarity of itself is set to 
    0.
    
    :param targets: List of targets
    :param word_embedding: Used to get word vectors to compute similarity
                           between targets. It is assumed that all targets 
                           have a vector within the word embedding.
    :returns: A matrix of shape [n_targets, n_targets] which represents 
              the similarity between targets, the similarity of itself 
              is set to 0.
    '''
    # word vectors of the targets
    word_vectors = np.zeros((len(targets), word_embedding.vector_size))
    for target_index, target in enumerate(targets):
        word_vectors[target_index] = word_embedding.wv[target]
    sim_matrix = cosine_similarity(word_vectors)
    # Set all of the similarties of a word to itself as zero
    self_sim = np.eye(len(targets)) + 0
    self_sim = self_sim == 0
    sim_matrix = sim_matrix * self_sim
    return sim_matrix

def number_target_words() -> None:
    return None

def word_embedding_augmentation(dataset: TargetCollection, embedding: Word2Vec,
                                k_nearest: int, lower: bool = True,
                                tokeniser: Callable[[str], List[str]] = stanford
                                ) -> Dict[str, List[Tuple[str, float]]]:
    targets = multi_word_targets(dataset, tokeniser, lower=lower)
    filtered_targets = [target for target in targets 
                        if target in embedding.wv]
    sim_matrix = target_similarity_matrix(filtered_targets, embedding)
    # remove the `_` from the filtered_targets
    filtered_targets = [' '.join(target.split('_')) 
                        for target in filtered_targets]
    related_targets: Dict[str, List[Tuple[str, float]]] = {}
    for target_index, target in enumerate(filtered_targets):
        # Returns a list of target indexs where the first values
        # are the least similar and the last most similar
        if k_nearest == -1:
            nearest_indexs = np.argsort(sim_matrix[target_index])
        else:
            nearest_indexs = np.argsort(sim_matrix[target_index])[-k_nearest:]
        nearest_word_sim_value = []
        for index in nearest_indexs:
            nearest_word = filtered_targets[index]
            # If we want to threshold on similarity value that can be done here
            similarity_value = sim_matrix[target_index, index]
            nearest_word_sim_value.append((nearest_word, similarity_value))
        related_targets[target] = nearest_word_sim_value
    return related_targets



