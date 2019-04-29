import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Any, Callable, Set, Union

from allennlp.data.dataset_readers import SimpleLanguageModelingDatasetReader, DatasetReader
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import load_archive, Model, BidirectionalLanguageModel
from allennlp.modules.sampled_softmax_loss import SampledSoftmaxLoss
from allennlp.nn.util import get_text_field_mask
from bella.data_types import TargetCollection
import torch

from augmentation_helper import word_embedding_augmentation

# This is form from allennlp.models import BidirectionalLanguageModel
def _forward(self,  # type: ignore
            source: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
    """
    Computes the averaged forward (and backward, if language model is bidirectional)
    LM loss from the batch.
    Parameters
    ----------
    source: ``Dict[str, torch.LongTensor]``, required.
        The output of ``Batch.as_tensor_dict()`` for a batch of sentences. By convention,
        it's required to have at least a ``"tokens"`` entry that's the output of a
        ``SingleIdTokenIndexer``, which is used to compute the language model targets.
    Returns
    -------
    Dict with keys:
    ``'loss'``: ``torch.Tensor``
        forward negative log likelihood, or the average of forward/backward
        if language model is bidirectional
    ``'forward_loss'``: ``torch.Tensor``
        forward direction negative log likelihood
    ``'backward_loss'``: ``torch.Tensor`` or ``None``
        backward direction negative log likelihood. If language model is not
        bidirectional, this is ``None``.
    ``'lm_embeddings'``: ``Union[torch.Tensor, List[torch.Tensor]]``
        (batch_size, timesteps, embed_dim) tensor of top layer contextual representations or
        list of all layers. No dropout applied.
    ``'noncontextual_token_embeddings'``: ``torch.Tensor``
        (batch_size, timesteps, token_embed_dim) tensor of bottom layer noncontextual
        representations
    ``'mask'``: ``torch.Tensor``
        (batch_size, timesteps) mask for the embeddings
    """
    # pylint: disable=arguments-differ
    mask = get_text_field_mask(source)

    # shape (batch_size, timesteps, embedding_size)
    embeddings = self._text_field_embedder(source)

    # Either the top layer or all layers.
    contextual_embeddings: Union[torch.Tensor, List[torch.Tensor]] = self._contextualizer(
            embeddings, mask
    )

    return_dict = {}

    # If we have target tokens, calculate the loss.
    token_ids = source.get("tokens")
    if token_ids is not None:
        assert isinstance(contextual_embeddings, torch.Tensor)

        # Use token_ids to compute targets
        forward_targets = torch.zeros_like(token_ids)
        forward_targets[:, 0:-1] = token_ids[:, 1:]

        if self._bidirectional:
            backward_targets = torch.zeros_like(token_ids)
            # This is only the same if all of the sentences are the same length
            # else the shorter sentences will have a </S> Special tokens
            backward_targets[:, 1:] = token_ids[:, 0:-1]
        else:
            backward_targets = None

        # add dropout
        contextual_embeddings_with_dropout = self._dropout(contextual_embeddings)

        # compute softmax loss
        forward_loss, backward_loss = self._compute_loss(contextual_embeddings_with_dropout,
                                                            embeddings,
                                                            forward_targets,
                                                            backward_targets)
        
        num_targets = torch.sum((forward_targets > 0).long())

        if num_targets > 0:
            if self._bidirectional:
                if getattr(self, 'batch_loss', None) is None:
                    average_loss = 0.5 * (forward_loss + backward_loss) / num_targets.float()
                else:
                    average_loss = 0.5 * (forward_loss + backward_loss)
            else:
                if getattr(self, 'batch_loss', None) is None:
                    average_loss = forward_loss / num_targets.float()
                else:
                    average_loss = forward_loss
        else:
            average_loss = torch.tensor(0.0).to(forward_targets.device)  # pylint: disable=not-callable

        if getattr(self, 'batch_loss', None) is None:
            self._last_average_loss[0] = average_loss.detach().item()

        if num_targets > 0:
            return_dict.update({
                    'loss': average_loss,
                    'forward_loss': forward_loss / num_targets.float(),
                    'backward_loss': (backward_loss / num_targets.float()
                                        if backward_loss is not None else None),
                    'batch_weight': num_targets.float()
            })
        else:
            # average_loss zero tensor, return it for all
            return_dict.update({
                    'loss': average_loss,
                    'forward_loss': average_loss,
                    'backward_loss': average_loss if backward_loss is not None else None
            })

    return_dict.update({
            # Note: These embeddings do not have dropout applied.
            'lm_embeddings': contextual_embeddings,
            'noncontextual_token_embeddings': embeddings,
            'mask': mask
    })

    return return_dict

# This is from from allennlp.modules.sampled_softmax_loss import SampledSoftmaxLoss
def _forward_eval(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # pylint: disable=invalid-name
    # evaluation mode, use full softmax
    if self.sparse:
        w = self.softmax_w.weight
        b = self.softmax_b.weight.squeeze(1)
    else:
        w = self.softmax_w
        b = self.softmax_b

    log_softmax = torch.nn.functional.log_softmax(torch.matmul(embeddings, w.t()) + b, dim=-1)
    if self.tie_embeddings and not self.use_character_inputs:
        targets_ = targets + 1
    else:
        targets_ = targets
    batch_loss = getattr(self, 'batch_loss', None)
    if batch_loss:
        return torch.nn.functional.nll_loss(log_softmax, targets_.long(), reduce=False)
    else:
        return torch.nn.functional.nll_loss(log_softmax, targets_.long(), reduction="sum")


def _loss_helper(self,  
                    direction: int,
                    direction_embeddings: torch.Tensor,
                    direction_targets: torch.Tensor,
                    token_embeddings: torch.Tensor) -> Tuple[int, int]:
    mask = direction_targets > 0
    # we need to subtract 1 to undo the padding id since the softmax
    # does not include a padding dimension

    # shape (batch_size * timesteps, )
    non_masked_targets = direction_targets.masked_select(mask) - 1

    # shape (batch_size * timesteps, embedding_dim)
    non_masked_embeddings = direction_embeddings.masked_select(
            mask.unsqueeze(-1)
    ).view(-1, self._forward_dim)
    # note: need to return average loss across forward and backward
    # directions, but total sum loss across all batches.
    # Assuming batches include full sentences, forward and backward
    # directions have the same number of samples, so sum up loss
    # here then divide by 2 just below
    if not self._softmax_loss.tie_embeddings or not self._use_character_inputs:
        loss = self._softmax_loss(non_masked_embeddings, non_masked_targets)
        batch_loss = getattr(self, 'batch_loss', None)
        if batch_loss:
            loss_reshaped = torch.zeros_like(direction_targets, 
                                             dtype=torch.float)
            loss_reshaped[mask] = loss
            loss = loss_reshaped.sum(1) / mask.sum(1, dtype=torch.float)
        return loss
    else:
        # we also need the token embeddings corresponding to the
        # the targets
        raise NotImplementedError("This requires SampledSoftmaxLoss, which isn't implemented yet.")
        # pylint: disable=unreachable
        non_masked_token_embeddings = self._get_target_token_embeddings(token_embeddings, mask, direction)
        return self._softmax(non_masked_embeddings,
                                non_masked_targets,
                                non_masked_token_embeddings)


def _compute_loss(self,
                      lm_embeddings: torch.Tensor,
                      token_embeddings: torch.Tensor,
                      forward_targets: torch.Tensor,
                      backward_targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:

    # If bidirectional, lm_embeddings is shape (batch_size, timesteps, dim * 2)
    # If unidirectional, lm_embeddings is shape (batch_size, timesteps, dim)
    # forward_targets, backward_targets (None in the unidirectional case) are
    # self (batch_size, timesteps) masked with 0
    if self._bidirectional:
        forward_embeddings, backward_embeddings = lm_embeddings.chunk(2, -1)
        backward_loss = self._loss_helper(1, backward_embeddings, backward_targets, token_embeddings)
    else:
        forward_embeddings = lm_embeddings
        backward_loss = None

    forward_loss = self._loss_helper(0, forward_embeddings, forward_targets, token_embeddings)
    return forward_loss, backward_loss




def allen_spacy_tokeniser(text: str) -> Callable[[str], List[str]]:
    '''
    Returns the allennlp English spacy tokeniser as a callable function which 
    takes a String and returns a List of tokens/Strings.
    '''
    splitter = SpacyWordSplitter()
    return [token.text for token in splitter.split_words(text)]

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

def targets_not_in_vocab(targets: Set[str], vocab: Set[str], 
                         tokeniser: Callable[[str], List[str]],
                         partial: bool = False) -> List[str]:
    '''
    Given a Set of unique targets from the training dataset and the vocabulary 
    of the Transformer model as another Set it will return all target words 
    that are in the training set but not in the vocabulary.

    Tokeniser is required to split the training target words as, target words 
    can be multi words expressions. It is assumed that the vocabulary from the 
    transformer is already tokenised.
    
    Partial here means it will return all target words that have at least one 
    of the words within the given vocabulary.
    '''

    not_in_vocab = []
    for target in targets:
        split_targets = tokeniser(target)
        if partial:
            number_split_targets = len(split_targets)
            not_in_count = 0
            for target_token in split_targets:
                if target_token not in vocab:
                    not_in_count += 1
            if not_in_count == number_split_targets:
                not_in_vocab.append(target)
        else:
            for target_token in split_targets:
                if target_token not in vocab:
                    not_in_vocab.append(target)
                    break
    return not_in_vocab

def sentence_perplexitys(model: Model, 
                         dataset_reader: SimpleLanguageModelingDatasetReader, 
                         sentences: List[str], tokeniser: Callable[[str], List[str]],
                         pre_tokenise: bool = True) -> List[float]:
    '''
    Given a language model, a dataset reader to convert a sentence into the 
    sutible form for the language model, a list of sentences, a tokeniser 
    that is only required if the pre_tokenise is True. Will return a list of 
    perplexity scores corresponding to each sentence in the list of sentences 
    given.

    If pre_tokenise is True then each sentence is tokenised with the given 
    tokeniser and then joined back together on whitespace. This is only to 
    ensure that the model is given sentences that have been tokenised in a 
    similar way to how the sentences given to the model in training were.
    '''
    sentence_instances = []
    for sentence in sentences:
        if pre_tokenise:
            sentence_tokens = tokeniser(sentence)
            sentence = ' '.join(sentence_tokens)
        sentence_instances.append(dataset_reader.text_to_instance(sentence))
    
    
    results = model.forward_on_instances(sentence_instances)
    result_perplexitys = [math.exp(result['loss']) for result in results]
    return result_perplexitys


def swap_targets(target_dict: Dict[str, Any], targets_to_swap: List[str]
                 ) -> Iterable[Tuple[str, str]]:
    '''
    Given a target dictionary that contains a `text`, `target`, and `spans`
    fields it will then return a Tuple containing the target that has been 
    swaped into the text and the text with that swaped target in the correct 
    start and end character index defined by the `spans` in the target 
    dictionary.
    '''
    target_text = target_dict['text']
    target_start, target_end = target_dict['spans'][0]
    target = target_dict['target']

    start_target_text = target_text[:target_start].strip()
    end_target_text = target_text[target_end:].strip()
    for new_target in targets_to_swap:
        if target == new_target:
            continue
        new_target_sentence = f'{start_target_text} {new_target} {end_target_text}'
        yield (new_target, new_target_sentence)

def augmented_dataset(transformer_model: Model, 
                      dataset_reader: SimpleLanguageModelingDatasetReader, 
                      alternative_targets: List[str],
                      tokeniser: Callable[[str], List[str]],
                      dataset: TargetCollection, save_fp: Path,
                      batch_size: int = 15) -> None:
    '''
    Given a language model, the reader to convert sentences into a format for 
    the language model, a list of alternative targets, and a TDSA dataset. It 
    will process each sample within the TDSA dataset by swaping the target 
    within that sentence with each target in the alternative set and record 
    the new sentence perplexity with this swaped target. After each alternative 
    target has been processed like this and the original target sentence 
    perplexity has been recorded it will save this sample to the given `save_fp`
    as a json dict with all of the the original training sample json data 
    with the addition of `alternative_targets`, `alternative_perplexity`, and 
    `original_perplexity` fields which has been defined below in the main 
    documentation.
    '''
    with save_fp.open('w+') as save_file:
        for index, target_dict in enumerate(training_data.data_dict()):
            from time import time
            t = time()
            orginal_text = target_dict['text']
            original_perplexity = sentence_perplexitys(transformer_model, dataset_reader, 
                                                    [orginal_text], tokeniser=tokeniser)[0]
            target_perplexity = []
            new_target_sentences = swap_targets(target_dict, alternative_targets)

            targets = []
            sentences = []
            count = 0
            for target_sentence in new_target_sentences:
                target, sentence = target_sentence
                targets.append(target)
                sentences.append(sentence)
                if count == batch_size:
                    perplexitys = sentence_perplexitys(transformer_model, another_reader, 
                                                       sentences, tokeniser=tokeniser)
                    
                    target_perplexity.extend(list(zip(targets, perplexitys)))
                    sentences = []
                    targets = []
                    count = 0
                count += 1
            if targets:
                perplexitys = sentence_perplexitys(transformer_model, another_reader, 
                                                    sentences, tokeniser=tokeniser)
                target_perplexity.extend(list(zip(targets, perplexitys)))
                sentences = []
                targets = []
                count = 0
            target_perplexity = sorted(target_perplexity, key=lambda x: x[1], 
                                       reverse=False)

            different_targets = [target for target, _ in target_perplexity]
            alternative_perplexity = [perplexity for _, perplexity in target_perplexity]
            target_dict['alternative_targets'] = different_targets
            target_dict['alternative_perplexity'] = alternative_perplexity
            target_dict['original_perplexity'] = original_perplexity
            target_dict['epoch_number'] = list(target_dict['epoch_number'])
            print(time() - t)
            json_target_dict = json.dumps(target_dict)
            if index != 0:
                json_target_dict = f'\n{json_target_dict}'
            save_file.write(json_target_dict)

if __name__=='__main__':
    '''
    This will create a file which will be saved at the location stated within 
    the `augmented_dataset_fp` argument that is json data where each line is an 
    original target however it will have three extra values within the usual 
    target dictionary:
    1. `alternative_targets` -- A list of alternative targets that can be 
       used instead of the one given
    2. `alternative_perplexity` -- A list of perplexity scores for the 
       alternative targets. The perplexity score is the score of the original 
       sentence but with the given alternative target instead of the original
    3. `original_perplexity` -- The perplexity score of the current sentence 
       with the original target within it.
    
    The first and second lists are indexed the same i.e. 2nd target 
    corresponds to the second perplexity score. Also the lists have been 
    ordered by lowest perplexity score first.

    We do not need to worry about target words in the training set that 
    are not in the Language Model vocab as they can guess by the sentence 
    context if new targets makes more sense.
    '''
    augmented_dataset_help = "File Path to save the augmented dataset where "\
                             "each new line will contain a json dictionary "\
                             "that will have the standard Target data from "\
                             "the original dataset but will also include three "\
                             "additional fields: 1. `alternative_targets` 2. "\
                             "`alternative_perplexity` and 3. "\
                             "`original_perplexity`"
    batch_size_help = "Number of sentences for the language model to process "\
                      "in one batch, larger the quicker it is but more memory "\
                      "it uses, if too large can cuase program to crash "\
                      "with out of memory"
    tokeniser_choices = ['spacy']
    parser = argparse.ArgumentParser()
    parser.add_argument("train_fp", help="File path to the training data", 
                        type=parse_path)
    parser.add_argument("transformer_fp", type=parse_path,
                        help="File path to the transformer model")
    parser.add_argument("additional_targets_fp", type=parse_path,
                        help='File Path to additional targets')
    parser.add_argument("augmented_dataset_fp", type=parse_path, 
                        help=augmented_dataset_help)
    parser.add_argument("tokeniser", type=str, choices=tokeniser_choices)
    parser.add_argument("--cuda", action="store_true", 
                        help='Whether or not to use the GPU')
    parser.add_argument("--batch_size", type=int, default=15, 
                        help=batch_size_help)

    args = parser.parse_args()

    # Load tokeniser
    if args.tokeniser == 'spacy':
        tokeniser = allen_spacy_tokeniser
    else:
        raise ValueError(f'Tokeniser has to be one of the following {tokeniser_choices}')

    # Load the model
    BidirectionalLanguageModel._loss_helper = _loss_helper
    BidirectionalLanguageModel._compute_loss = _compute_loss
    BidirectionalLanguageModel.forward = _forward
    SampledSoftmaxLoss._forward_eval = _forward_eval
    archive = load_archive(args.transformer_fp)
    transformer_model = archive.model
    if args.cuda:
        transformer_model.cuda()
    else:
        transformer_model.cpu()
    transformer_model.eval()
    transformer_model.batch_loss = True
    transformer_model._softmax_loss.batch_loss = True

    # Load the dataset reader that came with the transformer model and ensure 
    # that the max sequence length is set to infinte so that we can analysis 
    # any length sentence (problem can occur with Memory (GPU espically)) 
    # if a sentence is really long.
    config = archive.config
    dict_config = config.as_dict(quiet=True)
    dataset_reader_config = config.get("dataset_reader")
    if dataset_reader_config.get("type") == "multiprocess":
        dataset_reader_config = dataset_reader_config.get("base_reader")
        if 'max_sequence_length' in dataset_reader_config:
            dataset_reader_config['max_sequence_length'] = None
    another_reader = DatasetReader.from_params(dataset_reader_config)
    another_reader.lazy = False
    # Load the TDSA training data
    training_data = TargetCollection.load_from_json(args.train_fp)

    #transformer_vocab = transformer_model.vocab.get_token_to_index_vocabulary()
    #transformer_vocab = set(list(transformer_vocab.keys()))
    # #different_targets = set(list(training_data.target_set()))
    
    # Load the additional target set that will be used to swap original targets 
    # for these and see the perplexity score of the sentence afterwards
    with args.additional_targets_fp.open('r') as additional_targets_file:
        additional_targets = set(json.load(additional_targets_file))
    print(f'Number of additional targets {len(additional_targets)}')
    augmented_dataset(transformer_model, another_reader, additional_targets, 
                      tokeniser, training_data, args.augmented_dataset_fp,
                      batch_size=args.batch_size)