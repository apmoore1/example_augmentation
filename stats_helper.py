from typing import Callable, List

import numpy as np
import pandas as pd

def bootstrap(true: np.ndarray, predictions: np.ndarray,
              metric: Callable[[np.ndarray, np.ndarray], float],
              n: int = 1000, **metric_kwargs) -> np.ndarray:
    '''
    Given the true labels, predicted labels of *m* models, as well as the
    metric for evaluation will bootstrap *n* times over the predictions and
    true label evaluating each time based on the metrics. Returns all
    evaluations as an array of shape = [n, m].
    :param true: True labels, shape = [n_samples]
    :param predictions: Predictions, shape = [n_samples, n_models]
    :param metric: Function that evaluates the predictions e.g.
                   :py:func:`sklearn.metrics.accuracy_score`
    :param n: Number of times to bootstrap.
    :param **metric_kwargs: Keywords to provide to the metric function argument
    :return: Returns all *n* evaluations as a matrix, shape = [n, n_models].
    '''
    if len(predictions.shape) == 1:
        predictions = predictions.reshape(predictions.shape[0], 1)
    vector_size = true.shape[0]
    metric_scores = np.zeros((n, predictions.shape[1]))
    for index in range(n):
        random_index = np.random.choice(vector_size, vector_size,
                                        replace=True)
        true_random = true[random_index]
        predictions_random = predictions[random_index]
        for model_index in range(predictions_random.shape[1]):
            score = metric(true_random, predictions_random[:, model_index],
                           **metric_kwargs)
            metric_scores[index, model_index] = score
    return metric_scores


def bootstrap_one_t_test(bootstrap_samples: np.ndarray,
                         model_names: List[str]) -> pd.DataFrame:
    '''
    Creates a DataFrame of one tailed P-values for each model given a matrix
    of metric evaluations for each model. DataFrame shape =
    [n_models, n_models] where the models in the columns are tested if they are
    greater than the models in the rows.
    :param bootstrap_samples: Output of :py:func:`bootstrap`. A matrix of shape
                              = [n_evaluations, n_models] where an evaluation
                              is for example an accuracy score.
    :param model_names: A list of the model names in the same order as they
                        appear in the bootstrap_samples.
    :return: A DataFrame of one tailed test for each model where the index
             and columns are labelled by the model names. Shape = [n_models,
             n_models]
    '''
    num_bootstrap_evals = bootstrap_samples.shape[0]
    num_models = bootstrap_samples.shape[1]

    p_values = np.zeros((num_models, num_models))
    for model_index in range(num_models):
        model_bootstrap = bootstrap_samples[:, model_index]
        model_bootstrap = model_bootstrap.reshape(num_bootstrap_evals, 1)
        diff = model_bootstrap - bootstrap_samples
        diff = np.sort(diff, axis=0)
        is_better = diff > 0
        first_occurence = np.argmax(is_better, axis=0)
        # Needs to check that the differences are not all zeros. If they are
        # then the first occurence is equal to the num_bootstrap_evals to
        # make the p_value as high as possible.
        last_is_better = is_better[-1, :]
        actually_better_mask = (first_occurence != 0) + last_is_better
        not_better_mask = (actually_better_mask == 0)
        not_better_values = np.full(shape=num_models,
                                    fill_value=num_bootstrap_evals)
        not_better_values *= not_better_mask
        better_values = actually_better_mask * first_occurence
        first_occurence = better_values + not_better_values

        model_p_values = first_occurence / num_bootstrap_evals
        p_values[model_index] = model_p_values
    p_values = p_values.T
    p_values = pd.DataFrame(p_values, index=model_names, columns=model_names)
    return p_values