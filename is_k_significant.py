import argparse
import copy
import logging
import logging.handlers
import math
from pathlib import Path
import itertools
import statistics

from bella.data_types import TargetCollection
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from model_helper import get_result
from stats_helper import bootstrap, bootstrap_one_t_test

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__=='__main__':
    p_value_help = 'The threshold the P Value has to be less than '\
                   'to to be significant'
    second_best_help = 'Instead of comparing best to worse, it compares '\
                       'best to second best.'

    parser = argparse.ArgumentParser()
    parser.add_argument("num_runs", type=int, 
                        help='Number of times the model was ran for')
    parser.add_argument("result_dir", type=parse_path, 
                        help='Directory where the results were stored',
                        default='./results/baseline')
    parser.add_argument("test_data_fp", type=parse_path, 
                        help='File Path to the test data')
    parser.add_argument('metric_score', type=str, 
                        choices=['Accuracy', 'Macro F1'],
                        help='Metric score to use')
    parser.add_argument('dataset_name', type=str, 
                        choices=['Restaurant','Laptop'])
    parser.add_argument('p_value_threshold', type=float, 
                        help=p_value_help)
    parser.add_argument("--val", action="store_true", 
                        help='Use Validation data and not test')
    parser.add_argument("--bootstrap_samples", type=int, default=10000,
                        help='Number of times to bootstrap')
    parser.add_argument("--second_best", action="store_true", 
                        help=second_best_help)
    args = parser.parse_args()
    metric_score = args.metric_score
    result_dir = args.result_dir
    num_runs = args.num_runs
    model_names = ['atae', 'bilinear', 'ian', 'tdlstm', 'tclstm', 'atae_ds_lm_embedding']
    model_name_mapper = {'atae': 'ATAE', 'bilinear': 'BiLinear', 'ian': 'IAN',
                         'tdlstm': 'TDLSTM', 'tclstm': 'TCLSTM', 'atae_ds_lm_embedding': 'ATAE ELMo T'}
    dataset_name = args.dataset_name
    values_of_k = [2,3,5,10]
    augmentation_technique = ['embedding', 'lm']
    thresholding = ['no_threshold_', 'threshold_']
    technique_threshold_mapper = {'no_threshold_embedding': 'Embedding',
                                  'threshold_embedding': 'Embedding T',
                                  'no_threshold_lm': 'LM', 
                                  'threshold_lm': 'LM T'}

    test_data_fp = args.test_data_fp
    test_data = TargetCollection.load_from_json(test_data_fp)

    test = True
    if args.val:
        test = False

    folder_names = list(itertools.product(thresholding, augmentation_technique, 
                                          values_of_k))

    y_score = []
    x_k = []
    all_techniques = []
    all_model_names = []

    for threshold, tech, k in folder_names:
        folder_name = f'{threshold}{tech}{str(k)}'
        for model_name in model_names:
            model_result_dir = Path(result_dir, folder_name, model_name, dataset_name)
            model_results = get_result(model_result_dir, num_runs, test=test)
            data_copy = copy.deepcopy(test_data)
            data_copy.add_pred_sentiment(model_results)
            if metric_score == 'Accuracy':
                all_results = data_copy.dataset_metric_scores(accuracy_score)
            else:
                all_results = data_copy.dataset_metric_scores(f1_score,
                                                              average='macro')
            results_len_error = f'Number of runs {num_runs} does not match '\
                                f'number of results {len(all_results)} for '\
                                f'the following directory {model_result_dir}'
            assert len(all_results) == num_runs, results_len_error
            for result in all_results:
                y_score.append(result)
                x_k.append(k)
                all_techniques.append(technique_threshold_mapper[f'{threshold}{tech}'])
                all_model_names.append(model_name_mapper[model_name])

    df_data = pd.DataFrame({'Model': all_model_names, 'K': x_k, 'score': y_score, 'Technique': all_techniques})
    different_techniques = list(technique_threshold_mapper.values())
    df_model_names = list(model_name_mapper.values())
    inv_model_name_mapper = {value: key for key, value in model_name_mapper.items()}
    inv_tech_mapper = {value: key for key, value in technique_threshold_mapper.items()}
    # Need to get a path to the model results for the best and worse K as a pair
    best_worse_k_result_dirs = []
    for tech, model_name in itertools.product(different_techniques, df_model_names):
        tech_model_values = df_data[(df_data['Model']==model_name) & 
                                    (df_data['Technique']==tech)]
        mean_k_values = tech_model_values.groupby('K').mean()
        worse_k = mean_k_values.idxmin()[0]
        best_k = mean_k_values.idxmax()[0]
        if args.second_best:
            sort_k_mean_values = mean_k_values.sort_values('score')
            worse_k = sort_k_mean_values.index[-2]
        
        folder_tech = inv_tech_mapper[tech]
        folder_model_name = inv_model_name_mapper[model_name]
        folder_name_worse = f'{folder_tech}{str(worse_k)}'
        worse_dir = Path(result_dir, folder_name_worse, folder_model_name, dataset_name)
        folder_name_best = f'{folder_tech}{str(best_k)}'
        best_dir = Path(result_dir, folder_name_best, folder_model_name, dataset_name)

        best_worse_k_result_dirs.append((best_dir, worse_dir))

    num_times_to_boot = args.bootstrap_samples
    p_value_threshold_pass = 0
    for best_worse_dir in best_worse_k_result_dirs:
        raw_best_worse_results = []
        true_labels = None
        # Gets the raw median result values for the best worse K case.
        for k_dir in best_worse_dir:
            results = get_result(k_dir, num_runs, test=test)
            data_copy = copy.deepcopy(test_data)
            data_copy.add_pred_sentiment(results)
            if metric_score == 'Accuracy':
                all_results = data_copy.dataset_metric_scores(accuracy_score)
            else:
                all_results = data_copy.dataset_metric_scores(f1_score,
                                                                average='macro')
            true_labels = data_copy.sentiment_data()
            pred_matrix = data_copy.sentiment_data(sentiment_field='predicted')
            median_index_value = math.floor(len(all_results) / 2)
            median_result_index = np.argsort(all_results)[median_index_value]
            median_raw_results = np.array(pred_matrix)[:,median_index_value]
            raw_best_worse_results.append(median_raw_results)
        raw_best_worse_results = np.array(raw_best_worse_results).T
        metric_bootstrap_values = None
        true_labels = np.array(true_labels)
        if metric_score == 'Accuracy':
            metric_bootstrap_values = bootstrap(true_labels, raw_best_worse_results, 
                                                accuracy_score, num_times_to_boot)
        else:
            metric_bootstrap_values = bootstrap(true_labels, raw_best_worse_results, 
                                                f1_score, num_times_to_boot, 
                                                average='macro')
        t_results = bootstrap_one_t_test(metric_bootstrap_values, ['best', 'worse'])
        p_value = t_results['best']['worse']
        if p_value <= args.p_value_threshold:
            p_value_threshold_pass += 1

    worse = 'worse'
    if args.second_best:
        worse='second best'
    print('Number of models and augmentation techniques whose Best K is '
          f'Statistically significantly (p < {args.p_value_threshold}) '
          f'better than the {worse} K is {p_value_threshold_pass} out of '
          f'{len(best_worse_k_result_dirs)} cases.')
        


        