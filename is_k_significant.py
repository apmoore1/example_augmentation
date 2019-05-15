import argparse
import copy
from collections import Counter
import logging
import logging.handlers
import math
from pathlib import Path
import itertools
import statistics
import re

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
    verbose_help = 'Will print out the model name and the best and worse K '\
                   'values for each significant difference'

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
    parser.add_argument("--verbose", action="store_true", 
                        help=verbose_help)
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
    # if args.verbose then we don't get pairs but the first directory is the 
    # best K and all rest are worse as it requires performing sig test on all 
    # K values.
    best_worse_k_result_dirs = []
    # Contains strings of the models and their significant best K and worse K 
    # pairs
    best_worse_k_models = []
    best_worse_k_counts = {'best': Counter(), 'worse': Counter()}
    for tech, model_name in itertools.product(different_techniques, df_model_names):
        tech_model_values = df_data[(df_data['Model']==model_name) & 
                                    (df_data['Technique']==tech)]
        mean_k_values = tech_model_values.groupby('K').mean()
        worse_k = [mean_k_values.idxmin()[0]]
        best_k = mean_k_values.idxmax()[0]
        if args.second_best:
            sort_k_mean_values = mean_k_values.sort_values('score')
            worse_k = [sort_k_mean_values.index[-2]]
        if args.verbose:
            sort_k_mean_values = mean_k_values.sort_values('score')
            worse_k = sort_k_mean_values.index[:-1]
        
        best_worse_pair = []
        folder_tech = inv_tech_mapper[tech]
        folder_model_name = inv_model_name_mapper[model_name]
        best_worse_k_counts['best'].update([best_k])
        for k_value in [best_k, *worse_k]:
            folder_k_name = f'{folder_tech}{str(k_value)}'
            k_dir = Path(result_dir, folder_k_name, folder_model_name, dataset_name)
            best_worse_pair.append(k_dir)

        best_worse_k_result_dirs.append(tuple(best_worse_pair))

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
        column_names = ['best', 'worse']
        if args.verbose:
            if args.second_best:
                column_names = ['best', 'last', '2nd_last', 'worse']
            else:
                column_names = ['best', 'worse', '2nd_last', '2nd_best']
        t_results = bootstrap_one_t_test(metric_bootstrap_values, column_names)
        p_value = t_results['best']['worse']

        if p_value < args.p_value_threshold:
            p_value_threshold_pass += 1
        if args.verbose:
            all_ks = []
            for worse_dir in best_worse_dir[1:]:
                k = re.findall(r'\d+' ,worse_dir.parts[-3])[0]
                all_ks.append(k)
            t_results.index = ['best', *all_ks]
            significant_worse_ks = t_results['best'] < args.p_value_threshold
            significant_worse_ks = t_results[significant_worse_ks] == True
            significant_worse_ks = significant_worse_ks.index.to_list()
            significant_worse_ks = [int(_k) for _k in significant_worse_ks]
            best_worse_k_counts['worse'].update(significant_worse_ks)

    worse = 'worse'
    if args.second_best:
        worse='second best'
    print('Number of models and augmentation techniques whose Best K is '
          f'Statistically significantly (p < {args.p_value_threshold}) '
          f'better than the {worse} K is {p_value_threshold_pass} out of '
          f'{len(best_worse_k_result_dirs)} cases.')
    if args.verbose:
        print(pd.DataFrame(best_worse_k_counts))
    print('\n')
        


        