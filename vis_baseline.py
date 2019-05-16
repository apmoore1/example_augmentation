import argparse
import copy
import logging
import logging.handlers
from pathlib import Path
import itertools
import statistics

from bella.data_types import TargetCollection
from sklearn.metrics import accuracy_score, f1_score
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


from model_helper import get_result

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__=='__main__':
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
    parser.add_argument('save_fp', type=parse_path, 
                        help='File path to save figure')
    parser.add_argument('dataset_name', type=str, 
                        choices=['Restaurant','Laptop'])
    parser.add_argument("--val_fp", type=parse_path, 
                        help='File path to the validation data')
    args = parser.parse_args()
    metric_score = args.metric_score
    result_dir = args.result_dir
    num_runs = args.num_runs
    model_names = ['atae', 'bilinear', 'ian', 'tdlstm', 'tclstm', 'ds_elmo_t', 'ds_elmo_t_embedding', 'ds_elmo_t_embedding_tune', 'ds_elmo_t_fine_tune', 'elmo_t', 'lstm', 'lstm_random']
    model_names_plot_names = ['ATAE', 'BiLinear', 'IAN', 'TDLSTM', 'TCLSTM', 'DS ELMo', 'DS ELMo\nEmbed', 'DS ELMo\nEmbed Tune', 'DS ELMo\nTune', 'ELMo', 'LSTM', 'LSTM Rand']
    model_name_mapper = {m_name: p_name for m_name, p_name in zip(model_names, model_names_plot_names)}
    dataset_name = args.dataset_name

    test_data_fp = args.test_data_fp
    test_data = TargetCollection.load_from_json(test_data_fp)
    dataset_name_flag = [('Test', test_data, True)]
    if args.val_fp:
        val_data = TargetCollection.load_from_json(args.val_fp)
        dataset_name_flag.append(('Validation', val_data, False))

    y_score = []
    x_model_names = []
    dataset_split_names = []

    for dataset_split_name, dataset, test_flag in dataset_name_flag:
        for model_name in model_names:
            model_result_dir = Path(result_dir, 
                                    model_name, dataset_name)
            model_results = get_result(model_result_dir, num_runs, test=test_flag)
            data_copy = copy.deepcopy(dataset)
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
                x_model_names.append(model_name_mapper[model_name])
                dataset_split_names.append(dataset_split_name)
    sns.set(rc={'figure.figsize':(15.,10.)})
    if args.val_fp:
        fig, ax = plt.subplots(1,1,figsize=(20,20))
        plot_data = pd.DataFrame({'Model': x_model_names, 
                                  f'{metric_score}': y_score,
                                  'Dataset': dataset_split_names})
        g = sns.catplot(x='Model', y=f'{metric_score}',
                        col='Dataset', data=plot_data,
                        kind="point", ci='sd', dodge=0.6,  col_wrap=1,ax=ax)
        ax.legend()
        ax.figure.savefig(str(args.save_fp))

    else:
        plot_data = pd.DataFrame({'Model': x_model_names, 
                                  f'{metric_score}': y_score})
        g = sns.pointplot(data=plot_data, x='Model', y=f'{metric_score}',
                          ci='sd', dodge=0.6, col_wrap=1)
        g.figure.savefig(str(args.save_fp))
