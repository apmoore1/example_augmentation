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
                        choices=['accuracy', 'macro_f1'],
                        help='Metric score to use')
    parser.add_argument('save_fp', type=parse_path, 
                        help='File path to save figure')
    parser.add_argument('dataset_name', type=str, 
                        choices=['Restaurant','Laptop'])
    parser.add_argument("--val", action="store_true", 
                        help='Use Validation data and not test')
    args = parser.parse_args()
    metric_score = args.metric_score
    result_dir = args.result_dir
    num_runs = args.num_runs
    model_names = ['atae', 'bilinear', 'ian', 'tdlstm', 'tclstm']
    dataset_name = args.dataset_name
    values_of_k = [2,3,5,10]
    augmentation_technique = ['embedding', 'lm']
    thresholding = ['no_threshold_', 'threshold_']

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
            test_data.add_pred_sentiment(model_results)
            if metric_score == 'accuracy':
                all_results = test_data.dataset_metric_scores(accuracy_score)
            else:
                all_results = test_data.dataset_metric_scores(f1_score,
                                                              average='macro')
            result = statistics.mean(all_results)
            y_score.append(result)
            x_k.append(k)
            all_techniques.append(f'{tech} {threshold}')
            all_model_names.append(model_name)
            #for result in all_results:
            #    y_score.append(result)
            #    x_k.append(k)
            #    all_techniques.append(f'{tech} {threshold}')
            #    all_model_names.append(model_name)
    baseline_results_dir = Path('./results/baseline')
    for model_name in model_names:
        base_model_dir = Path(baseline_results_dir, model_name)
        base_data_dir = Path(base_model_dir, dataset_name)
        model_results = get_result(base_data_dir, num_runs, test=test)
        
        data_copy = copy.deepcopy(test_data)
        test_data.add_pred_sentiment(model_results)
        if metric_score == 'accuracy':
            all_results = test_data.dataset_metric_scores(accuracy_score)
        else:
            all_results = test_data.dataset_metric_scores(f1_score,
                                                          average='macro')
        result = statistics.mean(all_results)
        for k in values_of_k:
            y_score.append(result)
            x_k.append(k)
            all_techniques.append(f'baseline')
            all_model_names.append(model_name)

    plot_data = pd.DataFrame({'Model Name': all_model_names, f'{metric_score}': y_score,
                               'K': x_k, 'Augmentation Technique': all_techniques})
    g = sns.FacetGrid(plot_data, col="Model Name", col_wrap=2, height=5,
                      hue="Augmentation Technique")
    g = (g.map(plt.plot, "K", f'{metric_score}', marker=".")).add_legend()
    g.savefig(str(args.save_fp))
