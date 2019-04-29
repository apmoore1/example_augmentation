import argparse
import json
from pathlib import Path
import statistics

import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__=='__main__':
    '''
    This will create a distribution plot for all of the similarity scores 
    that are created from the expanded dataset given. This plot will be 
    saved to the location given in the arguments. Given the top **X** percent 
    argument it will also report the top **X** similarity value and plot this 
    on the similarity distribution a long with the mean value +/- Standard 
    Deviation.
    '''

    top_x_percent_help = 'Value to print what the top X percent similarity '\
                         'value is and it will be shown in the distribution plot'
    parser = argparse.ArgumentParser()
    parser.add_argument("expanded_dataset_fp", type=parse_path, 
                        help='File Path to the embedding expanded dataset')
    parser.add_argument("distribution_plot_fp", type=parse_path, 
                        help='File path to save the similarity distribution plot')
    parser.add_argument("top_x_percent", type=float, default=10.0, 
                        help=top_x_percent_help)
    args = parser.parse_args()

    similarity_values = []
    with args.expanded_dataset_fp.open('r') as expanded_dataset_file:
        for line in expanded_dataset_file:
            target_sample = json.loads(line)
            similarity_values.extend(target_sample['alternative_similarity'])
    top_x_percent = args.top_x_percent
    number_similarity_values = len(similarity_values)
    top_x_value = int((number_similarity_values / 100) * top_x_percent)
    top_x_similarity_value = sorted(similarity_values)[-top_x_value]
    print(f'Top {top_x_percent}% similarity value is {top_x_similarity_value}')
    mean_similarity = statistics.mean(similarity_values)
    sd_similarity = statistics.stdev(similarity_values)
    plus_one_sd = mean_similarity + sd_similarity
    minus_one_sd = mean_similarity - sd_similarity
    

    fig, ax = plt.subplots(figsize=(10,10))
    ax = sns.distplot(similarity_values, hist=True, kde=True, ax=ax)

    colors = ['red', 'green', 'pink', 'black']
    x_values = [minus_one_sd, mean_similarity, plus_one_sd, 
                top_x_similarity_value]
    labels = ['-SD', 'Mean', '+SD', f'Top {top_x_percent}%']
    for color, x, label in zip(colors, x_values, labels):
        ax.scatter(y=[0.5], x=[x], label=label, c=color)
    ax.legend()
    fig.savefig(str(args.distribution_plot_fp))

    k2, p = stats.normaltest(similarity_values)
    if p < 0.05:
        print('The similarity values do not come from a normal distribution')
    print(f'P value for the normal test {p}')
    
