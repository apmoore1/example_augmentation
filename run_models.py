import argparse
import logging
import logging.handlers
from pathlib import Path
from itertools import product

from bella_allen_nlp import AllenNLPModel
from bella.data_types import TargetCollection

from model_helper import run_n_times

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__=='__main__':
    dataset_name_choices = ['Restaurant', 'Laptop', 'Election']

    parser = argparse.ArgumentParser()
    num_runs_help = "Number of times to run each model. This is to take the " \
                    "random seeds affect into account"
    data_splits_dir_help = "File path to the directory that stores the " \
                           "data splits"
    result_dir_help = "File path to the directory to store the models raw results"
    model_config_help = "File path to the model config directory for the " \
                        "baseline models"
    log_fp_help = "Log file path to save the log to for this run"
    parser.add_argument("num_runs", help=num_runs_help, type=int)
    parser.add_argument("data_splits_dir", help=data_splits_dir_help,
                        type=parse_path, default='./data/splits')
    parser.add_argument("result_dir", help=result_dir_help,
                        type=parse_path, default='./results/baseline')
    parser.add_argument("model_config_dir", help=model_config_help, 
                        type=parse_path, default='./model_configs/baseline')
    parser.add_argument("dataset_name", type=str, help='Name of the dataset', 
                        choices=dataset_name_choices)
    parser.add_argument("log_fp", help=log_fp_help, 
                        type=parse_path)
    parser.add_argument("--augmented_data_fp", help="Augmented data directory",
                        type=parse_path)
    parser.add_argument("--model_names", nargs='+', type=str)
    parser.add_argument("--model_name_save_names", nargs='+', type=str)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    augmented_fp: Path = args.augmented_data_fp

    log_fp = args.log_fp
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_fp)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Data paths
    data_dir = args.data_splits_dir
    result_dir: Path = args.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    model_config_dir = args.model_config_dir

    # Models
    model_names = ['atae', 'bilinear', 'ian', 'tdlstm', 'tclstm']
    if args.model_names:
        model_names = args.model_names
    if args.model_name_save_names:
        model_name_save_names = args.model_name_save_names
        model_name_mapper = {model_name: save_name for model_name, save_name in 
                             zip(model_names, model_name_save_names)}
    else:
        model_name_mapper = {model_name: model_name for model_name in model_names}
    bilstm_names = ['']
    all_model_names = product(model_names, bilstm_names)
    all_models = []
    for model_name, bilstm_name in all_model_names:
        model_name = f'{model_name}{bilstm_name}'.strip()
        model_config_fp = Path(model_config_dir, f'{model_name}.json')
        all_models.append(AllenNLPModel(model_name_mapper[model_name], 
                                        model_config_fp))
    # Data
    path_dataset = lambda _dir, name: TargetCollection.load_from_json(Path(_dir, name))
    train = path_dataset(data_dir, f'{dataset_name} Train')
    logger.info(f'Size of the original dataset {len(train)}')
    if augmented_fp is not None:
        augmented_data = TargetCollection.load_from_json(augmented_fp)
        logger.info(f'Size of the augmented dataset {len(augmented_data)}')
        train = TargetCollection.combine_collections(train, augmented_data)
        logger.info(f'Size of the training augmented dataset {len(train)}')
    val = path_dataset(data_dir, f'{dataset_name} Val')
    logger.info(f'Size of the validation set {len(val)}')
    test = path_dataset(data_dir, f'{dataset_name} Test')
    logger.info(f'Size of the test set {len(test)}')

    model_data = product(all_models, [(train, val, test)])
    for model, data in model_data:
        logger.info(f'Running Model {model.name} on {dataset_name} dataset '
                    f'{args.num_runs} times')
        model_rep_dir = Path(result_dir, model.name)
        data_rep_dir = Path(model_rep_dir, f'{dataset_name}')
        if not data_rep_dir.exists():
            data_rep_dir.mkdir(parents=True)
        train, val, test = data
        try:
            run_n_times(model, train, val, test, data_rep_dir, args.num_runs)
        except:
            logger.info(f'Running Model {model.name} on {dataset_name} dataset '
                        f'{args.num_runs} times')
            raise ValueError(print(f'Running Model {model.name} on {dataset_name} dataset '
              f'{args.num_runs} times'))