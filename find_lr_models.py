import argparse
import logging
import logging.handlers
from pathlib import Path
from itertools import product
import tempfile

from bella_allen_nlp import AllenNLPModel
from bella.data_types import TargetCollection

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__=='__main__':
    dataset_name_choices = ['Restaurant', 'Laptop', 'Election']

    parser = argparse.ArgumentParser()
    lr_dir_help = "File path to the directory to store the models learning rates"
    model_config_help = "File path to the model config directory for the models"
    parser.add_argument("train_fp", help="File path to the training file",
                        type=parse_path)
    parser.add_argument("lr_dir", help=lr_dir_help,
                        type=parse_path, default='./results/learning_rates')
    parser.add_argument("model_config_dir", help=model_config_help, 
                        type=parse_path, default='./model_configs/baseline')
    parser.add_argument("dataset_name", type=str, help='Name of the dataset', 
                        choices=dataset_name_choices)
    parser.add_argument("log_fp", help='File path to log file', 
                        type=parse_path)
    parser.add_argument("--num_batches", type=int, default=100,
                        help="Number of mini-batches to run Learning rate finder.")
    args = parser.parse_args()

    # LOGGING
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(args.log_fp)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Data paths
    lr_dir: Path = args.lr_dir
    lr_dir.mkdir(parents=True, exist_ok=True)
    model_config_dir = args.model_config_dir

    # Models
    model_names = ['atae', 'bilinear', 'ian', 'tdlstm', 'tclstm']
    bilstm_names = ['']
    all_model_names = product(model_names, bilstm_names)
    all_models = []
    for model_name, bilstm_name in all_model_names:
        model_name = f'{model_name}{bilstm_name}'.strip()
        model_config_fp = Path(model_config_dir, f'{model_name}.json')
        all_models.append(AllenNLPModel(model_name, model_config_fp))

    # Data
    dataset_name = args.dataset_name
    train = TargetCollection.load_from_json(args.train_fp)

    model_data = product(all_models, [train])
    for model, train_data in model_data:
        logger.info(f'Finding learning rate for {model.name} on {dataset_name} '
                    f'dataset using {args.num_batches} batches')
        
        model_dir = Path(lr_dir, model.name)
        data_model_dir = Path(model_dir, f'{dataset_name}')
        if not data_model_dir.exists():
            data_model_dir.mkdir(parents=True)
        try:
            model.find_learning_rate(train, data_model_dir, 
                                     {'num_batches': args.num_batches})
        except:
            error_msg = f'Finding learning rate for {model.name} on {dataset_name} ' \
                        f'dataset using {args.num_batches} batches'
            logger.info(error_msg)
            raise ValueError(error_msg)


