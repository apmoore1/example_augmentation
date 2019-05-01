from pathlib import Path
from typing import Tuple

from bella_allen_nlp import AllenNLPModel
from bella.data_types import TargetCollection, Target
import numpy as np
from sklearn.model_selection import train_test_split

SENTI_MAPPER = {'positive': 1, 'neutral': 0, 'negative': -1}

def run_model(model: AllenNLPModel, train: TargetCollection, 
              val: TargetCollection, test: TargetCollection, 
              save_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Returns the array of predictions as shape [n_samples, 1]
    '''
    save_file_name = save_file.name
    test_file = save_file.with_name(f'{save_file_name} test.npy')
    val_file = save_file.with_name(f'{save_file_name} val.npy')
    if test_file.exists() and val_file.exists():
        test_data = np.expand_dims(np.load(test_file), 1)
        val_data = np.expand_dims(np.load(val_file), 1)
        return test_data, val_data
    _ = model.fit(train, val, test)
    test_results = model.predict_label(test, SENTI_MAPPER)
    val_results = model.predict_label(val, SENTI_MAPPER)
    np.save(test_file.resolve(), test_results)
    np.save(val_file.resolve(), val_results)
    return np.expand_dims(test_results, 1), np.expand_dims(val_results, 1)

def run_n_times(model: AllenNLPModel, train: TargetCollection,
                val: TargetCollection, test: TargetCollection,
                save_dir: Path, n: int) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Returns the *n* results over the same training, validation, 
    and test data but fitted each time therefore able to model 
    the random seeds affect.
    
    The returned results will be of shape [n_samples, n_runs]
    '''
    test_results = []
    val_results = []
    for num_times_run in range(n):
        save_file = Path(save_dir, f'run {num_times_run}')
        results = run_model(model, train, val, test, save_file)
        test_result, val_result = results
        test_results.append(test_result)
        val_results.append(val_result)
    return np.concatenate(test_results, 1), np.concatenate(val_results, 1)

def get_result(data_folder: Path, number_runs: int, test=True
               ) -> np.ndarray:
    '''
    Given a folder that contains numpy files of the following 
    naming structure:
    `run {run_number} {val/test}.npy` 
    it will load all run number files for a given dataset (test
    or val) and return a numpy array where the array is of shape:
    *n* * *number_runs* where *n* is the test or val sample size.
    '''
    results = []
    for run_number in range(number_runs):
        data_file_name = f'run {run_number} val.npy'
        if test:
            data_file_name = f'run {run_number} test.npy'
        
        result = np.load(Path(data_folder, data_file_name))
        results.append(np.expand_dims(result, 1))
    return np.concatenate(results, 1)
