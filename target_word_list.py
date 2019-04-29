import argparse
import json
from pathlib import Path

from bella.data_types import TargetCollection

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("tdsa_dataset_fp", type=parse_path,
                        help="File path to the TDSA dataset")
    parser.add_argument("target_set_fp", type=parse_path,
                        help='File path to save the targets to')
    parser.add_argument
    args = parser.parse_args()
    data = TargetCollection.load_from_json(args.tdsa_dataset_fp)
    all_targets = list(data.target_set())
    with args.target_set_fp.open('w+') as target_set_file:
        json.dump(all_targets, target_set_file)
    