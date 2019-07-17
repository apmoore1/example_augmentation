import argparse
from pathlib import Path

from ftfy import fix_encoding, fix_text


def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("data_fp", type=parse_path,
                        help='File path to the data that is to be checked for unicode errors')
    parser.add_argument("--fixed_data_fp", type=parse_path,
                        help='Whether to fix the data and if so the file to put the data into')
    args = parser.parse_args()

    encoding_errors = []
    text_errors = []
    with args.data_fp.open('r') as sentences:
        for index, sentence in enumerate(sentences):
            fixed_encoding = fix_encoding(sentence)
            fixed_text = fix_text(sentence)
            
            normal_text_len = len(sentence)
            if len(fixed_encoding) != normal_text_len:
                encoding_errors.append((fixed_encoding, sentence))
            if len(fixed_text) != normal_text_len:
                text_errors.append((fixed_text, sentence))
    print(f'Number of sentences in dataset: {index}')
    print(f'Number of text errors: {len(text_errors)}')
    if text_errors:
        print(f'Example of text error: {text_errors[0]}')
    print(f'Number of encoding errors: {len(encoding_errors)}')
    if encoding_errors:
        print(f'Example of encoding error: {encoding_errors[0]}')
