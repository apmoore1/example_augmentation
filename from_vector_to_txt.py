import argparse
from pathlib import Path

from gensim.models.word2vec import Word2Vec

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("vector_fp", type=parse_path, 
                        help='File Path to the vector file in binary format')
    parser.add_argument("text_fp", type=parse_path, 
                        help='File Path to the text vector file to save to')
    args = parser.parse_args()

    vector_fp = args.vector_fp
    text_vector_fp = args.text_fp

    embedding = Word2Vec.load(str(vector_fp))
    with text_vector_fp.open('w+') as text_vector_file:
        for index, word in enumerate(embedding.wv.vocab):
            line = ''
            if index != 0:
                line += '\n'
            word_vector = embedding.wv[word]
            word_vector = [str(vector_number) for vector_number in word_vector]
            word_vector = ' '.join(word_vector).strip()
            line += f'{word} {word_vector}'
            text_vector_file.write(line)

