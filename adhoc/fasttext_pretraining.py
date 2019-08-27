import os
from pkg_resources import resource_filename
from fasttext import train_supervised

CORPUS_PATH = os.path.join(resource_filename('glycobiology_tsne', 'data'), 'glyco_corpus.txt')


def train_model(output_path):
    assert os.path.exists(CORPUS_PATH)
    model = train_supervised(CORPUS_PATH)
    model.save_model(output_path)
