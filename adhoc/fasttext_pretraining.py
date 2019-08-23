import os
from pkg_resources import resource_filename
from fasttext import train_supervised

TEXTBOOK_PATH = os.path.join(resource_filename('glycobiology_tsne', 'data'), 'textbook.txt')


def train_model(pretrained_weights, output_path):
    model = train_supervised(TEXTBOOK_PATH, pretrainedVectors=pretrained_weights)
    model.save_model(output_path)
