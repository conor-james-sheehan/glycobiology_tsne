import os
from fasttext import train_supervised

TEXTBOOK_PATH = os.path.join('..', 'glycobiology_tsne', 'data', 'textbook.txt')


def train_model(pretrained_weights, output_path):
    model = train_supervised(pretrainedVectors=pretrained_weights)
    model.save_model(output_path)
