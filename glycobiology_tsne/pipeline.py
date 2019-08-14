import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pytorch_transformers import BertTokenizer, BertConfig, BertModel
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from torch.nn.utils.rnn import pad_sequence

from glycobiology_tsne.convert_bert import convert

BERT_DIM = 768
MAX_BERT_SEQ_LEN = 512

use_cuda = torch.cuda.is_available()
t = torch.cuda if use_cuda else torch
device = 'cuda:0' if use_cuda else 'cpu'


def _get_custom_bert(pretrained_weights):
    model_fname = 'pytorch_model.bin'
    if model_fname not in os.listdir(pretrained_weights):
        convert(pretrained_weights)
    model_fpath = os.path.join(pretrained_weights, model_fname)
    config_fpath = os.path.join(pretrained_weights, 'bert_config.json')
    config = BertConfig.from_json_file(config_fpath)
    custom_bert = BertModel(config)
    state_dict = torch.load(model_fpath)

    def _remove_prefix(string):
        prefix = 'bert.'
        if string.startswith(prefix):
            string = string[len(prefix):]
        return string

    state_dict = {_remove_prefix(k): v for k, v in state_dict.items() if not k.startswith('cls')}
    custom_bert.load_state_dict(state_dict)
    return custom_bert


class TokenizerTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, pretrained_weights):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_t = map(self.tokenizer.encode, X)
        X_t = [X_i[:MAX_BERT_SEQ_LEN] for X_i in X_t]
        X_t = list(map(torch.LongTensor, X_t))
        X_t = pad_sequence(X_t)
        return X_t.t()


class BertTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, pretrained_weights):
        super().__init__()
        try:
            self.bert = _get_custom_bert(pretrained_weights)
        except FileNotFoundError:
            self.bert = BertModel.from_pretrained(pretrained_weights)
            assert self.bert is not None, '{} is not a valid directory containing bert weights'

        self.bert.to(device)
        self.bert.eval()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        with torch.no_grad():
            _, bert_out = self.bert(X.to(device))
        bert_out = bert_out.cpu().numpy()
        return bert_out


class BertTsnePipeline(Pipeline):

    def __init__(self, pretrained_weights, **tnse_kwargs):
        steps = [('tokenizer', TokenizerTransformer(pretrained_weights)),
                 ('bert', BertTransformer(pretrained_weights)),
                 ('tsne', TSNE(**tnse_kwargs))]
        super(BertTsnePipeline, self).__init__(steps)

    def fit_transform_plot(self, X, c=None):
        X_t = self.fit_transform(X)

        f, axarr = plt.subplots()
        axarr.scatter(X_t[:, 0], X_t[:, 1], c=c)

        for X_i, X_t_i in zip(X, X_t):
            axarr.annotate(X_i, (X_t_i[0], X_t_i[1]))

        return X_t, (f, axarr)
