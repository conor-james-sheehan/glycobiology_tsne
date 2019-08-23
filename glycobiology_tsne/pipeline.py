import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import fasttext
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


def labelled_scatter_plot(labels, x, y, c=None):
    f, axarr = plt.subplots()
    axarr.scatter(x, y, c=c)

    for x_i, y_i, l_i in zip(x, y, labels):
        axarr.annotate(l_i, (x_i, y_i))

    return f, axarr


class BaseTsnePipeline(Pipeline):

    def __init__(self, embedding_transformer, **tsne_kwargs):
        steps = [('embedding', embedding_transformer),
                 ('tsne', TSNE(**tsne_kwargs))]
        super(BaseTsnePipeline, self).__init__(steps)

    def fit_transform_plot(self, X, c=None):
        X_t = self.fit_transform(X)
        f, axarr = labelled_scatter_plot(X, X_t[:, 0], X_t[:, 1], c=c)
        return X_t, (f, axarr)


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


class BertTokenizerTransformer(BaseEstimator, TransformerMixin):

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


class BertTsnePipeline(BaseTsnePipeline):

    def __init__(self, pretrained_weights, **tsne_kwargs):
        tokenizer_transformer = BertTokenizerTransformer(pretrained_weights)
        embedding_transformer = BertTransformer(pretrained_weights)
        embedding = Pipeline([('tokenizer', tokenizer_transformer), ('embedding', embedding_transformer)])
        super(BertTsnePipeline, self).__init__(embedding, **tsne_kwargs)


class FastTextEmbeddingTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, pretrained_weights):
        self.model = fasttext.load_model(pretrained_weights)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array(list(map(self.model.get_word_vector, X)))


class FastTextPipeline(BaseTsnePipeline):

    def __init__(self, pretrained_weights, **tsne_kwargs):
        embedding = FastTextEmbeddingTransformer(pretrained_weights)
        super(FastTextPipeline, self).__init__(embedding, **tsne_kwargs)


class DummyPipeline(BaseTsnePipeline):
    # for debugging

    class DummyEmbeddingTransformer(BaseEstimator, TransformerMixin):

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return np.random.rand(len(X), 100)

    def __init__(self, **tsne_kwargs):
        super(DummyPipeline, self).__init__(self.DummyEmbeddingTransformer(), **tsne_kwargs)
