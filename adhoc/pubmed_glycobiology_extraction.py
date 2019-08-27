import os
import json
from functools import reduce
import operator
from operator import attrgetter
from nltk.tokenize import sent_tokenize

from pymed import PubMed

GLYCOBIOLOGY_KWDS = ('glycan', 'saccharide')
QUERY = 'glycan'


def _is_relevant(result):
    relevance = False
    try:
        abstract = result.abstract
        relevance = any([kwd in abstract.lower() for kwd in GLYCOBIOLOGY_KWDS])
    except AttributeError:
        pass
    return relevance


def _join_lines(l1, l2):
    return l1 + '\n' + l2


def get_corpus(output_dir='.'):
    assert os.path.exists(output_dir)
    pmed = PubMed()
    results = pmed.query('glycan', max_results=100000)
    results = filter(_is_relevant, results)
    ids = map(attrgetter('pubmed_id'), results)
    abstracts = map(attrgetter('abstract'), results)
    del results
    results = dict(zip(ids, abstracts))
    print('Fetched {} results'.format(len(results)))
    print('Writing .json file')
    with open(os.path.join(output_dir, 'glyco_corpus.json'), 'w+') as outfile:
        json.dump(results, outfile)
    print('Tokenizing sentences')
    results_txt = map(sent_tokenize, results.values())
    results_txt = reduce(operator.concat, results_txt)
    results_txt = reduce(_join_lines, results_txt)
    print('Writing .txt file')
    with open(os.path.join(output_dir, 'glyco_corpus.txt'), 'w+') as outfile:
        outfile.write(results_txt)


if __name__ == '__main__':
    get_corpus()
