from collections import defaultdict

from gensim.models import FastText
import numpy as np
from feature_engineering.utils.file import load_pickle

PATH_TO_MODEL = '/home/tong/Corpus/FastText/cc.en.300/cc.en.300.bin'
model = FastText.load_fasttext_format(PATH_TO_MODEL)
print('Fast text model loaded')

IDF_PATH = '/home/tong/Corpus/RichContext/phrase_1/processed/idf/idf_weight_global.pickle'
IDF_DICT = load_pickle(IDF_PATH)
MAX_IDF = max(IDF_DICT.values())

word2weight = defaultdict(
    lambda: MAX_IDF,
    [(word, weight) for word, weight in IDF_DICT.items()])


def text2vec(doc):
    dim = len(model.wv['a'])
    word_wv_list = []
    for word in doc:
        try:
            vec = model.wv[word]
        # out of char n-gram vocab
        except KeyError:
            vec = np.zeros(dim)
        word_wv_list.append(vec)
    # sent may empty
    if len(word_wv_list) == 0:
        word_wv_list.append(np.zeros(dim))

    return np.mean(word_wv_list, axis=0)


weighted_vec_cache = {}


def weighted_text2vec(doc):
    doc_text = "".join(doc)
    cached_vec = weighted_vec_cache.get(doc_text)
    if cached_vec is not None:
        return cached_vec
    dim = len(model.wv['a'])
    word_wv_list = []
    for word in doc:
        try:
            vec = model.wv[word]
        # out of char n-gram vocab
        except KeyError:
            vec = np.zeros(dim)
        word_wv_list.append(vec * word2weight[word])
    # sent may empty
    if len(word_wv_list) == 0:
        word_wv_list.append(np.zeros(dim))
    doc_vec = np.mean(word_wv_list, axis=0)
    weighted_vec_cache[doc_text] = doc_vec
    return doc_vec
