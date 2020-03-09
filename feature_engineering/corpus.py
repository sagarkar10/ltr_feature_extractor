import math
from typing import List
from collections import defaultdict, OrderedDict
from feature_engineering.lmir import lmir
from feature_engineering.bm25 import bm25


class Corpus(object):
    def __init__(self, doc_list=[]) -> object:
        self.doc_list = doc_list
        self.__length = len(self.doc_list)
        # self.__init_doc_list()
        self.doc_count = 0
        self.__tf = None
        self.__df = None
        self.__idf = None
        self.__lmir_model = None
        self.__bm25_model = None
        self.__calc_global_tf_idf()
        self.__train_lmir()
        self.__train_bm25()

    def __len__(self):
        return self.__length

    def __train_bm25(self):
        self.__bm25_model = bm25.BM25(self.doc_list)
        print('bm25 model training finished')

    def __train_lmir(self):
        self.__lmir_model = lmir.LMIR(self.doc_list)
        print('lmir model training finished')

    def __init_doc_list(self):
        for doc in self.doc_list:
            doc.set_corpus(self)

    def __calc_global_tf_idf(self):
        tf = defaultdict(lambda: 0.0)
        df = defaultdict(lambda: 0.0)
        doc_count = 0
        for doc in self.doc_list:
            doc_unique_words = set()
            for word in doc.token_list:
                tf[word] += 1
                doc_unique_words.add(word)
            for unique_word in doc_unique_words:
                df[unique_word] += 1
            doc_count += 1
        # At least 1.0 for token not exists in corpus
        default_idf = math.log(doc_count / 1.0)
        idf = defaultdict(lambda: default_idf)
        for w, value in df.items():
            idf[w] = math.log(doc_count / float(value + 1.0))
        # Saving globally
        self.doc_count = doc_count
        self.__tf = tf
        self.__df = df
        self.__idf = idf

    def get_idf(self, token):
        """

        :param token:
        :type token:
        :return:
        :rtype:
        """
        return self.__idf[token]

    def get_tf(self, token):
        """
        Get global term frequency in Corpus
        :param token: the token to query
        :return: term frequency, int
        """
        return self.__tf[token]

    def get_igtf(self, token):
        global_tf = self.__tf[token]
        corpus_size = self.__length
        return math.log((corpus_size / (global_tf + 1)) + 1)

    def get_bm25_score(self, query_token_list, doc_token_list):
        return self.__bm25_model.get_sim(query_token_list, doc_token_list)

    def get_lmir_score(self, query_token_list, doc_token_list):
        lmir_abs = self.__lmir_model.absolute_discount_doc(query_token_list, doc_token_list)
        lmir_dir = self.__lmir_model.dirichlet_doc(query_token_list, doc_token_list)
        lmir_jm = self.__lmir_model.jelinek_mercer_doc(query_token_list, doc_token_list)
        return lmir_abs, lmir_dir, lmir_jm
