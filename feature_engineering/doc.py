from collections import defaultdict

import numpy as np
from feature_engineering.corpus import Corpus
import math
from feature_engineering.text2vector import weighted_text2vec
from scipy.spatial import distance


class DocBase(object):
    def __init__(self, token_list):
        self.token_list = token_list
        self.__length = len(self.token_list)
        self.__tf = None
        self.__calc_tf()

    def __len__(self):
        return self.__length

    def __iter__(self):
        for token in self.token_list:
            yield token

    def __str__(self):
        return str(self.token_list)

    def __repr__(self):
        return self.__str__()

    def __calc_tf(self):
        __tf = defaultdict(lambda: 0.0)
        for word in self.token_list:
            __tf[word] += 1
        self.__tf = __tf

    def get_tf(self, token):
        return self.__tf[token]

    def get_weighted_tf(self, token):
        if self.__length == 0:
            return 0
        return self.__tf[token] / self.__length

    def has_token(self, token):
        return token in self.token_list


class Query(DocBase):
    def __init__(self, text):
        if len(text) == 0:
            raise Exception('query length should not zero')
        super().__init__(text)


class Document(DocBase):

    def __init__(self, text):
        super().__init__(text)
        self.corpus = None

    def set_corpus(self, corpus):
        self.corpus = corpus


class QueryDocPair(object):
    def __init__(self, query, doc, corpus):
        """

        :param query: the query
        :type query: Query
        :param doc: the document
        :type doc: Document
        """
        self.query = query
        self.doc = doc
        self.corpus = corpus

    def calc_covered_query_term_number_log_and_ratio(self):
        """
        count the frequency and ratio of query terms which appears in the document
        Relationship: Q-D
        :return:
        :rtype: tuple(int, float)
        """
        query_tokens = self.query.token_list
        if len(self.query) == 0:
            return 0.0, 0.0, 0.0
        covered_count = 0
        for token in query_tokens:
            if self.doc.has_token(token):
                covered_count += 1
        ratio = covered_count / len(self.query)
        return covered_count, math.log(covered_count + 1), ratio

    def calc_doc_length(self):
        """
        get the length of document
        Relationshop: D
        :return: the length of document
        :rtype: int
        """
        return len(self.doc)

    def calc_tf_idf_stats(self):
        """
        calc tf and idf related features.
        TF: calc the term frequency statistics.
        normalized-TF: calc doc length normalized term frequency statistics.
        IDF: calc the IDF statistics
        TF-IDF: calc the TF-IDF statistics.
        Statistics: sum, min, max, mean, variance
        The terms are from query, tf is counted in the document, idf calc in the corpus which contains the document
        :return: tf_sum, tf_min, tf_max, tf_mean, tf_variance, tf_sum / doc_length, tf_min / doc_length, tf_max / doc_length, tf_mean / doc_length, tf_variance / doc_length, idf_sum, idf_min, idf_max, idf_mean, idf_variance, tfidf_sum, tfidf_min, tfidf_max, tfidf_mean, tfidf_variance
        :rtype: list
        """

        def statistic(nda):
            _sum = np.sum(nda)
            _min = np.amin(nda)
            _max = np.amax(nda)
            _mean = np.mean(nda)
            _variance = np.var(nda)
            return _sum, _min, _max, _mean, _variance

        query_tokens = self.query.token_list
        tf_list = []
        doc_length_weighted_tf_list = []
        idf_list = []
        igtf_list = []
        for token in query_tokens:
            tf_value = self.doc.get_tf(token)
            weighted_tf_value = self.doc.get_weighted_tf(token)
            idf_value = self.corpus.get_idf(token)
            igtf = self.corpus.get_igtf(token)
            tf_list.append(tf_value)
            doc_length_weighted_tf_list.append(weighted_tf_value)
            idf_list.append(idf_value)
            igtf_list.append(igtf)
        tf_nda = np.array(tf_list)
        weighted_tf_nda = np.array(doc_length_weighted_tf_list)
        idf_nda = np.array(idf_list)
        igtf_nda = np.array(igtf_list)
        tf_idf_nda = np.multiply(tf_nda, idf_nda)
        weighted_tf_idf_nda = np.multiply(weighted_tf_nda, idf_nda)

        idf_sum = np.sum(idf_nda)
        igtf_sum = np.sum(igtf_nda)

        ret = []
        ret.extend(statistic(tf_nda))
        ret.extend(statistic(weighted_tf_nda))
        ret.append(math.log(igtf_sum))
        ret.append(idf_sum)
        ret.append(math.log(idf_sum))
        ret.extend(statistic(tf_idf_nda))
        ret.extend(statistic(weighted_tf_idf_nda))
        return ret

    def calc_bool(self):
        """
        is there any query term appears in doc ?
        :return: is there any query term appears in doc ?
        :rtype: bool
        """
        query_tokens = self.query.token_list
        has_token = 0
        for token in query_tokens:
            if self.doc.has_token(token):
                has_token = 1
                break
        return has_token

    def calc_bm25(self):
        query_tokens = self.query.token_list
        doc_tokens = self.doc.token_list
        bm25_score = self.corpus.get_bm25_score(query_tokens, doc_tokens)
        return bm25_score

    def calc_lmir(self):
        query_tokens = self.query.token_list
        doc_tokens = self.doc.token_list
        lmir_score_tuple = self.corpus.get_lmir_score(query_tokens, doc_tokens)
        return lmir_score_tuple

    def calc_wv_cosine_sim(self):
        query_wv = weighted_text2vec(self.query.token_list)
        doc_wv = weighted_text2vec(self.doc.token_list)
        sim = 1 - distance.cosine(query_wv, doc_wv)
        if not (1 >= sim >= 0):
            return 0.0
        return sim

    def get_all_features(self):
        ret = []
        ret.extend(self.calc_covered_query_term_number_log_and_ratio())
        ret.append(self.calc_doc_length())
        ret.extend(self.calc_tf_idf_stats())
        # ret.append(self.calc_bool())
        ret.append(self.calc_bm25())
        ret.extend(self.calc_lmir())
        ret.append(self.calc_wv_cosine_sim())
        return ret
