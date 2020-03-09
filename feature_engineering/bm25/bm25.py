#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from six import iteritems
from six.moves import xrange

# BM25 parameters.
PARAM_K1 = 1.5
PARAM_B = 0.75
EPSILON = 0.25


class BM25(object):

    def __init__(self, corpus):
        self.corpus_size = len(corpus)
        self.avgdl = sum(map(lambda x: float(len(x)), corpus)) / self.corpus_size
        self.corpus = corpus
        self.df = {}
        self.idf = {}
        # self.f = []
        self.initialize()
        self.average_idf = sum(map(lambda k: float(self.idf[k]), self.idf.keys())) / len(self.idf.keys())

    def initialize(self):
        for document in self.corpus:
            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            # self.f.append(frequencies)

            for word, freq in iteritems(frequencies):
                if word not in self.df:
                    self.df[word] = 0
                self.df[word] += 1

        for word, freq in iteritems(self.df):
            self.idf[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)

    def get_sim(self, query, doc):
        sim = 0
        doc_word_count = {}
        for word in doc:
            doc_word_count[word] = doc_word_count.get(word, 0) + 1

        # for i, k in enumerate(doc_word_count):
        #    print(str(i) + ":" + str(k) +":"+str(doc_word_count.get(k)))

        for word in query:
            if word not in doc_word_count:
                continue
            idf = self.idf[word] if self.idf[word] >= 0 else EPSILON * self.average_idf
            sim += (idf * doc_word_count[word] * (PARAM_K1 + 1)
                    / (doc_word_count[word] + PARAM_K1 * (1 - PARAM_B + PARAM_B * len(doc) / self.avgdl)))
        return sim

    def get_score(self, query, index):
        doc = self.corpus[index]
        return self.get_sim(query, doc)

    def get_scores(self, query):
        scores = []
        for index in xrange(self.corpus_size):
            score = self.get_score(query, index)
            scores.append(score)
        return scores

    def get_bm25_weights(corpus):
        bm25 = BM25(corpus)
        weights = []
        for doc in corpus:
            scores = bm25.get_scores(doc)
            weights.append(scores)

        return weights



