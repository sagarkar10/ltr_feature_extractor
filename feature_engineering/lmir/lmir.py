# author: longfengwu
# date 2018-10-30

from math import log


class LMIR:
    def __init__(self, corpus, lamb=0.1, mu=2000, delta=0.7):
        """Use language models to score query/document pairs.

        :param corpus:
        :param lamb:
        :param mu:
        :param delta:
        """
        self.lamb = lamb
        self.mu = mu
        self.delta = delta

        # Fetch all of the necessary quantities for the document language
        # models.
        doc_lens = []
        all_token_counts = {}
        for doc in corpus:
            doc_len = len(doc)
            doc_lens.append(doc_len)
            for token in doc:
                all_token_counts[token] = all_token_counts.get(token, 0) + 1

        total_tokens = sum(all_token_counts.values())
        p_C = {token: token_count / total_tokens
               for (token, token_count) in all_token_counts.items()}

        self.doc_lens = doc_lens
        self.p_C = p_C

    def jelinek_mercer_doc(self, query_tokens, doc_tokens):
        """Calculate the Jelinek-Mercer scores for a given query and a given document"""
        doc_len = len(doc_tokens)
        if doc_len <= 0:
            return 0.0

        lamb = self.lamb
        p_C = self.p_C

        doc_token_counts = {}
        for token in doc_tokens:
            doc_token_counts[token] = doc_token_counts.get(token, 0) + 1

        p_ml = {}
        for token in doc_token_counts:
            p_ml[token] = doc_token_counts[token] / doc_len

        score = 0
        for token in query_tokens:
            if token not in p_C:
                continue

            score -= log((1 - lamb) * p_ml.get(token, 0) + lamb * p_C[token])

        return score

    def dirichlet_doc(self, query_tokens, doc_tokens):
        """Calculate the Dirichlet scores for a given query and a given document"""
        doc_len = len(doc_tokens)
        if doc_len <= 0:
            return 0.0

        mu = self.mu
        p_C = self.p_C

        doc_token_counts = {}
        for token in doc_tokens:
            doc_token_counts[token] = doc_token_counts.get(token, 0) + 1

        p_ml = {}
        for token in doc_token_counts:
            p_ml[token] = doc_token_counts[token] / doc_len

        score = 0
        for token in query_tokens:
            if token not in p_C:
                continue

            score -= log((doc_token_counts.get(token, 0) + mu * p_C[token]) / (doc_len + mu))

        return score

    def absolute_discount_doc(self, query_tokens, doc_tokens):
        """Calculate the absolute discount scores for a given query and a given document"""

        doc_len = len(doc_tokens)
        if doc_len <= 0:
            return 0.0

        delta = self.delta
        p_C = self.p_C

        doc_token_counts = {}
        for token in doc_tokens:
            doc_token_counts[token] = doc_token_counts.get(token, 0) + 1

        d_u = len(doc_token_counts)

        p_ml = {}
        for token in doc_token_counts:
            p_ml[token] = doc_token_counts[token] / doc_len

        score = 0
        for token in query_tokens:
            if token not in p_C:
                continue

            score -= log(max(doc_token_counts.get(token, 0) - delta, 0) / doc_len + delta * d_u / doc_len * p_C[token])

        return score
