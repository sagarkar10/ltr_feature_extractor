gensim
import gensim_bm25
corpus = [
      ["black", "cat", "white", "cat"],
      ["cat", "outer", "space"],
      ["wag", "dog"]
  ]

bm25Model = gensim_bm25.BM25(corpus)
average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())
query=["my","cat"]
scores=bm25Model.get_scores(query, average_idf)

sim=bm25Model.get_sim(query,corpus[0], average_idf)

