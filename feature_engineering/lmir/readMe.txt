lmir is the original version, it can calculate every document with one sentence.

lmir_update is a updated version, it can ony calculate relevance between one document and one query.
usage:
import lmir_update
doc_1 = "This is document one.".split()
doc_2 = "This is document two. It contains different words.".split()
#corpus
docs = [doc_1, doc_2] 

models = lmir_update.LMIR(docs)

#smoothing_1:jelinek_mercer_doc
print(models.jelinek_mercer_doc("This query has words that are found in the corpus.".split(),doc_2))
print(models.jelinek_mercer_doc("No matches.".split(), doc_1))

#smoothing_2:dirichlet_doc
print(models.dirichlet_doc("This query has words that are found in the corpus.".split(),doc_2))
print(models.dirichlet_doc("No matches.".split(), doc_1))

#smoothing_3:absolute_discount_doc
print(models.absolute_discount_doc("This query has words that are found in the corpus.".split(),doc_2))
print(models.absolute_discount_doc("No matches.".split(), doc_1))
