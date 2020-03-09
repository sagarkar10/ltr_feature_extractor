from feature_engineering.corpus import Corpus
from feature_engineering.doc import Doc

d1 = Doc('This is TOng')
d2 = Doc('I Like play FOOTBALL')

corpus = Corpus([d1, d2, d1, d2, d1, d2, d1, d2])
print(corpus.get_idf('haha'))
print(corpus.get_idf('Tong'))
print(corpus.get_tf('TONG'))
print(corpus.get_idf('I'))
print(corpus.doc_count)
print(corpus.__tf)
print(corpus.__df)
print(corpus.__idf)
print(corpus.doc_list)
