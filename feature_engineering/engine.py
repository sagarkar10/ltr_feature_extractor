from feature_engineering.utils.file import load_json
from feature_engineering.corpus import Corpus
from feature_engineering.doc import Query, Document, QueryDocPair


def calc_ir_features(query_doc_list, text_list):
    features_list = []
    doc_list = []
    for text in text_list:
        doc = Document(text)
        doc_list.append(doc)
    corpus = Corpus(doc_list)
    print('Corpus prepared')
    total = len(query_doc_list)
    count = 0
    for query_text, doc_text in query_doc_list:
        query = Query(query_text)
        document = Document(doc_text)
        pair = QueryDocPair(query, document, corpus)
        features = pair.get_all_features()
        features_list.append(features)
        count += 1
        if count % 1000 == 0:
            print("{}/{} , {}".format(count, total, count / total))

    return features_list
