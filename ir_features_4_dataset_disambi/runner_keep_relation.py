import numpy

from feature_engineering.utils.file import load_json, load_pickle, save_pickle
from feature_engineering.engine import calc_ir_features
import json
import pandas as pd


def get_ir_features(training_df, dataset_meta_list):
    title_list = []
    description_list = []
    mentions_list = []
    title_desc_mentions_list = []
    for dataset in dataset_meta_list:
        title = dataset.get('title_token')
        description = dataset.get('description_token')
        mentions = dataset.get('mention_list_token')
        title_desc_mentions = dataset.get('title_desc_mention_token')

        title_list.append(title)
        description_list.append(description)
        mentions_list.append(mentions)
        title_desc_mentions_list.append(title_desc_mentions)

    sent_title_pair = []
    sent_desc_pair = []
    sent_mentions_pair = []
    sent_whole_pair = []
    for index, row in training_df.iterrows():
        ent_sent = row.get('entity_sent_token')

        if len(ent_sent) == 0:
            print(json.dumps(row))
            continue

        title = row.get('ds_title_token')
        description = row.get('ds_description_token')
        mentions = row.get('ds_mention_list_token')
        whole_doc = row.get('ds_title_desc_mention_token')

        sent_title_pair.append((ent_sent, title))
        sent_desc_pair.append((ent_sent, description))
        sent_mentions_pair.append((ent_sent, mentions))
        sent_whole_pair.append((ent_sent, whole_doc))

    ir_features_title_corpus = calc_ir_features(sent_title_pair, title_list)
    print('ir_features_title_corpus finished')
    ir_features_desc_corpus = calc_ir_features(sent_desc_pair, description_list)
    print('ir_features_desc_corpus finished')
    ir_features_mention_corpus = calc_ir_features(sent_mentions_pair, mentions_list)
    print('ir_features_mention_corpus finished')
    ir_features_whole_doc_corpus = calc_ir_features(sent_whole_pair, title_desc_mentions_list)
    print('ir_features_whole_doc_corpus finished')

    ir_features_all = column_combiner(
        [ir_features_title_corpus, ir_features_desc_corpus, ir_features_mention_corpus, ir_features_whole_doc_corpus])
    print('combiner finished')

    training_df["ir_features_all"] = ir_features_all

    return training_df


def column_combiner(lol):
    first_size = len(lol[0])
    for ele in lol:
        if not len(ele) == first_size:
            raise Exception("the two lists are not in same size")
    combined_list = []
    for i in range(first_size):
        combined = []
        for ele in lol:
            combined += ele[i]
        combined_list.append(combined)
    return combined_list


def save_to_jsonl(ir_features_df, path):
    def default(o):
        if isinstance(o, numpy.int64): return int(o)
        raise TypeError
    with open(path, 'w') as f:
        for index, row in ir_features_df.iterrows():
                f.write(json.dumps({
                    'ir_features_all':row.get('ir_features_all'),
                    'label':(row.get('label')),
                    'sent_id':(row.get('sent_id')),
                    'data_set_id':(row.get('data_set_id'))
                }, default=default) + '\n')


# if __name__ == "__main__":
DATASET_META_TOKENIZED_PATH = '/home/tong/Corpus/RichContext/phrase_1/processed/dataset/data_sets_tokenized.json'
LTR_CASES_TRAIN_PATH = "/home/tong/Corpus/RichContext/phrase_1/processed/ltr/ltr_cases/top10/ltr_cases.train.pickle"
LTR_CASES_VALID_PATH = "/home/tong/Corpus/RichContext/phrase_1/processed/ltr/ltr_cases/top10/ltr_cases.valid.pickle"
LTR_FEATURES_TRAIN_PATH = '/home/tong/Corpus/RichContext/phrase_1/processed/ltr/ltr_features/top10/ltr_feature.train.jsonl'
LTR_FEATURES_VALID_PATH = '/home/tong/Corpus/RichContext/phrase_1/processed/ltr/ltr_features/top10/ltr_feature_query_sim.valid.jsonl'
dataset_meta_list = load_json(DATASET_META_TOKENIZED_PATH)

for ltr_cases_path, ltr_features_path in [
    (LTR_CASES_TRAIN_PATH, LTR_FEATURES_TRAIN_PATH),
    (LTR_CASES_VALID_PATH, LTR_FEATURES_VALID_PATH)]:
ltr_cases = load_pickle(ltr_cases_path)
ltr_cases_df = pd.DataFrame(ltr_cases)
ltr_cases_df = ltr_cases_df[
    ['sent_id', 'data_set_id', 'entity_sent_token', 'ds_title_token', 'ds_description_token',
     'ds_mention_list_token',
     'ds_title_desc_mention_token', 'label']]
ir_features_df = get_ir_features(ltr_cases_df, dataset_meta_list)
save_to_jsonl(ir_features_df, ltr_features_path)

        # ir_features_df.to_pickle(ltr_features_path)
