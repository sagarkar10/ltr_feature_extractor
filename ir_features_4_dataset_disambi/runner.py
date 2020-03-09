from feature_engineering.utils.file import load_json, load_pickle, save_pickle
from feature_engineering.engine import calc_ir_features
import json


def combine_positive_negative_cases(positive_path, negative_path):
    positive_cases = load_pickle(POSITIVE_CASES_PATH)
    negative_cases = load_pickle(NEGATIVE_CASES_PATH)
    return positive_cases + negative_cases


def get_ir_features(training_set, dataset_meta_list):
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
    label_column = []
    for inst in training_set:
        sent = inst.get('sent_token')

        if len(sent) == 0:
            print(json.dumps(inst))
            continue

        title = inst.get('ds_title_token')
        description = inst.get('ds_description_token')
        mentions = inst.get('ds_mention_list_token')
        whole_doc = inst.get('ds_title_desc_mention_token')

        sent_title_pair.append((sent, title))
        sent_desc_pair.append((sent, description))
        sent_mentions_pair.append((sent, mentions))
        sent_whole_pair.append((sent, whole_doc))

        label_column.append([inst.get('label')])

    ir_features_title_corpus = calc_ir_features(sent_title_pair, title_list)
    ir_features_desc_corpus = calc_ir_features(sent_desc_pair, description_list)
    ir_features_mention_corpus = calc_ir_features(sent_mentions_pair, mentions_list)
    ir_features_whole_doc_corpus = calc_ir_features(sent_whole_pair, title_desc_mentions_list)

    ir_features_all = column_combiner(
        [ir_features_title_corpus, ir_features_desc_corpus, ir_features_mention_corpus, ir_features_whole_doc_corpus,
         label_column])
    return ir_features_all


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


if __name__ == "__main__":
    DATASET_META_TOKENIZED_PATH = '/home/tong/Corpus/RichContext/phrase_1/processed/dataset/data_sets_tokenized.json'
    POSITIVE_CASES_PATH = '/home/tong/Corpus/RichContext/phrase_1/processed/disambiguation_train_data/positive_cases_text_separate.pickle'

    for k in [10]:
        for category in ['full_random']:
            NEGATIVE_CASES_PATH = '/home/tong/Corpus/RichContext/phrase_1/processed/disambiguation_train_data/negative_cases/' + category + '/negative_cases_ratio_' + str(
                k) + '.pickle'
            IR_FEATURES_PATH = '/home/tong/Corpus/RichContext/phrase_1/processed/ir_features/' + category + '/negative_cases_ratio_' + str(
                k) + '.pickle'
            training_set = combine_positive_negative_cases(POSITIVE_CASES_PATH, NEGATIVE_CASES_PATH)
            dataset_meta_list = load_json(DATASET_META_TOKENIZED_PATH)
            ir_features = get_ir_features(training_set, dataset_meta_list)
            save_pickle(ir_features, IR_FEATURES_PATH)
