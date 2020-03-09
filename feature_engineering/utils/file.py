import pickle
import json
import os


def load_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
        print('read from pickle successfully')
        return obj


def save_pickle(obj, path):
    with open(path, 'w+b') as f:
        print('dumping pickle...')
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        print('dumping XML_FILE_LIST finished')


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
        return data


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)


def save_jsonl(data, path):
    with open(path, 'w') as f:
        for obj in data:
            f.write(json.dumps(obj)+'\n')


def convert_pickle_to_json(list_of_pickles):
    print('----------unpickle------------')
    for path in list_of_pickles:
        pickle_path = path + '.pickle'
        json_path = path + '.json'
        pickle_data = load_pickle(pickle_path)
        print('there are {} items after unpickle'.format(len(pickle_data)))
        with open(json_path, 'w') as outfile:
            json.dump(pickle_data, outfile)


def get_file_path_in_folder(dir, ext_list):
    xml_path_list = []
    for dirpath, dirnames, files in os.walk(dir):
        for names in files:
            names_lower = names.lower()
            for ext in ext_list:
                if names_lower.endswith(ext):
                    path = os.path.join(dirpath, names)
                    # print(path)
                    xml_path_list.append(path)
    print('reading complete: there are {} files'.format(len(xml_path_list)))
    return xml_path_list
