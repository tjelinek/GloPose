# Python 2.7 code
import pickle
import json


def convert_pickle_to_json(pickle_file, json_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

    with open(json_file, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


convert_pickle_to_json('data.pkl', 'data.json')
