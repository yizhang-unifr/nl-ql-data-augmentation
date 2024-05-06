import json
import logging
import os
import argparse

def load_args():
    parser = argparse.ArgumentParser(description)
    parser.add_argument('--labeled_data_path', type=str, required=True)
    parser.add_argument('--output_data_path', type=str, required=True)
    return parser.parse_args()


"""
input: labeled auto-generated data
output: only positive labeled data samples with aligned format to the `handmade_data_train`
"""

def remove_dict_keys(d, keys):
    for key in keys:
        if key in d.keys():
            del d[key]

def main():
    logger = logging.getLogger(__name__)
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(format=FORMAT)
    args = load_args()
    """
    dir_path = os.path.dirname(__file__)
    parent_path = os.path.dirname(dir_path)
    root_path = os.path.dirname(parent_path)
    data_path = os.path.join(
        root_path, 'data', 'skyserver_dr16_2020_11_30', 'data_aug')
    labeled_data_filename = 'labeled_dataset_sdss.json'
    labeled_data_path = os.path.join(data_path, labeled_data_filename)
    output_data_filename = 'sdss_generated_dataset.json'
    output_data_path = os.path.join(data_path, output_data_filename)
    """
    
    with open(labeled_data_path, 'r') as f_in:
        data = json.load(f_in)
    res = []
    keys = ['id', 'label', 'readable_query']
    for d in data:
        if d['label'] == 1:
            remove_dict_keys(d, keys=keys)
            res.append(d)
    with open(output_data_path, 'w') as f_out:
        json.dump(res, f_out, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
