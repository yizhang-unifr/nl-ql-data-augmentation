import os
import json

"""
generates trainging data for CRITIC model
"""

dir_path = os.path.dirname(os.path.realpath(__file__))
par_path = os.path.dirname(dir_path)

input_file = os.path.join(par_path, "critic", "data",
                          "spider", "generated_spider_for_critic.json")
output_file = os.path.join(par_path, "critic", "data",
                           "spider", "train_on_critic.json")

"""
{
    "db_id": "department_management",
    "generated_ast": "Root1(3) Root(3) Sel(0) N(0) A(3) Op(0) C(0) T(1) Filter(4) A(0) Op(0) C(9) T(1) V(0)",
    "generated_query": "SELECT COUNT(*) FROM head AS T1 WHERE T1.age < 56",
    "generated_values": [
      "56"
    ],
    "original_AST": "Root1(3) Root(3) Sel(0) N(0) A(3) Op(0) C(0) T(1) Filter(5) A(0) Op(0) C(9) T(1) V(0)",
    "original_query": "SELECT COUNT(*) FROM head AS T1 WHERE T1.age > 56",
    "original_values": [
      "56"
    ],
    "question": "How many heads of the departments are older than 56 ?"
}
"""


def postive_labeled_data(d, i):
    label = 1
    res = {
        'db_id': d['db_id'],
        'id': i,
        'question': d['question'],
        'query': d['original_query'],
        'label': label
    }
    return res


def negative_labeled_data(d, i):
    label = 0
    res = {
        'db_id': d['db_id'],
        'id': i,
        'question': d['question'],
        'query': d['generated_query'],
        'label': label
    }
    return res


def dataset_builder(data):
    res = []
    i = 1
    for d in data:
        res.append(postive_labeled_data(d, i))
        i += 1
        res.append(negative_labeled_data(d, i))
        i += 1
    return res


def main():
    with open(input_file, 'r') as f_in:
        data = json.load(f_in)
    data = dataset_builder(data)
    with open(output_file, 'w') as f_out:
        json.dump(data, f_out, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
