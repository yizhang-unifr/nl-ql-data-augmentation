import argparse
import json
import random
from collections import Counter
from operator import itemgetter
from typing import Tuple, List
from pathlib import Path
import pandas as pd

from matplotlib import pyplot as plt

from synthetic_data.helper import map_semql_actions_only


def group_query_types(data_path: str) -> Tuple[Counter, List]:

    with open(data_path, 'r') as fp:
        data = json.load(fp)

    # There are often two samples which share the same SQL (so 2 NL-questions, but same query). To avoid leaking information
    # when sampling later we remove duplicates.
    data = deduplicate_queries(data)

    all_semql = [sample['rule_label'] for sample in data]
    all_semql_action_only = [map_semql_actions_only(s) for s in all_semql]

    grouped_semql = Counter(all_semql_action_only)

    return grouped_semql, data


def deduplicate_queries(data):
    data_by_sql = {}

    for example in data:
        data_by_sql[example['query']] = example

    data_deduplicated = list(data_by_sql.values())

    return data_deduplicated


def main(data_path:str):
    grouped_semql, data = group_query_types(data_path)

    # plt.hist(list(map(itemgetter(1), grouped_semql.most_common())), range=[0, 75], bins=50)
    # plt.show()

    res = []

    for idx, (k, v) in enumerate(grouped_semql.most_common()):
        # print(f'The following appeared {v} times: {k} Example questions are:')
        example_questions = [e for e in data if map_semql_actions_only(e['rule_label']) == k]

       # random_samples = random.sample(example_questions, k=min(10, len(example_questions)))
        random_samples = random.sample(example_questions, k = 1)
        for example in random_samples:
            temp = {
                'template_id': idx,
                'template': k,
                'count': v,
                'sample_db_id': example['db_id'],
                'sample_question':example['question'],
                'sample_query': example['query'],
                'sample_semql': example['rule_label']
            }
            res.append(temp)
            # print(example['question'])
            # print(example['query'])
            # print()
    df = pd.DataFrame(res)
    output_file = Path(data_path).parent / 'templates_report_dev.xlsx'
    if not output_file.exists():
        with pd.ExcelWriter(output_file) as writer:
            df.to_excel(writer, index=False)
            writer.save()
    else:
        print('file existed. Task cancelled.')
        
        # print()
        # print()



if __name__ == '__main__':
    random.seed(42)

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_containing_semql', type=str, default='data/spider/dev.json')

    args = arg_parser.parse_args()
    main(args.data_containing_semql)
