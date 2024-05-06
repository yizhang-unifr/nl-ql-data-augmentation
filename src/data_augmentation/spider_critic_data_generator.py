import os
import json
from sql_generator_for_critic import load_datasets, sql2semQL, ast_parser, upper_all_keywords, semQL2sql, generate_new_ast
import logging

'''
A convinient way to to generate spider CRITIC dataset
'''


logger = logging.getLogger(__name__)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)

par_path = os.path.dirname(os.path.dirname(__file__))
root_path = os.path.dirname(par_path)
table_path = os.path.join(root_path, "data/spider/original/tables.json")

data_path = os.path.join(root_path, "data/spider/original/dev.json")
output_path = os.path.join(par_path, "critic/data/spider/dev_on_critic.json")


def main():
    data, tables = load_datasets(data_path, table_path)
    data, tables = sql2semQL(data, tables)
    res = []
    failed = 0
    start_idx = 0
    for i, d in enumerate(data):
        try:
            d = ast_parser(d)
        except Exception as e:
            failed += 1
            logging.error(
                f"Error parsing, skipping the {i}th data.\nQuery: {d['query']}\nQuestion: {d['question']}\n")
            continue
        temp_pos = {
            'db_id': d['db_id'],
            'question': d['question'],
            'id': start_idx+1,
            'label': 1
        }
        temp_pos['query'] = upper_all_keywords(
            semQL2sql(d, tables[d['db_id']], origin=d['rule_label'])[0].strip())
        start_idx += 1
        res.append(temp_pos)
        try:
            d = generate_new_ast(d)
        except Exception as e:
            failed += 1
            logging.error(
                f"Error parsing, skipping the {i}th data.\nQuery: {d['query']}\nQuestion: {d['question']}\n")
            continue
        temp_neg = {
            'db_id': d['db_id'],
            'question': d['question'],
            'id': start_idx + 1,
            'label': 0
        }
        temp_neg['query'] = upper_all_keywords(
            semQL2sql(d, tables[d['db_id']], origin=d['generated_ast'])[0].strip())
        start_idx += 1
        res.append(temp_neg)
    logging.error(
        f"###\n# Total Data: {len(data)}\n# Generated successfully: {len(res)}\n# Failed to transform: {failed}\n###")
    # print(res[0])
    try:
        with open(output_path, 'w') as f_out:
            json.dump(res, f_out, indent=2, sort_keys=True)
    except IOError as e:
        logging.error(f"Unexpected error {e}")


if __name__ == "__main__":
    main()
