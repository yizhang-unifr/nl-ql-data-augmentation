import os
import re
import json
import argparse

"""
This script transform the real SQL queries into more readable queries, 
which can help generative models (e.g. GPT-3) to generate corresponding 
NL questions.
Notes: The generated readable queries might probably not executable for the real database.
"""

# schema_file = "/home/ubuntu/fraunhofer/valuenet/data/skyserver_dr16_2020_11_30/original/tables.json"
# data = ["select p.objid, s.specobjid from photoobj as p join specobj as s on s.bestobjid = p.objid where p.cmodelmag_g > 0 and p.cmodelmag_g < 23 or p.cmodelmagerr_g < 0.2 and p.clean = 1 and s.z <= 0.1 and s.class = 'GALAXY'"]


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--db_schema', help='Path to the database schema file in JSON format', type=str)
    parser.add_argument(
        '--in_path', help='Path to the input queries in JSON format', type=str)
    parser.add_argument(
        '--out_path', help='Path to the input queries in JSON format', type=str)
    return parser.parse_args()


def _transform_tables(sql, schema) -> dict:
    table_names = schema['table_names']
    table_names_original = schema['table_names_original']
    table_names = schema['table_names']
    table_names_original = schema['table_names_original']
    res_dict = {}
    reg_pattern = r'(\w+)\s+as\s+(\w+)'
    matches = re.finditer(reg_pattern, sql, re.IGNORECASE | re.MULTILINE)
    transformed_sql = ''
    start = 0
    for idx, match in enumerate(matches, start=1):
        # print("Match {matchNum} was found at {start}-{end}: {match}".format(matchNum=idx, start=match.start(), end=match.end(), match=match.group()))
        assert len(match.groups()) == 2
        table_name_original = match.group(1).lower()
        alias = match.group(2)
        # replace ' ' with '_'
        table_name = re.sub(
            ' ', '_', table_names[table_names_original.index(table_name_original)])
        replacement = ' '.join([table_name, 'as', alias])
        # TODO replace tables_
        end = match.start()
        transformed_sql += sql[start:end] + replacement
        start = match.end()
        res_dict[alias] = table_name_original
    transformed_sql += sql[start:]
    return transformed_sql, res_dict


def _transform_sql(d, schema) -> dict:
    column_names = schema['column_names']
    column_names_original = schema['column_names_original']
    table_names_original = schema['table_names_original']
    reg_pattern = r'(\w+)\.(\w+)'
    start = 0
    transformed_sql = ''
    sql, alias2table_dict = _transform_tables(d, schema)
    matches = re.finditer(reg_pattern, sql, re.MULTILINE)
    for idx, match in enumerate(matches, start=1):
        # check match.group() is not a float
        if not is_float(match.group()):
            assert len(match.groups()) == 2
            alias = match.group(1)
            table_idx = table_names_original.index(alias2table_dict[alias])
            column_name_original = [table_idx, match.group(2).lower()]
            column_name = re.sub(
                ' ', '_', column_names[column_names_original.index(column_name_original)][1])
            replacement = '.'.join([alias, column_name])
        else:
            replacement = match.group()
        # print("Match {matchNum} was found at {start}-{end}: {match}".format(matchNum=idx, start=match.start(), end=match.end(), match=match.group()))
        end = match.start()
        transformed_sql += sql[start:end] + replacement
        start = match.end()
    transformed_sql += sql[start:]
    return transformed_sql


def sql_transformation(data, schema):
    res = []
    db_id = schema['db_id']
    for d in data:
        transformed_sql = _transform_sql(d, schema)
        format_dict = {
            'db_id': db_id,
            'query': d,
            'readable_query': transformed_sql
        }
        res.append(format_dict)
    return res


def is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def main():
    args = load_args()
    schema_file = args.db_schema
    in_file = args.in_path
    out_file = args.out_path
    with open(schema_file, 'r') as schema_input:
        schema = json.load(schema_input)[0]
    with open(in_file, 'r') as f_in:
        data = json.load(f_in)
    output = sql_transformation(data, schema)
    with open(out_file, 'w') as f_out:
        json.dump(output, f_out, indent=2)


if __name__ == '__main__':
    main()
