import json
import re
from pathlib import Path
from data_augmentation.utilities import load_data, strip_sql, get_table_dict, is_joined_query, parser_sql
import argparse

def load_parser():
    parser = argparse.ArgumentParser(description="Adding virtual foreign key from data samples")
    parser.add_argument('--dataset', default='spider', type=str)
    parser.add_argument('--old_tables', default='__tables.json', type=str, help='the path of old table schema: data/<dataset>/original/<old_tables>')
    parser.add_argument('--new_tables', default='tables.json', type=str, help='the path of new table schema: data/<dataset>/original/<new_tables>')
    parser.add_argument('--data_file', default='handmade_data_train.json', type=str, help='the path of data file: data/<dataset>/handmade_training_data/<data_file>')
    return parser

def update_fks(fk, schema):
    tab_1, col_1, tab_2, col_2 = fk[0].lower(), fk[1].lower(), fk[2].lower(), fk[3].lower()
    tables = schema['table_names_original']
    tables = [table.lower() for table in tables]
    cols = schema['column_names_original']
    cols = [[col[0], col[1].lower()] for col in cols]
    try:
        tab_1_idx = tables.index(tab_1.lower())
        tab_2_idx = tables.index(tab_2.lower())
        _col_1 = [tab_1_idx, col_1]
        _col_2 = [tab_2_idx, col_2]
        col_1_idx = cols.index(_col_1)
        col_2_idx = cols.index(_col_2)
        temp_fk = [col_1_idx, col_2_idx]
        temp_fk.sort(reverse=True)
        if temp_fk not in schema['foreign_keys'] and temp_fk[::-1] not in schema['foreign_keys']: # new FK found
            print(f'New FK in {schema["db_id"]} found: {temp_fk}')
            print(f'Original FKs: {schema["foreign_keys"]}')
            schema['foreign_keys'].append(temp_fk)
            print(f'Updated FKs: {schema["foreign_keys"]}')
    except ValueError as e:
        print("***\n", e, fk, '\n***')
    return schema['foreign_keys']
    
def add_missing_fks(data, tables):
    table_dict = get_table_dict(tables)
    for d in data:
        db_id = d['db_id']
        schema = tables[table_dict[db_id]]
        sql = strip_sql(d['query'])
        if is_joined_query(sql):
            res_from, res_join, res_on = parser_sql(sql)
            if any(res_on):
                fks = []
                if any(res_from) and any(res_join): # with alias
                    alias_dict = {}
                    for table, alias in res_from + res_join:
                        alias_dict[alias] = table
                    for fk_relation in res_on:
                        assert len(fk_relation) == 4
                        fks.append((alias_dict[fk_relation[0]], fk_relation[1], alias_dict[fk_relation[2]], fk_relation[3]))
                else: # without alias
                    fks = res_on
                for fk in fks:
                    tab_1, col_1, tab_2, col_2 = fk[0], fk[1], fk[2], fk[3]
                    if not (tab_1 == tab_2 and col_1 == col_2):
                        schema['foreign_keys'] = update_fks(fk, schema)
    return tables

def main():
    parser = load_parser()
    args = parser.parse_args()
    root_path = Path.cwd()
    data_path = root_path / 'data' / args.dataset
    old_tables_path = data_path / 'original'/ args.old_tables
    new_tables_path = data_path / 'original' / args.new_tables
    input_data_path = data_path / 'handmade_training_data' / args.data_file
    data, tables = load_data(input_data_path, old_tables_path)
    updated_tables = add_missing_fks(data, tables) 
    with open(new_tables_path, 'w') as f_out:
        json.dump(updated_tables, f_out, indent=2)

if __name__ == '__main__':
    main()