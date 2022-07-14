from pathlib import Path
import json
import argparse
from tools.summarise_and_reduce_schemata import extract_attributes, transform_attributes_to_columns, count_cols_references, is_key
from data_augmentation.sql_generator_for_critic import load_datasets, sql2semQL
from tqdm import tqdm
from dash import Dash, html, dcc, Input, Output
import pandas as pd
import plotly.express as px
from spider.test_suite_eval.evaluation import Evaluator

def load_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument('--dataset', default='skyserver_dr16_2020_11_30')
    parser.add_argument('--training_data_file', default='seed_training_data.json')
    parser.add_argument('--dev_data_file', default='dev.json')
    parser.add_argument('--data_schema_file', default='tables.json')
    parser.add_argument('--output_xlsx_path', default='analytics.xlsx')
    return parser.parse_args()

def score_df(table_df):
    sum_occurance = sum(table_df['occurance'])
    column_frequency = table_df.apply(lambda row: row['occurance']/ sum_occurance if sum_occurance > 0 else 0, axis=1)
    table_df.loc[:,('frequency')] = column_frequency
    return table_df

def generate_dataframe_from_schema(db_id, schema, generative_alias_dict, counter_dict):
    """
    DataFrame - keys: 
        unique_name str (db_id.table_name.column_name),
        db_id: str,
        table_name: str, 
        table_alias: str, 
        table_logical_name: str, 
        column_name: str, 
        column_alias: str, 
        column_logical_name: str, 
        is_key: int (0/1)
        type: str
        logical_type: str
        occurance_in_dataset: int
    """
    if generative_alias_dict is None:
        tables_logical_alias = ['_'.join(name.split(' ')) for name in schema['table_names']]
        columns_logical_alias = None
    else:
        tables_logical_alias = generative_alias_dict['tables_alias']
        columns_logical_alias = generative_alias_dict['columns_alias']
    tables_alias = schema['table_names']
    tables_names = schema['table_names_original']
    columns_alias = schema['column_names']
    columns_names = schema['column_names_original']
    columns_types = schema['column_types']
    keys = schema['foreign_keys']+[schema['primary_keys']]
    keys = set([k for key_pair in keys for k in key_pair])
    list4df = []
    for (i, (_ca, _cn)) in enumerate(zip(columns_alias, columns_names)):
        assert _ca[0] == _cn[0], f"_ca: {_ca}, _cn: {_cn}"
        if _cn[0] > -1: # skip the '*' column
            t_id = _ca[0]
            c_id = i
            tn = tables_names[t_id]
            ta = tables_alias[t_id]
            if generative_alias_dict is None:
                tln = '_'.join(ta.split(' '))
            else:
                tln = tables_logical_alias[tn]
            ca = _ca[1]
            cn = _cn[1]
            ct = columns_types[i]
            un = '.'.join([db_id, tn, cn])
            _un = '.'.join([tn, cn])
            cln = cn
            clt = ct
            if columns_logical_alias is not None:
                try:
                    cln = columns_logical_alias[tn][cn][0]
                    clt = columns_logical_alias[tn][cn][1]
                except KeyError as e:
                    print(f'{tn}.{cn} does not exist in the generative schema')
            if cln == cn: 
                cln = '_'.join(cn.split(' '))
            isk = int(is_key(i, keys))
            oc = counter_dict.get(_un, 0)
            list4df.append(
                {   
                    "db_id": db_id,
                    "unique_name": un, 
                    "table_name": tn, 
                    "table_alias": ta, 
                    "table_logical_name": tln, 
                    "column_name": cn, 
                    "column_alias": ca, 
                    "column_logical_name": cln, 
                    "is_key": isk,
                    "column_type": ct,
                    "column_logical_type": clt,
                    "occurance": oc
                }
            )
    df = pd.DataFrame.from_records(list4df)
    return df

def get_hardness(sql):
    evaluator = Evaluator()
    hardness = evaluator.eval_hardness(sql)
    return hardness

def summarize_db_level_data_distribution(data_file, table_file):
    data, tables = load_datasets(data_file, table_file)
    res_dict = {key: {'count': 0, 'easy': 0, 'medium': 0, 'hard': 0, 'extra': 0} for key in tables.keys()}
    for d in data:
        db_id = d["db_id"]
        res_dict[db_id]['count'] += 1
        hardness = get_hardness(d['sql'])
        res_dict[db_id][hardness] += 1
    res = []
    for k, v in res_dict.items():
        res.append({'db_id': k, 'occurance': v['count'], 'easy': v['easy'], 'medium': v['medium'], 'hard': v['hard'],'extra': v['extra']})
    return pd.DataFrame.from_records(res)

def summarize_detailed_data_distribution(data_file, table_file):
    data, tables = load_datasets(data_file, table_file)
    # table_file covers all required schemata 
    # db_id in table_file is a super set of the db_id in the data_file
    dfs = []
    db_id_data_dict = {} # store all data in a form as [{db_id: []}]
    for d in tqdm(data):
        if not db_id_data_dict.get(d['db_id'], None):
            db_id_data_dict[d['db_id']] = {'data': [], 'schema': tables[d['db_id']]}
        k =  db_id_data_dict[d['db_id']]['data']
        k.append(d)
    for db_id, d in db_id_data_dict.items():
        _data, _schema = sql2semQL(d['data'], d['schema'])
        _attributes_in_semql = [extract_attributes(_d['rule_label']) for _d in _data]
        attributes_in_semql = [attr for attributes in _attributes_in_semql for attr in attributes]
        table_col_list = transform_attributes_to_columns(attributes_in_semql, _data[0], _schema)
        counter_dict = count_cols_references(table_col_list)
        table_df = generate_dataframe_from_schema(db_id, _schema, None, counter_dict)
        score_df(table_df)
        table_df = table_df.sort_values(by='frequency', ascending=False)
        dfs.append(table_df)
    total_df = None
    for df in dfs:
        if total_df is None:
            total_df = df
        else:
            total_df = pd.concat([total_df, df])
    return total_df
    
    """
    data, schema = sql2semQL(data, tables[db_id])
    if semql_list is None:
        semql_list = []
    for d in data:
        semql_list.append(d)
    return semql_list, schema
    """

def main():
    args = load_args()
    root_path = Path.cwd()
    dataset = args.dataset
    data_path = root_path / "data" / dataset / "original"
    train_data_file = data_path / args.training_data_file
    table_file = data_path / args.data_schema_file
    dev_data_file = data_path / args.dev_data_file
    output_xls_path = data_path / args.output_xlsx_path
    
    train_res = summarize_detailed_data_distribution(train_data_file, table_file)
    train_res['dataset'] = 'train'
    dev_res = summarize_detailed_data_distribution(dev_data_file, table_file)
    dev_res['dataset'] = 'dev'
    total_res = pd.concat([train_res, dev_res]).sort_values(by=['db_id', 'frequency', 'dataset'], ascending=[True, False, False]).reindex()
    # print(dev_res)
    train_db_res = summarize_db_level_data_distribution(train_data_file, table_file)
    train_db_res['dataset'] = 'train'

    dev_db_res = summarize_db_level_data_distribution(dev_data_file, table_file)
    dev_db_res['dataset'] = 'dev'
    # print(train_db_res)
    print(train_db_res.head(20))

    with pd.ExcelWriter(output_xls_path) as f_out:
        total_res.to_excel(f_out, sheet_name='detailed_total', index=False)
        train_db_res.to_excel(f_out, sheet_name='db_train', index=False)
        dev_db_res.to_excel(f_out, sheet_name='db_dev', index=False)
        f_out.save()

if __name__ == '__main__':
    main()