import argparse
import json
import pandas as pd
from pathlib import Path
from data_augmentation.sql_generator_for_critic import load_datasets, sql2semQL
from intermediate_representation.semQL import Root1, Root, Sel, Sup, N, A, T, C, Op, Filter,V,Order
from collections import Counter
pd.options.mode.chained_assignment = None

def load_args():
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='skyserver_dr16_2020_11_30')
    parser.add_argument('--data_file', default='seed_training_data.json')
    parser.add_argument('--table_file', default='tables.json')
    parser.add_argument('--new_table_file', default='new_tables.json')
    parser.add_argument('--generative_schema_file', default='generative_schema.json')
    parser.add_argument('--new_generative_schema_file', default='new_generative_schema.json')
    parser.add_argument('--summary_file', default='cols_summary.xlsx')
    parser.add_argument('--max_cols', default=25)
    return parser.parse_args()

def transform_datasets_to_semql(data_file, table_file, db_id, semql_list = None):
    data, tables = load_datasets(data_file, table_file)
    data, schema = sql2semQL(data, tables[db_id])
    if semql_list is None:
        semql_list = []
    for d in data:
        semql_list.append(d)
    return semql_list, schema

def extract_attributes(semql):
    indices = []
    semql = semql.split(' ')
    for i, node in enumerate(semql):
        if isinstance(eval(node), A):
            indices.append((i, i + 6))
    return [semql[start_idx:end_idx] for start_idx, end_idx in indices]

def attributes2ct(attribute, d, schema):
    col_set = d['col_set']
    table_names = schema['table_names_original']
    column_names = schema['column_names_original']
    col_alias = d['col_names']
    c,t = attribute
    col_id = eval(c).id_c
    t_id = eval(t).id_c
    if col_id > 0:
        return '.'.join([table_names[t_id], column_names[col_alias.index([t_id, col_set[col_id]])][1]])
    else:
        return '.'.join([table_names[t_id], "*"])
    

def transform_attributes_to_columns(attributes_list, d, schema):
    res = []
    for attribute in attributes_list:
        assert isinstance(eval(attribute[1]), Op)
        if eval(attribute[1]).id_c > 0:
            c_1, t_1, c_2, t_2 = attribute[2:]
            res.append(attributes2ct((c_1, t_1), d, schema))
            res.append(attributes2ct((c_2, t_2), d, schema))
        else:
            c_1, t_1 = attribute[2], attribute[3]
            res.append(attributes2ct((c_1, t_1), d, schema))
    return res

def is_key(c_id, keys):
    return c_id in keys

def generate_dataframe_from_schema(schema, generative_alias_dict, counter_dict):
    """
    DataFrame - keys: 
        unique_name str (table_name.column_name), 
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
        assert _ca[0] == _cn[0]
        if _cn[0] > -1: # skip the '*' column
            t_id = _ca[0]
            c_id = i
            tn = tables_names[t_id]
            ta = tables_alias[t_id]
            tln = tables_logical_alias[tn]
            ca = _ca[1]
            cn = _cn[1]
            ct = columns_types[i]
            un = '.'.join([tn, cn])
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
            oc = counter_dict.get(un, 0)
            list4df.append(
                {
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

def split_df_by_tables(df):
    tables = df['table_name'].unique()
    table_dfs = [df.loc[df['table_name'] == table] for table in tables]
    res_df = []
    for table_df in table_dfs:
        temp_df = score_df(table_df).sort_values(by=['column_scores'], ascending=False)
        # print(temp_df.head(20))
        res_df.append(temp_df)
    return res_df


def load_generative_schema_dict(generative_schema_path):
    with open(generative_schema_path, 'r') as f_in:
        generative_schema = json.load(f_in)
    tables_alias = {}
    columns_alias = {}
    for table in generative_schema:
        table_name = table['original_name']
        table_logical_name = table['logical_name']
        tables_alias[table_name] = table_logical_name
        columns_alias[table_name] = {} # init for columns_alias
        columns = table['columns']
        for column in columns:
            column_name = column['original_name']
            column_logical_name = column['logical_name']
            column_logical_type = column['logical_datatype']
            columns_alias[table_name][column_name] = [column_logical_name, column_logical_type]
    return {'tables_alias': tables_alias, 'columns_alias': columns_alias}

def count_cols_references(table_col_list):
    return Counter(table_col_list)

def _score_fn(row, sum_occurance):
    if sum_occurance == 0:
        oc_score = 0
    else:
        oc_score = row['occurance'] / sum_occurance * 100
    res = oc_score \
        + row['is_key'] * 100 \
        + int(row['column_name'] != row['column_logical_name']) * 0.1 \
        + int(row['column_type'] != row['column_logical_type']) * 0.1
    return res

def score_df(table_df):
    sum_occurance = sum(table_df['occurance'])
    column_scores = table_df.apply(lambda row: _score_fn(row, sum_occurance), axis=1)
    table_df.loc[:,('column_scores')] = column_scores
    return table_df

def save_summary(res_df, output_path):
    if Path(output_path).exists():
        print('Summary existed. Skipped')
        return
    with pd.ExcelWriter(output_path) as f_out:
        for df in res_df:
            sheet_name = df['table_name'].iloc[0]
            df.to_excel(f_out, sheet_name=sheet_name, index=False)
        f_out.save()
    print("Summary generated successfully!")

def generate_reserved_cols(res_df, max_cols):
    reserved_cols = []
    for df in res_df:
        _df = df[:max_cols]
        reserved_cols.extend(_df['unique_name'].tolist())
    reserved_cols = [rc.split('.') for rc in reserved_cols]
    return reserved_cols

def save_new_generative_schema(reserved_cols, input_path, output_path):
    with open(input_path, 'r') as f_in:
        g_schema = json.load(f_in)
    for table in g_schema:
        
        if table['logical_name'] == table['original_name']:
            table['logical_name'] = '_'.join(table['name'].split(' '))
        cols = [col[1] for col in reserved_cols if col[0] == table['original_name']]
        new_columns = []
        for column in table['columns']:
            if column['original_name'] in cols:
                # @TODO check the logical_name
                if column['logical_name'] == column['original_name']:
                    column['logical_name'] = '_'.join(column['name'].split(' '))
                new_columns.append(column)
        table['columns'] = new_columns
    with open(output_path, 'w') as f_out:
        json.dump(g_schema, f_out, indent=2)
        print(f'reduce generative schema successfully. File is written to {str(output_path)}')

def keys_idx2str(pk_indices, fk_indices, table_names_original, column_names_original):
    pks = []
    for pk_idx in pk_indices:
        tab_col = column_names_original[pk_idx]
        table = table_names_original[tab_col[0]]
        col = tab_col[1]
        pks.append('.'.join([table, col]))
    
    fks = []
    for fk_idx_1, fk_idx_2 in fk_indices:
        tab_col_1, tab_col_2 = column_names_original[fk_idx_1], column_names_original[fk_idx_2]
        table_1, table_2 = table_names_original[tab_col_1[0]], table_names_original[tab_col_2[0]]
        col_1, col_2 = tab_col_1[1], tab_col_2[1]
        fks.append(['.'.join([table_1, col_1]), '.'.join([table_2, col_2])])
    
    return pks, fks

def keys_str2idx(pks, fks, table_names_original, column_names_original):
    pk_indices = []
    for pk in pks:
        table, col = pk.split('.')
        tab_col = [table_names_original.index(table), col]
        pk_idx = column_names_original.index(tab_col)
        pk_indices.append(pk_idx)
    
    fk_indices = []
    for fk in fks:
        [table_1, col_1], [table_2, col_2] = fk[0].split('.'), fk[1].split('.')
        tab_col_1, tab_col_2 = [table_names_original.index(table_1), col_1], [table_names_original.index(table_2), col_2]
        fk_idx_1, fk_idx_2 = column_names_original.index(tab_col_1), column_names_original.index(tab_col_2)
        fk_indices.append([fk_idx_1, fk_idx_2])
    return pk_indices, fk_indices

def save_new_tables(reserved_cols, input_path, output_path, db_id = None):
    with open(input_path, 'r') as f_in:
        tables = json.load(f_in)
    if db_id is None:
        table = tables[0]
        db_id = table['db_id']
    else:
        for t in tables:
            if t['db_id'] == db_id:
                table = t
                break
    # no modification in tables
    table_names_original = table['table_names_original']
    column_names_original = table['column_names_original']
    column_names = table['column_names']
    column_types = table['column_types']
    
    # need t transform the idx of cols in keys
    old_pk_indices = table['primary_keys']
    old_fk_indices = table['foreign_keys']

    pks, fks = keys_idx2str(old_pk_indices, old_fk_indices, table_names_original, column_names_original)
    new_column_names_original = [[-1, '*']]

    new_column_names = [[-1, '*']]
    new_column_types = ['text']
    for i, cno in enumerate(column_names_original[1:], start=1):
        if [table_names_original[cno[0]], cno[1]] in reserved_cols:
            new_column_names_original.append(column_names_original[i])
            new_column_names.append(column_names[i])
            new_column_types.append(column_types[i])
    assert len(new_column_names_original) == len(new_column_names) and len(new_column_names_original) == len(new_column_types)
    new_pk_indices, new_fk_indices = keys_str2idx(pks, fks, table_names_original, new_column_names_original)
    
    table['column_names_original'] = new_column_names_original
    table['column_names'] = new_column_names
    table['column_types'] = new_column_types
    table['primary_keys'] = new_pk_indices
    table['foreign_keys'] = new_fk_indices
    
    with open(output_path, 'w') as f_out:
        json.dump([table], f_out, indent=2)
        print(f'reduce original table schema "{db_id}" successfully. File is written to {str(output_path)}')


def main():
    args = load_args()
    root_path = Path(*Path(__file__).parent.parts[:-2])
    dataset = args.dataset
    data_path = root_path  / 'data'
    original_path = data_path / dataset/ 'original'
    generative_path = data_path / dataset/ 'generative'
    data_file = original_path / args.data_file
    table_file = original_path / args.table_file
    new_table_file = original_path / args.new_table_file
    generative_schema_file = generative_path / args.generative_schema_file
    new_generative_schema_file = generative_path / args.new_generative_schema_file
    summary_file = original_path / args.summary_file

    generative_alias_dict = load_generative_schema_dict(generative_schema_file)
    data, schema = transform_datasets_to_semql(data_file, table_file, db_id=dataset)
    _attributes_in_semql = [extract_attributes(d['rule_label']) for d in data]
    attributes_in_semql = [attr for attributes in _attributes_in_semql for attr in attributes]
    table_col_list = transform_attributes_to_columns(attributes_in_semql, data[0], schema)
    counter_dict = count_cols_references(table_col_list)
    df = generate_dataframe_from_schema(schema, generative_alias_dict, counter_dict)
    res_df = split_df_by_tables(df)
    
    # save the xlsx
    save_summary(res_df, summary_file)
    
    # generate the reserved cols
    reserved_cols = generate_reserved_cols(res_df, max_cols=args.max_cols)

    # save the generative schema
    save_new_generative_schema(reserved_cols, generative_schema_file, new_generative_schema_file)

    # save the original schema (be careful with keys)
    save_new_tables(reserved_cols, table_file, new_table_file)

if __name__ == '__main__':
    main()

