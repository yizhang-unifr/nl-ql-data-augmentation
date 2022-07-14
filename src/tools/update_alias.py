import pandas as pd
import json
import logging
from pathlib import Path
import argparse

logger = logging.getLogger(__name__)
FORMAT = "[%(filename)s:%(lineno)s:%(levelname)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)

"""
this script is applied to fill in the table / columns alias 
into 
1. original/tables.json
2. generative/generative_schema.json
"""

def load_args():
    parser = argparse.ArgumentParser(description='This command is applied to update the table and columnss alias.')
    parser.add_argument('--dataset', type=str, default='skyserver_dr16_2020_11_30')
    parser.add_argument('--alias_file', type=str, default='all_reserved_columns.xlsx')
    parser.add_argument('--schema_type', nargs='?', default='original', const='original', choices=['original', 'generative'])
    parser.add_argument('--schema_file', type=str, required=False)
    parser.add_argument('--output_file', type=str, required=False)
    return parser.parse_args()

def load_alias_data(file_path):
    ext = file_path.parts[-1].split('.')[1]
    print(ext)
    if ext == 'json':
        return load_json_data(file_path)
    if ext == ('xlsx'):
        return load_xlsx_data(file_path)
    else:
        raise ValueError("Alias File: Unsupported file format")

def load_json_data(alias_path):
    pass

def load_schema_data(schema_path):
    with open(schema_path, 'r') as f_in:
        schema_data = json.load(f_in)
    return schema_data

def load_xlsx_data(alias_path):
    column_alias_df = pd.read_excel(alias_path, sheet_name='column_alias')
    table_alias_df = pd.read_excel(alias_path, sheet_name='table_alias')
    table_alias_list = table_alias_df.to_dict(orient='records')
    column_alias_list = column_alias_df.to_dict(orient='records')
    table_alias = {item['table name']: item['table alias'] for item in table_alias_list}
    column_alias = {}
    for item in column_alias_list:
        column_alias[item['table name']] = column_alias.get(item['table name'], {})
        column_alias[item['table name']][item['column name']] = item['column alias']
    return {'table_alias': table_alias, 'column_alias': column_alias}

def get_schema_path(dataset, schema_type, schema_file=None):
    if schema_type == 'original':
        if not schema_file:
            schema_file = 'tables.json'
    elif schema_type == 'generative':
        if not schema_file:
            schema_file = 'generative_schema.json'
    else:
        raise ValueError('not supported type')
    return Path(dataset, schema_type, schema_file)

def update_alias(schema_type, alias_data, schema_data):
    if schema_type == 'original':
        return update_alias_of_original_schema(alias_data, schema_data)
    elif schema_type == 'generative':
        return update_alias_of_generative_schema(alias_data, schema_data)
    else:
        raise ValueError('not supported type')
        

def update_alias_of_original_schema(alias_data: dict, schema_data: list, db_id=None):
    if db_id:
        for i, s in enumerate(schema_data):
            if db_id == s['db_id']:
                schema = schema_data[i]
                break
    else: # default the first schema in schema_data
        schema = schema_data[0]

    # update table alias:
    table_alias = alias_data['table_alias']
    table_names = schema['table_names']
    table_names_original = schema['table_names_original']
    new_table_names = []
    t_cnt = 0
    for name, original_name in zip(table_names, table_names_original):
        try:
            new_name = table_alias[original_name]
        except KeyError as e:
            logger.warning(f'alias of table {original_name} not found in alias data')
            new_name = name
        new_table_names.append(new_name)
        if new_name != name:
            logger.info(f'table {original_name}: replaced old alias "{name}" by "{new_name}"')
            t_cnt += 1
    logger.info(f'Table alias update finished: {str(t_cnt)} table alias updated')
    
    # update column alias
    column_alias = alias_data['column_alias']
    column_names = schema['column_names']
    column_names_original = schema['column_names_original']
    new_column_names = []
    c_cnt = 0
    for column_name, column_name_original in zip(column_names, column_names_original):
        assert column_name[0] == column_name_original[0] and isinstance(column_name[0], int)
        t_id = column_name[0]
        col_name_original = column_name_original[1]
        if t_id > -1:
            try:
                new_col_name = column_alias[table_names_original[t_id]][col_name_original]
            except KeyError as e:
                logger.warning(f'alias of column {table_names_original[t_id]}.{col_name_original} not found in alias data')
                new_col_name = column_name[1]
            new_column_names.append([t_id, new_col_name])
            if new_col_name != column_name[1]:
                logger.info(f'column {table_names_original[t_id]}.{column_name_original}: replaced old alias "{column_name[1]}" by "{new_col_name}"' )
                c_cnt += 1
        else:
            new_column_names.append(column_name)
    logger.info(f'Column alias update finished: {str(c_cnt)} column alias updated')

    new_schema = schema.copy()
    new_schema['table_names'] = new_table_names
    new_schema['column_names'] = new_column_names
    return [new_schema]

def update_alias_of_generative_schema(alias_data: dict, schema_data: list): 
    """
    This function should be only called,
    when the alias data need to be changed
    after the generative_schema being created.
    
    mappiings
    "name": <table_name> / <column_name> (to update)
    "original_name": <table_name_original> / <column_name_original>, (no change)
    "logical_name": <table_name_original> / <column_name_original>, (no change)
    """
    table_alias = alias_data['table_alias']
    column_alias = alias_data['column_alias']
    new_schema = []
    for table in schema_data:
        table_name_original = table['original_name']
        
        # update table
        if table_alias.get(table['original_name'], None):
            new_table = table.copy()
            new_table['name'] = table_alias[table_name_original]
            logger.info(f'table {new_table["original_name"]}: replaced old alias {table["name"]} by new alias {new_table["name"]}')
            if new_table['logical_name'] == new_table['original_name']: # only change the default logical name
                new_table['logical_name'] = '_'.join(new_table['name'].strip().split(' '))
                logger.info(f'table {new_table["original_name"]}: replaced old logical name {table["logical_name"]} by new alias {new_table["logical_name"]}')
            new_schema.append(new_table)
        else:
            # table not listed in the alias,
            new_schema.append(table)
    
    # update columns alias
    for table in new_schema:
        columns = table['columns']
        table_name_original = table['original_name']
        new_columns = []
        for column in columns:
            column_name_original = column['original_name']
            if column_alias.get(table_name_original, None) and column_alias[table_name_original].get(column_name_original, None):
                new_column = column.copy()
                new_column['name'] = column_alias[table_name_original][column_name_original]
                logger.info(f'column {table_name_original}.{column_name_original}: replace old alias {column["name"]} by new alias {new_column["name"]}')
                if new_column['logical_name'].strip().lower() == new_column['original_name'].strip().lower():
                    new_column['logical_name'] = '_'.join(new_column['name'].strip().lower().split(' '))
                    logger.info(f'column {table_name_original}.{column_name_original}: replace old logical name {column["logical_name"]} by new logical name {new_column["logical_name"]}')
                new_columns.append(new_column)
            else:
                # table or column not listed in the alias
                new_columns.append(column)
        table['columns'] = new_columns
    return new_schema

def get_new_schema_path(dataset, schema_type, output_file=None):
    if not output_file:
        if schema_type == 'original':
            output_file = 'new_tables.json'
        elif schema_type == 'generative':
            output_file = 'new_generative_schema.json'
        else:
            raise ValueError('Unsupported schema type')
    return Path(dataset, schema_type, output_file)

def save_new_schema(new_schema, output_path):
    if not output_path.exists():
        with open(output_path, 'w') as f_out:
            json.dump(new_schema, f_out, indent=2)
            logger.info(f'New schema saved successfully in "{str(output_path)}"')
    else:
        logger.warning(f'File "{str(output_path)}" exsited. Saving file aborted')

def main():
    args = load_args()
    dataset = args.dataset
    schema_type = args.schema_type
    root_path = Path(*Path(__file__).parent.parts[:-2])
    data_path = root_path  / 'data' 
    
    alias_path = data_path / dataset / args.alias_file
    alias_data = load_alias_data(alias_path)
    
    schema_file = args.schema_file
    schema_path = data_path / get_schema_path(dataset, schema_type, schema_file)
    schema_data = load_schema_data(schema_path)
    
    new_schema = update_alias(schema_type, alias_data, schema_data)
    print(new_schema)
    output_file = args.output_file
    output_path = data_path / get_new_schema_path(dataset, schema_type, output_file)
    
    # persist the new_schema
    save_new_schema(new_schema, output_path)



if __name__ == '__main__':
    main()
