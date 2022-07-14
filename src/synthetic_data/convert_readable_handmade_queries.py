from pathlib import Path
import argparse
import json
from synthetic_data.sample_queries.sample_query import transform, replace_logic_names, GenerativeSchema
from preprocessing.sql2SemQL import Parser
from tools.training_data_builder.training_data_builder import transform_sample
from spacy.lang.en import English
from tqdm import tqdm
#'--schema_type', nargs='?', default='original', const='original', choices=['original', 'generative']

def load_args():
    args = argparse.ArgumentParser(description='')
    args.add_argument('--dataset', nargs= '?', default = 'skyserver_dr16_2020_11_30', \
        const = 'skyserver_dr16_2020_11_30', choices = ['cordis', 'skyserver_dr16_2020_11_30'])
    args.add_argument('--input_file_path', type=str)    
    args.add_argument('--output_file_path',type=str)
    args.add_argument('--original_schema_path', type=str)
    args.add_argument('--generative_schema_path', type=str)
    return args.parse_args()

def transform_data(input_data_with_semql, original_schema, generative_schema):
    res = []
    for d in input_data_with_semql:
        print(d['question'], '\n', d['query'])
        transformed_query = transform(d, original_schema, origin=d['rule_label'], readable_alias=True)[0]
        columns = d['names']
        tables = d['table_names']
        readable_query = replace_logic_names(transformed_query, tables, columns, generative_schema)
        
        temp = {
                'db_id': d['db_id'],
                'question': d['question'],
                'query': d['query'],
                'readable_query': readable_query
        }
        res.append(temp)
    return res

def transform_file(input_file_path: Path, original_schema_file: Path, generative_schema_path: Path, output_file: Path):
    with open(input_file_path, 'r') as f_in:
        input_data_with_semql = json.load(f_in)
    with open(original_schema_file, 'r') as f_in:
        original_schema = json.load(f_in)[0] # we assume here only 
    
    generative_schema = GenerativeSchema(generative_schema_path) # we assume here only
    output_data = transform_data(input_data_with_semql, original_schema, generative_schema)
    if not isinstance(output_file, Path):
        output_file = Path(output_file)
    if not output_file.exists():
        with open(output_file, 'w') as f_out:
            json.dump(output_data, f_out, indent=2)
            print(f'File {str(output_file)} generated successfully')
    else:
        print(f'File {str(output_file)} already exists. Task cancelled.')
    

def get_paths_from_args(args):
    data_path = Path('data')
    dataset = args.dataset
    
    original_schema_path = data_path / dataset / 'original' / 'tables.json'
    if args.original_schema_path:
        original_schema_path = args.original_schema_path
    
    generative_schema_path = data_path / dataset / 'generative' / 'generative_schema.json'
    if args.generative_schema_path:
        generative_schema_path = args.generative_schema_path
    
    input_file_path = data_path / dataset / 'seed_training_data.json'
    if args.input_file_path:
        input_file_path = args.input_file_path
    
    output_file_path = data_path / dataset / 'handmade_training_data' / 'readable_seed_training_data.json'
    if args.output_file_path:
        output_file_path = args.output_file_path
    
    return original_schema_path, generative_schema_path, input_file_path, output_file_path


def main():
    args = load_args()
    original_schema_path, generative_schema_path, input_file_path, output_file_path = get_paths_from_args(args)
    transform_file(input_file_path, original_schema_path, generative_schema_path, output_file_path)

def test():
    data_path = Path('data')
    dataset = 'skyserver_dr16_2020_11_30'
    
    original_schema_path = data_path / dataset / 'original' / 'tables.json'
    
    generative_schema_path = data_path / dataset / 'generative' / 'generative_schema.json'

    input_file_path = data_path / dataset / 'dev.json'
    
    with open(input_file_path, 'r') as f_in:
        input_data_with_semql = json.load(f_in)
    with open(original_schema_path, 'r') as f_in:
        original_schema = json.load(f_in)[0] # we assume here only 
    
    input_data_with_semql = input_data_with_semql
    generative_schema = GenerativeSchema(generative_schema_path) # we assume here only
    output_data = transform_data(input_data_with_semql, original_schema, generative_schema)
    print(output_data)

def load_dataSets(data, table_file):
    # @todo simplify the output for performance optimization.
    with open(table_file, 'r', encoding='utf8') as f:
        table_data = json.load(f)

    output_tab = {}
    tables = {}
    tabel_name = set()
    for i in range(len(table_data)):
        table = table_data[i]
        temp = {}
        temp['col_map'] = table['column_names']
        temp['table_names'] = table['table_names']
        tmp_col = []
        for cc in [x[1] for x in table['column_names']]:
            if cc not in tmp_col:
                tmp_col.append(cc)
        table['col_set'] = tmp_col
        db_name = table['db_id']
        tabel_name.add(db_name)
        table['schema_content'] = [col[1] for col in table['column_names']]
        table['col_table'] = [col[0] for col in table['column_names']]
        output_tab[db_name] = temp
        tables[db_name] = table

    for d in data:
        d['names'] = tables[d['db_id']]['schema_content']
        d['table_names'] = tables[d['db_id']]['table_names']
        d['col_set'] = tables[d['db_id']]['col_set']
        d['col_table'] = tables[d['db_id']]['col_table']
        keys = {}
        for kv in tables[d['db_id']]['foreign_keys']:
            keys[kv[0]] = kv[1]
            keys[kv[1]] = kv[0]
        for id_k in tables[d['db_id']]['primary_keys']:
            keys[id_k] = id_k
        d['keys'] = keys
        # add cols_type_list
        # d['col_types'] = tables[d['db_id']]['column_types']

    return data, tables

def sql2semQL(dataset, schema_dict, table_file):
    nlp = English()
    processed = [transform_sample(
        ds, schema_dict, nlp.tokenizer) for ds in dataset]
    data, table = load_dataSets(processed, table_file)

    for row in tqdm(data):
        parser = Parser(build_value_list=True)
        parser.full_parse(row)
        row['values'] = parser.values
        parser = Parser(row['values'])
        semql_result = parser.full_parse(row)
        row['rule_label'] = ' '.join([str(x) for x in semql_result])
    return data, table

def add_dummy_question(data):
    _data = data[:]
    for d in _data:
        if not d.get('question', None):
            d['question'] = 'DUMMY QUESTIONS for test.'
    return _data 

def convert_readable_synthetic_data():
    # for sdss 
    data_path = Path('data')
    dataset = 'skyserver_dr16_2020_11_30'
    original_schema_path = data_path / dataset / 'original' / 'tables.json'
    generative_schema_path = data_path / dataset / 'generative' / 'generative_schema.json'

    input_file_path = data_path / dataset / 'generative' / '_syntetic_queries.json'
    output_file_path = data_path / dataset / 'generative' / 'syntetic_queries.json'
    with open(input_file_path, 'r') as f_in:
        input_data = json.load(f_in)
    with open(original_schema_path, 'r') as f_in:
        original_schema = json.load(f_in)[0] # we assume here only 
    schema_dict = {original_schema['db_id']: original_schema}
    data = add_dummy_question(input_data)
    data, table = sql2semQL(data, schema_dict, original_schema_path)

    generative_schema = GenerativeSchema(generative_schema_path) # we assume here only
    transformed_data = transform_data(data, original_schema, generative_schema)
    output_data = []
    for d, _d in zip(input_data, transformed_data):
        assert d['query'] == _d['query']
        temp = {
            'db_id': d['db_id'],
            'template_id': d['template_id'],
            'query_type': d['query_type'],
            'query': d['query'],
            'readable_query': _d['readable_query']
        }
        output_data.append(temp)
    assert len(output_data) == len(input_data)

    if not output_file_path.exists():
        with open(output_file_path, 'w') as f_out:
            json.dump(output_data, f_out, indent=2)
            print(f'File {str(output_file_path)} generated successfully')
    else:
        print(f'File {str(output_file_path)} already exists. Task cancelled.')


if __name__ == '__main__':
    # main()
    convert_readable_synthetic_data()
    # test()



