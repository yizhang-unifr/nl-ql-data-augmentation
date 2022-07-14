import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple, Union, Any, Dict
from tools.training_data_builder.training_data_builder import transform_sample
from spacy.lang.en import English

nlp = English()

random.seed(42)


from intermediate_representation.sem2sql.sem2SQL import transform, build_graph
# DO NOT remove this imports! They are use by the dynamic eval() command in to_semql()
from intermediate_representation.semQL import Sup, Sel, Order, Root, Filter, A, Op, N, C, T, V, Root1, Action
from tools.transform_generative_schema import GenerativeSchema
from tools.transform_generative_schema import transform as transform_generative_schema

from synthetic_data.sample_queries.sample_query import sample_query

from synthetic_data.common_query_types import spider_query_types


def get_dev_db_ids(handmade_dev_file: Path) -> list:
    with open(handmade_dev_file, 'r') as f_in:
        dev_data = json.load(f_in)
    res = set()
    for d in dev_data:
        db_id = d['db_id']
        res.add(db_id)
    return list(res)


def init_generative_schema_spider_dev(handmade_dev_file: Path, original_schema_path: Path, new_schema_parent_path: Path):
    db_ids = get_dev_db_ids(handmade_dev_file)

    for db_id in db_ids:
        new_schema_path = new_schema_parent_path / db_id / 'generative_schema.json'
        if not new_schema_path.parent.exists():
            new_schema_path.parent.mkdir(parents=True, exist_ok=True)
        transform_generative_schema(original_schema_path, new_schema_path, tables_of_interest = [], db_id=db_id)

def init():
    """
    this method is called for init the generative schemata
    """
    data_path = Path('data/spider')
    handmade_dev_file = data_path / 'original' / 'dev.json'
    original_schema_path = data_path / 'original' / 'tables.json'
    new_schema_parent_path = data_path / 'generative'
    
    init_generative_schema_spider_dev(handmade_dev_file, original_schema_path, new_schema_parent_path)


def generate_spider_dev_queries(db_ids = []):
    data_path = Path('data/spider')
    handmade_dev_file = data_path / 'original' / 'dev.json'
    original_schema_path = data_path / 'original' / 'tables.json'
    new_schema_parent_path = data_path / 'generative'
    db_parent_path = data_path / 'original' / 'database'

    _db_ids = get_dev_db_ids(handmade_dev_file)
    if any(db_ids):
        db_ids = [db_id for db_id in db_ids if db_id in _db_ids]
    else:
        db_ids = _db_ids
    
    query_types = spider_query_types()

    with open(original_schema_path, 'r') as f_in:
        original_schemata = json.load(f_in)

    for db_id in db_ids:
        db_path = db_parent_path / db_id / f'{db_id}.sqlite'
        db_config = SimpleNamespace(path=db_path)
        output_json = new_schema_parent_path / db_id / 'synthetic_queries.json'
        original_schema = [os for os in original_schemata if os['db_id'] == db_id][0]
        generative_schema_path = new_schema_parent_path / db_id / 'generative_schema.json'
        generative_schema = GenerativeSchema(generative_schema_path)
        for idx, (query_type, factor) in enumerate(query_types.items()):
            max_success = 10
            succeed = 0
            max_iter = 300
            attemps = 0
            max_success = max_success * factor
            sampled_queries, res = [], []
            while attemps < max_iter and succeed < max_success:
                try:
                    sampled_query, sampled_query_replaced = sample_query(query_type, original_schema, generative_schema, db_config)
                    if sampled_query not in sampled_queries:
                        sampled_queries.append(sampled_query)
                        res.append(
                            {
                            'db_id': db_id,
                            'template_id': idx,
                            'query_type': query_type, 
                            'query': sampled_query,
                            'readable_query': sampled_query_replaced
                        }
                        )
                        succeed += 1
                    else: # counted it as a failed attemp
                        attemps += 1
                except ValueError as e:
                    print(e)
                    attemps += 1
            if output_json.exists():
                with open(output_json, 'r') as f_in:
                    _res = json.load(f_in)
            else:
                _res = []
            _res.extend(res)
            with open(output_json, 'w') as f_out:
                json.dump(_res, f_out, indent=2)
            print(f'{str(len(_res))} results persisted')

        print( f'creating synthetic queries of "{db_id}" successfully')

def concat_synthetic_queries(output_json_path = None, db_ids = [], max_results = 0):
    
    data_path = Path('data/spider')
    handmade_dev_file = data_path / 'original' / 'dev.json'
    original_schema_path = data_path / 'original' / 'tables.json'

    data_path = Path('data/spider')
    new_schema_parent_path = data_path / 'generative'

    if output_json_path is None:
        output_json_path = new_schema_parent_path / 'synthetic_queries.json'
    with open(original_schema_path, 'r') as f_in:
        original_schemata = json.load(f_in)
    res = []
    _db_ids = get_dev_db_ids(handmade_dev_file)
    if any(db_ids):
        db_ids = [db_id for db_id in db_ids if db_id in _db_ids]
    else:
        db_ids = _db_ids
    for db_id in db_ids:
        synthetic_queries_file = new_schema_parent_path / db_id / 'synthetic_queries.json'
        if synthetic_queries_file.exists():
            with open(synthetic_queries_file, 'r') as f_in:
                temp_res = json.load(f_in)
                temp_res = check_spider_compabilities(temp_res, original_schemata)
                if max_results > 0 and max_results < len(temp_res):
                    temp_res = sorted(random.sample(temp_res, max_results), key=lambda x: x['template_id'])
                res.extend(temp_res)
    with open(output_json_path, 'w') as f_out:
        json.dump(res, f_out, indent=2)
        print(f'successfully generated {str(len(res))} synthetic Spider Dev data under {str(output_json_path)}')


def check_spider_compabilities(data, original_schemata):
    samples = []
    schema_dict = {}
    for schema in original_schemata:
        db_id = schema['db_id']
        schema_dict[db_id] = schema
    check_failed = 0
    for sample in data:
        sample['question'] = "Dummy question for checking sanity."
        try:
            transformed = transform_sample(sample, schema_dict, nlp.tokenizer)
            del sample['question']
            samples.append(sample)
        except Exception as e:
            print(f'Error while transforming sample: {e}')
            print(f'Sample: {sample}')
            check_failed += 1
    print(f'Sanity Check for {str(len(data))} generated samples, found {str(check_failed)} samples with failed check. Finally {str(len(samples))} samples created')
    return samples
    

def main():
    init()
    # generate_spider_dev_queries(db_ids=['car_1'])
    generate_spider_dev_queries()
    concat_synthetic_queries(max_results = 200) # set max_results to 0 to have all results

if __name__ == '__main__':
    main() 