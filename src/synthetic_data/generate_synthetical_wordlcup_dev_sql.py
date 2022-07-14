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



def generate_synthetic_queries(db_id = 'world_cup_data_v2'):
    data_path = Path('data') / db_id
    handmade_dev_file = data_path / 'original' / 'dev.json'
    original_schema_path = data_path / 'original' / 'tables.json'
    new_schema_parent_path = data_path / 'generative'
    db_parent_path = data_path / 'original' / 'database'
    
    query_types = spider_query_types()

    with open(original_schema_path, 'r') as f_in:
        original_schemata = json.load(f_in)

    
    db_path = db_parent_path / db_id / f'{db_id}.sqlite'
    db_config = SimpleNamespace(path=db_path)
    output_json = new_schema_parent_path / 'synthetic_queries.json'
    original_schema = [os for os in original_schemata if os['db_id'] == db_id][0]
    generative_schema_path = new_schema_parent_path / 'generative_schema.json'
    generative_schema = GenerativeSchema(generative_schema_path)
    for idx, (query_type, factor) in enumerate(query_types.items()):
        max_success = 50
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
        res = check_spider_compabilities(res, original_schema)
        _res.extend(res)
        with open(output_json, 'w') as f_out:
            json.dump(_res, f_out, indent=2)
        print(f'{str(len(_res))} results persisted')

        print( f'creating synthetic queries of "{db_id}" successfully')


def check_spider_compabilities(data, original_schema):
    schema_dict = {original_schema['db_id']: original_schema}
    samples = []
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
    generate_synthetic_queries()


if __name__ == '__main__':
    main() 