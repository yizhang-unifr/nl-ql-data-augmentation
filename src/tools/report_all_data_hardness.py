from pathlib import Path
import json
import pandas as pd
from spider.test_suite_eval.evaluation import Evaluator
from tools.training_data_builder.training_data_builder import transform_sample
from spacy.lang.en import English
from tqdm import tqdm
from synthetic_data.convert_readable_handmade_queries import load_dataSets, add_dummy_question
from spacy.lang.en import English

# prepocess spider if necessary
def build_datasets():
    databases = ['Spider', 'WorldCup', 'Cordis', 'SDSS', 'OncoMX']
    data_files = {
        'Cordis': {
            'Seed': 'cordis/original/seed_training_data.json',
            'Dev': 'cordis/original/dev.json',
            'Reshuffle_Seed': 'cordis/shuffle_seed.json',
            'Reshuffle_Dev': 'cordis/shuffle_dev.json',
            'Synth_Ursin': 'cordis/original/all_synthetic.json',
            'Synth_Yi': 'cordis/generative_data_train.json',
            'schema': 'cordis/original/tables.json'
            },
        'WorldCup' : {
            'Synth': 'world_cup_data_v2/generative_data_train.json',
            'Dev': 'world_cup_data_v2/dev.json',
            'schema': 'world_cup_data_v2/original/tables.json'
        },
        'Spider': {
            'Train': 'spider/train.json',
            'Dev': 'spider/original/dev.json',
            'Synth': 'spider/original/generative_data_train.json',
            'schema': 'spider/original/tables.json'
        },
        'SDSS': {
            'Seed': 'skyserver_dr16_2020_11_30/original/seed_training_data.json',
            'Dev': 'skyserver_dr16_2020_11_30/original/dev.json',
            'Reshuffle_Deed': 'skyserver_dr16_2020_11_30/shuffle_seed.json',
            'Reshuffle_Dev': 'skyserver_dr16_2020_11_30/shuffle_dev.json',
            'Synth_Yi': 'skyserver_dr16_2020_11_30/generative/synthetic_queries.json',
            'schema': 'skyserver_dr16_2020_11_30/original/tables.json'
            },
        'OncoMX': {
            'Seed': 'oncomx/original/handmade_data_seed.json',
            'Dev': 'oncomx/original/dev.json',
            'Synth_Ursin': 'oncomx/original/all_synthetic.json',
            'Synth_Yi': 'oncomx/generative_data_train.json',
            'schema': 'oncomx/original/tables.json'
            },
        }
    
    return databases, data_files 

def process(dataset, data_path, schema_path):
    _data_path = Path('data')
    schema_path = _data_path  / schema_path
    data_path = _data_path / data_path

    with open(data_path, 'r') as f_in:
        data = json.load(f_in)
    if data[0].get('sql', None):
        return data
    with open(schema_path, 'r') as f_in:
        schemata = json.load(f_in)
    schema_dict = {}
    for schema in schemata:
        schema_dict[schema['db_id']] = schema
    dataset = add_dummy_question(data)
    # data, table = sql2semQL(data, schema_dict, original_schema_path)
    nlp = English()
    processed = [transform_sample(
        ds, schema_dict, nlp.tokenizer) for ds in dataset]
    data, table = load_dataSets(processed, schema_path)
    return data

def retreive_sql_category(sql):
    evaluator = Evaluator()
    return evaluator.eval_hardness(sql)

def eval_spider_hardness(data):
    for d in data:
        d['hardness'] = retreive_sql_category(d['sql'])
    return data

def analyze_df(df, database, key):
    res = {}
    levels = ['easy', 'medium', 'hard', 'extra']
    res['database'] = database
    res['dataset'] = key
    res['total'] = len(df)
    for l in levels:
        _filter = df['hardness'] == l
        res[l] = len(df[_filter])
    return res


def generate_summary():
    databases, data_files = build_datasets()
    df_dict = {}
    overview = []
    for database in databases:
        files = data_files[database]
        schema_path = files['schema']
        df_dict[database]=[]
        for key in files.keys():
            if key != 'schema' and len(files[key]) > 0:
                data_path = files[key]
                data = eval_spider_hardness(process(database, data_path, schema_path))
                temp = []
                for d in data:
                    temp.append(
                        {   'database': database,
                            'dataset': key,
                            'db_id': d['db_id'],
                            'question': d['question'],
                            'query': d['query'],
                            'hardness': d['hardness']}
                    )
                _df = pd.DataFrame(temp)
                analysis_res = analyze_df(_df, database, key)
                overview.append(analysis_res)
                df_dict[database].append(_df)
    overview = pd.DataFrame(overview)
    # print(df_dict, overview)
    # return df_dict, overview
    
    # persistance data in xlsx format
    _data_path = Path('data')
    output_path = _data_path / 'spider_hardness_summary.xlsx'
    
    if not output_path.exists():
        with pd.ExcelWriter(output_path) as f_out:
            for key in df_dict.keys():
                df_list = df_dict[key]
                temp_df = pd.concat(df_list, sort=False)
                temp_df.to_excel(f_out, sheet_name=key, index=False)
            overview.to_excel(f_out, sheet_name='overview', index=False)
            f_out.save()


def eval_single_sql_spider_hardness(database, sql):
    databases, data_files = build_datasets()
    if database in databases:
        schema_path = data_files[database]['schema']


if __name__ == '__main__':
    generate_summary()

