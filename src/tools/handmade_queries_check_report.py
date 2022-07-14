from pathlib import Path
import json
import pandas as pd
import psycopg2
import sqlite3

from types import SimpleNamespace

def get_db_connection_params(dataset):
    
    params = {
        'database': 'skyserver_dr16_2020_11_30',
        'user': 'postgres',
        'password': 'vdS83DJSQz2xQ',
        'host': 'testbed.inode.igd.fraunhofer.de',
        'port': '18001',
        'options': "-c search_path=public"
    }

    dataset_dict = {
        'cordis': {
            'database': 'cordis_temporary',
            'options': '-c search_path=unics_cordis,public'
            },
        'skyserver_dr16_2020_11_30': {},
        'spider': {
            'database': 'data/spider/original/database',
            'user': None,
            'password': None,
            'host': None,
            'port': None,
            'options': None
        }
    }
    assert dataset_dict.get(dataset, None) is not None
    params.update(dataset_dict[dataset])
    return params

def load_data(data_file: Path):
    with open(data_file, 'r') as f_in:
        data = json.load(f_in)
        return data

def get_conn(db_conn_params, database=None):

    if db_conn_params.get('host', None) is None:
        db_conn_params['database'] = Path(db_conn_params['database']) / database / f'{database}.sqlite'
        db_conn_params = {k: v for k, v in db_conn_params.items() if v is not None}

        conn = sqlite3.connect(**db_conn_params)
        # this is necessary to avoid decoding errors with non-utf-8 content of the database
        # https://stackoverflow.com/questions/22751363/sqlite3-operationalerror-could-not-decode-to-utf-8-column
        conn.text_factory = lambda b: b.decode(errors='ignore')
    else:
        # conn for postgresSQL DB
        conn = psycopg2.connect(**db_conn_params)
    
    print(f'{db_conn_params} successful')

    return conn

def exe_query(conn, query):
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        res = cursor.fetchmany(10)
    except Exception as e:
        print(e)
        cursor.execute("rollback")
        res = f'Error in query execution, {e}'
    cursor.close()
    return res

def sanity_check(dataset):
    data_par_path = Path('data') / dataset / 'handmade_training_data'
    data_files = ['seed_training_data.json', 'handmade_data_dev.json']
    params = get_db_connection_params(dataset)
    output_file = data_par_path / 'sanity_check.xlsx'

    conn = get_conn(params)
    dfs = []
    for data_file in data_files:
        data_file_path = data_par_path / data_file
        data = load_data(data_file_path)
        results = []
        for d in data:
            query = d['query']
            d['exec_result'] = exe_query(conn, query)
            results.append(d)
        df = pd.DataFrame(results)
        dfs.append(df)
    if not output_file.exists():
        with pd.ExcelWriter(output_file) as writer:
            for df, data_file in zip(dfs, data_files):
                sheet_name = data_file.split('.')[0]
                df.to_excel(writer, sheet_name = sheet_name, index=False)
            writer.save()

def main():
    sanity_check('cordis')
    sanity_check('skyserver_dr16_2020_11_30')

if __name__ == '__main__':
    main()