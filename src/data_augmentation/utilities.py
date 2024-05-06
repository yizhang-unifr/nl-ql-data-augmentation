import json
import re
from types import SimpleNamespace
import psycopg2

def load_data(input_data_path, tables_path):
    with open(input_data_path, 'r') as f_in:
        data = json.load(f_in)
    with open(tables_path, 'r') as f_in:
        tables = json.load(f_in)
    return data, tables

def strip_sql(sql):
    tokens = sql.strip().split(' ')
    tokens = [token for token in tokens if token != '']
    return ' '.join(tokens)

def load_schema(database_schema_path, database_name):
    with open(database_schema_path, 'r', encoding='utf-8') as json_file:
        schemas = json.load(json_file)
        for db_schema in schemas:
            if db_schema['db_id'] == database_name:
                return db_schema

        raise ValueError(f'Schema of database "{database_name}" not found in schemas')

def parser_sql(sql):
    pattern_from = r'FROM\s(\w+)\sAS\s(\w+)'
    pattern_join = r'JOIN\s(\w+)\sAS\s(\w+)'
    pattern_on = r'ON\s(\w+)\.(\w+)\s\=\s(\w+)\.(\w+)'
    res_from = re.findall(pattern_from, sql)
    res_join = re.findall(pattern_join, sql)
    res_on = re.findall(pattern_on, sql)
    return res_from, res_join, res_on

def get_table_dict(tables):
    res = {}
    for i, table in enumerate(tables):
        res[table['db_id']] = i
    return res

def is_joined_query(sql):
    pattern_if_join = r'JOIN\s(.+)\sON\s(.+)\s'
    res = re.findall(pattern_if_join, sql)
    return any(res)

class Query(SimpleNamespace):
    def __init__(self, query):
        if isinstance(query, dict):
            assert sorted(query.keys()) == ['params', 'pattern'], 'dict must have and only have keys \'params\' and \'pattern\''
            super().__init__(**query)
        else:
            raise TypeError('please input a dict-type')

class DatabaseValueSampler:
    def __init__(self, database_name, database_schema_path, max_results):
        self.database = database_name
        self.database_schema = load_schema(database_schema_path, database_name)
        self.max_results = max_results

class DatabaseValueSamplerPostgresSQL(DatabaseValueSampler):
    def __init__(self, database_name, db_schema_information, connection_config, max_results=100):
        super().__init__(database_name, db_schema_information, max_results)
        self.db_host = connection_config['database_host']
        self.db_port = connection_config['database_port']
        self.db_user = connection_config['database_user']
        self.db_password = connection_config['database_password']
        self.db_options = f"-c search_path={connection_config['database_schema']},public"

    def _execute_query(self, query):
        conn = psycopg2.connect(database=self.database, user=self.db_user, password=self.db_password,
                                host=self.db_host, port=self.db_port, options=self.db_options)
        res = []
        try:
            cur = conn.cursor()
            cur.execute(query.pattern % query.params)
            res = cur.fetchmany(self.max_results)
        except Exception as e:
            print(e)
        conn.close()
        return res

    def values_sampling(self, col, table):
        query_dict = {
        'pattern': """SELECT DISTINCT %s FROM %s;""",
        'params': (col, table),
        }
        return [o for (o,) in self._execute_query(Query(query_dict))]


def test_cordis():
    # test Database ValueSampler
    connection_config = {
        'database_host': 'testbed.inode.igd.fraunhofer.de',
        'database_port': '18001',
        'database_user': 'postgres',
        'database_password': 'vdS83DJSQz2xQ',
        'database_schema': 'unics_cordis',
    }
    query = {
        'pattern': """SELECT DISTINCT %s FROM %s;""",
        'params': ("member_role", "project_members")
    }
    q1 = Query(query)
    res = DatabaseValueSamplerPostgresSQL('cordis_temporary', 'data/cordis/original/tables.json', connection_config).values_sampling('member_role', 'project_members')
    print(res)

if __name__ == '__main__':
    test_cordis()
    # pass

