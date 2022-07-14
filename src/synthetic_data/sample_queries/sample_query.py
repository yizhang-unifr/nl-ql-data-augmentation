import json
import random
import re
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple, Union, Any, Dict
from time import sleep

import psycopg2
import sqlite3

random.seed(42)

from intermediate_representation.sem2sql.sem2SQL import transform, build_graph
# DO NOT remove this imports! They are use by the dynamic eval() command in to_semql()
from intermediate_representation.semQL import Sup, Sel, Order, Root, Filter, A, Op, N, C, T, V, Root1, Action
from tools.transform_generative_schema import GenerativeSchema

def get_ops(op) -> str:
    id_c = op.id_c
    return op.grammar_dict[id_c].split(' ')[1]

def to_semql(semql_st: str):
    return [eval(x) for x in semql_st.strip().split(' ')]


def filter_column_value_quadruplets(query_as_semql: List[Action]) -> List[Tuple]:
    column_value_quadruplets = []
    idx = 0

    while idx < len(query_as_semql):
        if isinstance(query_as_semql[idx], A):
            # by definition of SemQL, columns always appear in the form A, Op, C, T, C, T. We therefore just search for the A action
            current_quadruplet = (query_as_semql[idx], query_as_semql[idx + 1], query_as_semql[idx + 2], query_as_semql[idx + 3], query_as_semql[idx + 4], query_as_semql[idx + 5])

            # in some cases, the A, C, T triplet is followed by a Value V
            if len(query_as_semql) > idx + 6 and isinstance(query_as_semql[idx + 6], V):
                current_quadruplet = (*current_quadruplet, query_as_semql[idx + 6])
                idx += 7
            else:
                idx += 6

            column_value_quadruplets.append(current_quadruplet)
        else:
            idx += 1

    return column_value_quadruplets


def find_unused_tables_closest_to_used_tables(unused_tables: List[str], used_tables: List[str], original_schema: dict, generative_schema: GenerativeSchema):
    original_schema_graph = build_graph(original_schema)

    distance: Dict[str, int] = {}

    # we search for each unused table the closest used table in the schema graph
    for unused_table in unused_tables:
        distance[unused_table] = 10000
        unused_table_original = generative_schema.get_original_table_name(unused_table)

        for used_table in used_tables:
            used_table_original = generative_schema.get_original_table_name(used_table)
            try:
                hops = original_schema_graph.dijkstra(used_table_original, unused_table_original)
            except AssertionError as e:
                print(e)
                raise ValueError('Found orphan relations')

            if len(hops) < distance[unused_table]:
                distance[unused_table] = len(hops)
    try:
        # sort by distance (ASC)
        distance_sorted: Dict[str, int] = dict(sorted(distance.items(), key=lambda item: item[1]))
        min_distance = list(distance_sorted.values())[0]

        # return all tables with a minimal distance. This is the set we then sample from.
        tables_with_min_distance = [table for table, distance in distance_sorted.items() if distance == min_distance]
        return tables_with_min_distance
    except Exception as e:
        raise ValueError(e)


def sample_table(t: T, op: Op, tables: dict, generative_schema: GenerativeSchema, original_schema: dict, t1_c1=None) -> Tuple[int, str]:
    # there is a good chance that this table has been used before and is just re-mentioned
    # (e.g. multiple columns selected on the same table). In that case, don't sample a new one!
    if t.id_c in tables:
        return t.id_c, tables[t.id_c]

    if op.id_c > 0: # sample tables with colunmns contains the  
        ops = get_ops(op)
        if t1_c1: # sample the 2ed 
            unused_tables = generative_schema.get_math_operands_columns(t1_c1[0], t1_c1[1])[ops].keys()
            # we also need to remove the tables from tables
            unused_tables = [t for t in unused_tables if t['names'] not in tables.values()]
        else:
            unused_tables = generative_schema.get_math_ops_tables()
        unused_tables = [t['name'] for t in unused_tables]
    else:
        table_names = generative_schema.tables
        # only sample on tables which are not yet used
        unused_tables = list(set(table_names) - set(tables.values()))

        # if it's not the first table, we want to know only the unused_tables closest to the current graph
        if len(tables) > 0:
            unused_tables = find_unused_tables_closest_to_used_tables(unused_tables,
                                                                    list(tables.values()),
                                                                    original_schema,
                                                                    generative_schema)

    assert len(unused_tables) > 0, "we try to sample more different tables than the schema has."
    sampled_table = random.choice(unused_tables)

    return t.id_c, sampled_table

def sample_column(c: C, a: A, op: Op, columns: dict, table_value: str, generative_schema: GenerativeSchema, t1_c1=None) -> Tuple[int, str]:
    if c.id_c in columns:
        return c.id_c, columns[c.id_c]
    
    if op.id_c > 0:
        ops = get_ops(op)
        if t1_c1:
            # sample the 2ed columns
            table_value = t1_c1[0]
            column_value = t1_c1[1]
            try:
                unused_columns = generative_schema.get_schema_for_math_operands_columns(t1_c1[0], t1_c1[1])[ops][table_value]
            except ValueError:
                raise ValueError(f"we try to sample math operands of {t1_c1[0]}.{t1_c1[1]} with operators \"{ops}\", but no operands found")
        else:
            try:
                unused_columns = generative_schema.get_schema_for_math_ops_columns(table_value)[ops]
                # we need to remove the columns which in columns
                unused_columns = [c for c in unused_columns if c not in columns.values()]
            except ValueError:
                raise ValueError()
            assert len(unused_columns) > 0, f"No more columns can be found. Please check the generative schema or the template"
        sampled_column = random.choice(unused_columns)
        return c.id_c, sampled_column

    # C(-1) is a special case, meaning it is referring to the * of the table and we can't just sample a column
    if c.id_c == -1:
        return c.id_c, '*'

    column_names = generative_schema.all_columns_of_table(table_value)

    # only sample on columns which are not yet used
    unused_columns = list(set(column_names) - set(columns.values()))

    # depending on the aggregation-type (max, min, sum, avg) we need to further filter for numeric types only
    if a.id_c in [1, 2, 4, 5]:
        unused_columns = [
            column for column in unused_columns
            if generative_schema.get_schema_for_column(table_value, column)['logical_datatype'] == 'number'
        ]
    
    # C(-2) is a special case, referring to a superlative (Sup()) case. Here we can also only handle dates or numbers,
    # even though the aggregator is a A(0)
    if c.id_c <= -2:
        unused_columns = [
            column for column in unused_columns
            if generative_schema.get_schema_for_column(table_value, column)['logical_datatype'] == 'number' or
               generative_schema.get_schema_for_column(table_value, column)['logical_datatype'] == 'time'
        ]
    if c.id_c >= 100: # text columns
        unused_columns = [
            column for column in unused_columns
            if generative_schema.get_schema_for_column(table_value, column)['logical_datatype'] == 'categorical'
        ]
    if len(unused_columns) == 0:
        raise ValueError("we try to sample more different columns than there is in this table(s)")

    sampled_column = random.choice(unused_columns)

    return c.id_c, sampled_column


def sample_value(v: V, values: dict, table_value: str, column_value: str, generative_schema: GenerativeSchema, db_connection: Any, max_iter = 50):
    if v.id_c in values:
        return v.id_c, values[v.id_c]

    table_meta_information = generative_schema.get_schema_for_table(table_value)
    column_meta_information = generative_schema.get_schema_for_column(table_value, column_value)

    cursor = db_connection.cursor()
    # we select a random value from the table/column we selected before.
    # Why don't we use a random selection directly on the database, e.g. by using TABLESAMPLE SYSTEM? -->
    # we want reproducible results, which requires us to set the seed for any random component
    query = f"SELECT DISTINCT {table_meta_information['original_name']}.{column_meta_information['original_name']} " \
            f"FROM {table_meta_information['original_name']} "

    if column_meta_information['original_datatype'] == 'text':
        query += f"WHERE {table_meta_information['original_name']}.{column_meta_information['original_name']} <> '' "

    query += f"LIMIT 1000"
    try:
        cursor.execute(query)
    except Exception as e:
        raise ValueError(f'Got error\n {e}\nin value-finding query: {query}') 

    all_values = cursor.fetchall()
    all_values = list(set(all_values) - set(values.values()))
    
    if len(all_values) == 0:
        raise ValueError("we try to sample a value, but didn't get any value for this column which is not null")

    sampled_value = random.choice(all_values)
    
    return v.id_c, sampled_value[0]

def sample_value_with_ops(op:Op, v: V, values: dict, table_value: str, column_value: str, table_value_2: str, column_value_2: str, generative_schema, original_schema, db_connection: Any):
    if v.id_c in values:
        return v.id_c, values[v.id_c]
    
    columns = {}
    tables = {}
    values = {}
    tables[0] = table_value
    columns[0] = column_value
    if table_value != table_value_2:
        tables[1] = table_value_2
    if column_value != column_value_2:
        columns[1] = column_value_2

    query_with_sampled_elements = assemble_spider_query_structure(columns, tables, values)

    op_idc = op.id_c

    c2_idc = len(columns) - 1
    t2_idc = len(tables) - 1

    query_type = f"Root1(3) Root(3) Sel(0) N(0) A(0) Op({str(op_idc)}) C(0) T(0) C({str(c2_idc)}) T({str(t2_idc)})"

    # print(query_type)

    query = transform(query_with_sampled_elements, original_schema, query_type, readable_alias=False)[0].strip()
    query += f" LIMIT 1000"
    
    # print("query to sample: ", query)
    
    cursor = db_connection.cursor()
    cursor.execute(query)

    all_values = cursor.fetchall()

    if len(all_values) == 0:
        raise ValueError("we try to sample a value, but didn't get any value for this column which is not null")

    sampled_value = random.choice(all_values)

    return v.id_c, sampled_value[0]

    
    
def sort_quadruplets(column_value_quadruplets: List):
    # sort by op.id_c in reverse
    return sorted(column_value_quadruplets, key=lambda x: x[1].id_c, reverse=True)

def resolve_quadruplet(column_value_quadruplet: Tuple,
                       columns: dict,
                       tables: dict,
                       values: dict,
                       original_schema: dict,
                       generative_schema: GenerativeSchema,
                       db_connection: Any) -> Union[Tuple, Tuple, Tuple, Tuple, Tuple, Tuple]:

    a: A = column_value_quadruplet[0]
    op: Op = column_value_quadruplet[1]
    c: C = column_value_quadruplet[2]
    t: T = column_value_quadruplet[3]
    c2: C = column_value_quadruplet[4]
    t2: T = column_value_quadruplet[5]

    table_key, table_value = sample_table(t, op, tables, generative_schema, original_schema)
    column_key, column_value = sample_column(c, a, op, columns, table_value, generative_schema)
    table_key_2, table_value_2 = None, None
    column_key_2, column_value_2 = None, None
    if op.id_c > 0:
        # @TODO need to add more info in generative_schema for meaningful sampling columns with math operations
        if t2.id_c == table_key:
            table_key_2, table_value_2 = table_key, table_value
        else:
            table_key_2, table_value_2 = sample_table(t2, op, tables, generative_schema, original_schema)
        t1_c1 = [table_value, column_value]
        column_key_2, column_value_2 = sample_column(c2, a, op, columns, table_value_2, generative_schema, t1_c1)
    if len(column_value_quadruplet) > 6:
        v: V = column_value_quadruplet[6]
        if op.id_c > 0:
            value_key, value_value = sample_value_with_ops(op, v, values, table_value, column_value, table_value_2, column_value_2, generative_schema, original_schema, db_connection)
        else:
            value_key, value_value = sample_value(v, values, table_value, column_value, generative_schema, db_connection)
        return (column_key, column_value), (table_key, table_value), (column_key_2, column_value_2), (table_key_2, table_value_2), (value_key, value_value)
    else:
        return (column_key, column_value), (table_key, table_value), (column_key_2, column_value_2), (table_key_2, table_value_2),()


def assemble_spider_query_structure(columns, tables, values):
    query = {
        'col_set': {}, # change the logic here now all column must be unique even in different tables
        # 'col_set': [None] * len(columns.keys()),
        'table_names': [None] * len(tables.keys()),
        'values': [None] * len(values.keys())
    }

    for idx, column_name in columns.items():
        query['col_set'][idx] = column_name

    for idx, table_name in tables.items():
        query['table_names'][idx] = table_name

    for idx, value in values.items():
        query['values'][idx] = value

    return query


def replace_logic_names(sampled_query: str, tables: List[str], columns: List[str], generative_schema: GenerativeSchema):
    """
    Replace original table and column names with the "logic" names from the generative schema.
    """
    sampled_query_replaced = sampled_query
    # @TODO There is a bug in the code below. Table.column must be unique even thouth the different columns in different tables share the same column name
    # need to first build a table_alias_columns:  dict 
    """    
    for table in tables:
        try:
            original_table_name = generative_schema.get_original_table_name(table)
            logical_table_name = generative_schema.get_logical_table_name(table)

            sampled_query_replaced = sampled_query_replaced.replace(original_table_name, logical_table_name)

            # there is a good chance that we find the wrong column if we search in all columns - because of duplicates.
            # by focusing on a table it is still possible to find wrong columns, but far less likely
            table_schema = generative_schema.get_schema_for_table(table)
            
            for column in table_schema['columns']:
                try:
                    if column['name'] in columns:
                        sampled_query_replaced = sampled_query_replaced.replace(f".{column['original_name']}", f".{column['logical_name']}")
                except Exception as e:
                    continue
        except Exception as e:
            continue
    """
    pattern = r'(\w+)\sAS\s(\w+)'
    found_table_alias = re.findall(pattern, sampled_query, flags=re.IGNORECASE)
    found_table_alias_dict = {}
    _ = [found_table_alias_dict.update({t:a}) for t, a in found_table_alias]
    
    table_alias_original_column_to_logical_table_column = {}
    for table in tables:
        try:
            _columns = generative_schema.get_schema_for_table(table)['columns']
            original_table_name = generative_schema.get_original_table_name(table)
            if found_table_alias_dict.get(original_table_name, None):
                table_alias = found_table_alias_dict[original_table_name]
                for column in _columns:
                    if column['name'] in columns:
                        original_column_name = column['original_name']
                        logical_column_name = column['logical_name']
                        table_alias_column = '.'.join([table_alias, original_column_name])
                        logical_table_column = '.'.join([table_alias, logical_column_name])
                        table_alias_original_column_to_logical_table_column.update({table_alias_column: logical_table_column})
        except Exception as e:
            continue
    keys = table_alias_original_column_to_logical_table_column.keys()
    sorted_keys = sorted(keys, key=lambda x: len(x.split('.')[1]), reverse=True)
    if sorted_keys:
        regex = re.compile("|".join(map(re.escape, sorted_keys)))
        sampled_query_replaced = regex.sub(lambda match: table_alias_original_column_to_logical_table_column[match.group(0)], sampled_query_replaced)
    # for tac, lac in table_alias_original_column_to_logical_table_column:
    #    pattern_tac = re.compile(tac)
    #    sampled_query_replaced = re.sub(pattern_tac, lac, sampled_query_replaced)
    return sampled_query_replaced





def sample_query(query_type: str, original_schema: dict, generative_schema: GenerativeSchema, db_connection: SimpleNamespace) -> Tuple[str, str]:
    # conn for SQLite DB
    if db_connection.path:
        conn = sqlite3.connect(db_connection.path)
        # this is necessary to avoid decoding errors with non-utf-8 content of the database
        # https://stackoverflow.com/questions/22751363/sqlite3-operationalerror-could-not-decode-to-utf-8-column
        conn.text_factory = lambda b: b.decode(errors='ignore')

    else:
        # conn for postgresSQL DB
        connected = False
        max_tries = 50
        sleep_time = 3
        tries = 0
        while not connected:
            try:
                if tries > max_tries:
                    raise ValueError("Cannot connect to DB")
                conn = psycopg2.connect(database=db_connection.database,
                                        user=db_connection.db_user,
                                        password=db_connection.db_password,
                                        host=db_connection.db_host,
                                        port=db_connection.db_port,
                                        options=db_connection.db_options,
                                        connect_timeout=10)
                connected = True
            except psycopg2.DatabaseError as e:
                print(e, sleep_time)
                sleep(sleep_time)
                sleep_time += sleep_time
                tries += 1

    semql_structure = to_semql(query_type)

    column_value_quadruplets = filter_column_value_quadruplets(semql_structure)

    columns = {}
    tables = {}
    values = {}

    # @TODO we need to sort the column_value_quadruplets
    column_value_quadruplets = sort_quadruplets(column_value_quadruplets)
    for column_value_quadruplet in column_value_quadruplets:
        new_columns, new_tables, new_columns_2, new_tables_2, new_values = resolve_quadruplet(column_value_quadruplet,
                                                                 columns,
                                                                 tables,
                                                                 values,
                                                                 original_schema,
                                                                 generative_schema,
                                                                 conn)
        assert len(new_tables) == len(new_columns) and len(new_columns_2 ) ==  len(new_columns_2)
        
        columns[new_columns[0]] = new_columns[1]
        tables[new_tables[0]] = new_tables[1]
        if any(new_tables_2):
            columns[new_columns_2[0]] = new_columns_2[1]
            tables[new_tables_2[0]] = new_tables_2[1]
            
        if new_values:
            values[new_values[0]] = new_values[1]

    query_with_sampled_elements = assemble_spider_query_structure(columns, tables, values)
    
    original_query = transform(query_with_sampled_elements, original_schema, query_type, readable_alias=False)[0].strip()
    # Here we sample query with readable alias instead of numerical alias, i.e. T1, T2, ...
    transformed_sql_query = transform(query_with_sampled_elements, original_schema, query_type, readable_alias=True)[0].strip()

    sampled_query_replaced = replace_logic_names(transformed_sql_query, list(tables.values()), list(columns.values()), generative_schema)
    conn.close()
    return original_query, sampled_query_replaced
