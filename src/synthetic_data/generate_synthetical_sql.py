import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple, Union, Any, Dict

random.seed(42)

from intermediate_representation.sem2sql.sem2SQL import transform, build_graph
# DO NOT remove this imports! They are use by the dynamic eval() command in to_semql()
from intermediate_representation.semQL import Sup, Sel, Order, Root, Filter, A, Op, N, C, T, V, Root1, Action
from tools.transform_generative_schema import GenerativeSchema

from synthetic_data.sample_queries.sample_query import sample_query

from synthetic_data.specific_query_types import sdss_query_types, cordis_query_types, oncomx_query_types

from generate_synthetical_spider_dev_sql import check_spider_compabilities

import argparse

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="oncomx")
    parser.add_argument('--database', type=str, default="oncomx_v1_0_25_small")
    parser.add_argument('--db_user', type=str, default="postgres")
    parser.add_argument('--db_password', type=str, default="vdS83DJSQz2xQ")
    parser.add_argument('--db_host', type=str, default="testbed.inode.igd.fraunhofer.de")
    parser.add_argument('--db_port', type=int, default="18001")
    parser.add_argument('--db_schema', type=str, default="oncomx_v1_0_25")
    return parser.parse_args()

def get_type_queries(dataset):
    queries_type_dict = {
        'skyserver_dr16_2020_11_30': sdss_query_types,
        'cordis': cordis_query_types,
        'oncomx': oncomx_query_types
    }
    if dataset in queries_type_dict.keys():
        return queries_type_dict[dataset]()

def main():
    random.seed(42)
    args = load_args()
    dataset = args.dataset
    with open(Path('data')/ dataset / 'original' / 'tables.json') as f:
        original_schemata = json.load(f)
        original_schema = original_schemata[0]

    generative_schema = GenerativeSchema(Path('data') / dataset / 'generative' / 'generative_schema.json')


    db_config = SimpleNamespace(database=args.database,
                                db_user=args.db_user,
                                db_password=args.db_password,
                                db_host=args.db_host,
                                db_port=args.db_port,
                                db_options=f"-c search_path={args.db_schema}",
                                path=None)

    """
    query_types = ['Root1(3) Root(3) Sel(0) N(1) A(0) Op(0) C(0) T(0) C(0) T(0) A(0) Op(1) C(1) T(0) C(2) T(0) Filter(5) A(0) Op(1) C(1) T(0) C(2) T(0) V(0)',
                   'Root1(3) Root(3) Sel(0) N(1) A(0) Op(0) C(0) T(0) C(0) T(0) A(0) Op(1) C(0) T(0) C(1) T(0) Filter(5) A(0) Op(1) C(1) T(0) C(2) T(0) V(0)',
                   'Root1(3) Root(3) Sel(0) N(1) A(0) Op(0) C(0) T(0) C(0) T(0) A(0) Op(1) C(1) T(0) C(2) T(0) Filter(0) Filter(5) A(0) Op(1) C(1) T(0) C(2) T(0) V(0) Filter(5) A(0) Op(1) C(2) T(0) C(3) T(0) V(1)',
                   'Root1(3) Root(3) Sel(0) N(1) A(0) Op(0) C(0) T(0) C(0) T(0) A(0) Op(1) C(1) T(0) C(2) T(0) Filter(0) Filter(5) A(0) Op(1) C(2) T(0) C(3) T(0) V(0) Filter(4) A(0) Op(1) C(3) T(0) C(4) T(0) V(1)',
                   "Root1(3) Root(3) Sel(0) N(3) A(0) Op(0) C(0) T(0) C(0) T(0) A(0) Op(0) C(1) T(0) C(1) T(0) A(0) Op(0) C(2) T(0) C(2) T(0) A(0) Op(0) C(-2) T(0) C(-2) T(0) Filter(0) Filter(2) A(0) Op(0) C(100) T(0) C(100) T(0) V(0) Filter(0) Filter(5) A(0) Op(0) C(-2) T(0) C(-2) T(0) V(1) Filter(2) A(0) Op(0) C(-3) T(0) C(-3) T(0) V(2)",
                   'Root1(3) Root(3) Sel(0) N(3) A(0) Op(0) C(0) T(0) C(0) T(0) A(0) Op(0) C(-2) T(0) C(-2) T(0) A(0) Op(0) C(-3) T(0) C(-3) T(0) A(0) Op(0) C(-4) T(0) C(-4) T(0) Filter(0) Filter(2) A(0) Op(0) C(100) T(0) C(100) T(0) V(0) Filter(0) Filter(5) A(0) Op(0) C(-4) T(0) C(-4) T(0) V(1) Filter(4) A(0) Op(0) C(-4) T(0) C(-4) T(0) V(2)',
                   'Root1(3) Root(3) Sel(0) N(1) A(0) Op(0) C(0) T(0) C(0) T(0) A(0) Op(0) C(1) T(1) C(1) T(1) Filter(0) Filter(2) A(0) Op(0) C(100) T(1) C(100) T(1) V(0) Filter(0) Filter(4) A(0) Op(1) C(2) T(0) C(3) T(0) V(1) Filter(5) A(0) Op(1) C(2) T(0) C(3) T(0) V(2)' 
             ]
    """
    query_types = get_type_queries(dataset)
    output_json = Path('data') / dataset / 'generative' / 'syntetic_queries.json'
    for idx, (query_type, factor) in enumerate(list(query_types.items())):
        if idx < 13:
            continue
        max_success = 150
        succeed = 0
        max_iter = 300
        attemps = 0
        max_success = max_success * factor
        sampled_queies, res = [], []
        while attemps < max_iter and succeed < max_success:
            try:
                sampled_query, sampled_query_replaced  = sample_query(query_type, original_schema, generative_schema, db_config)
                succeed += 1
                if sampled_query not in sampled_queies:
                    sampled_queies.append(sampled_query)
                    res.append(
                        {
                            'db_id': db_config.database,
                            'template_id': idx,
                            'query_type': query_type, 
                            'query': sampled_query,
                            'readable_query': sampled_query_replaced
                        }
                    )
                # print(sampled_query, '\n', sampled_query_replaced)
            except ValueError as e:
                print(e)
                attemps += 1
            # Sanity check for spider compabilities
        res = check_spider_compabilities(res, original_schemata)
        if output_json.exists():
            with open(output_json, 'r') as f_in:
                _res = json.load(f_in)
        else:
            _res = []
        _res.extend(res)
        with open(output_json, 'w') as f_out:
            json.dump(_res, f_out, indent=2)
        print(f'{str(len(_res))} results persisted')

    print('task finished successfully')

if __name__ == '__main__':
    main() 