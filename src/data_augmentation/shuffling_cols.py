import os
import json
import re
import random
from math import comb
from itertools import combinations, product, islice
from preprocessing.sql2SemQL import Parser
from tools.training_data_builder.training_data_builder import transform_sample
from spacy.lang.en import English
from manual_inference.helper import get_schema_sdss
from intermediate_representation.sem2sql.sem2SQL import transform
from intermediate_representation.semQL import Root1, Root, Sel, Sup, N, A, C, T, Filter, Order, V, Op
from tqdm import tqdm
import argparse


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--db_schema', help='The database schema files in JSON format', type=str)
    parser.add_argument(
        '--seed_path', help='The seed files in JSON format', type=str)
    parser.add_argument(
        '--max_per_seed', help='The maximal number of generated data samples by permutations per each sample seed', type=int, default=100)
    parser.add_argument('--shuffle_agg', help='Enable the permuation on aggregator, if set',
                        action='store_true', dest='shuffle_agg')
    parser.add_argument('--shuffle_math_operator', help='Enable the permuation on math operators, if set',
                        action='store_true', dest='shuffle_math_operator')
    parser.add_argument(
        '--output_path', help='The path of output result', type=str)
    return parser.parse_args()


# _, schema_dict, _ = get_schema_sdss()

# table_file = '/home/ubuntu/fraunhofer/valuenet/data/skyserver_dr16_2020_11_30/original/tables.json'
ACCOUNTABLE_TYPES = ['number']


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

def c_id2col_name_idx(c_id, t_id, d, schema):
    """
    return corresponding index of col_names
    """
    col_set = d['col_set']
    col_name = col_set[c_id]
    col_names = schema['column_names']
    try:
        res = col_names.index([t_id, col_name])
        return res
    except ValueError as e:
        print(e.__traceback__)
        return None

def col_name_idx2c_id(col_name_idx, d, schema):
    """
    return corresponding index of col_set
    """
    col_set = d['col_set']
    col_names = schema['column_names']
    t_id, col_name = col_names[col_name_idx]
    return col_set.index(col_name)


def semQL2sql(data, table, origin):
    return transform(data, table, origin)


def ast_parser(ast_toks: list) -> dict:
    regex = re.compile('(\w+)\((\d+)\)')
    combination_op_list = ['Op', 'A', 'C', 'T']
    table_set = set()
    filter_flag = False
    len_tokens = len(ast_toks)
    ops = [None]*len_tokens
    op_vals = [-1]*len_tokens  # init all vals to -1
    res = {'ops': ops, 'op_vals': op_vals}
    n_combinations = -1
    proj_pattern_tuples = []
    proj_pattern_tuple = []
    for idx, ast_tok in enumerate(ast_toks):
        tuple_start_flag = False
        # check the format
        reg_res = re.match(regex, ast_tok)
        assert reg_res != None

        op_class = eval(ast_tok)
        op = ops[idx] = type(op_class).__name__
        op_val = op_class.id_c
        # check filter_flag:
        if op == 'Filter':
            if any(proj_pattern_tuple):
                proj_pattern_tuples.append(proj_pattern_tuple)
                proj_pattern_tuple = []
            filter_flag = True
        if op == 'A':
            if any(proj_pattern_tuple):
                proj_pattern_tuples.append(proj_pattern_tuple)
                proj_pattern_tuple = []
        # add tables
        if op == 'T':
            table_set.add(op_val)
        # get number of combinations
        if op == 'N':
            assert n_combinations < 0, 'Error: Got N More than once'
            n_combinations = op_val + 1
        if op not in combination_op_list or filter_flag == True:
            op_vals[idx] = op_val
        if op in combination_op_list and filter_flag == False:
            proj_pattern_tuple.append((op, op_val))
    # in case there is no filter
    if any(proj_pattern_tuple):
        proj_pattern_tuples.append(proj_pattern_tuple)

    res['table_set'] = list(table_set)
    res['n_combinations'] = n_combinations
    res['proj_pattern_tuples'] = proj_pattern_tuples
    # 
    assert len([x for x in res['op_vals'] if x < 0]
            ) == len([x for sublist in res['proj_pattern_tuples'] for x in sublist])
    return res


def generate_vars_by_pattern(d, schema_dict, max_res=100, shuffle_agg=False, shuffle_math_operator=False):
    '''
    For a semQL only permutate the value of A/Op/C/T/C/T
    N value doesn't change
    '''
    A_VALS_COUNTER = 1
    OP_VALS_COUNTER = 1
    MAX_NUM = 100000
    pattern = d['pattern']
    # print(schema_dict.keys())
    table_set = pattern['table_set']
    n_combinations = pattern['n_combinations']
    if n_combinations > 2:
        A_VALS_COUNTER = 0
    _tables_cols = []
    tables_cols_types = []
    for table_id in table_set:
        _tables_cols.extend([(idx, table_id) for idx, col in enumerate(
            schema_dict['column_names_original']) if (col[0] == table_id)])
    # modify the tables_cols with col_set idx
    tables_cols = [(col_name_idx2c_id(col_name_idx, d, schema_dict), t_id) for col_name_idx, t_id in _tables_cols]
    pattern['tables_cols'] = tables_cols
    # to get cols_types with same index as in tables_cols
    pattern['cols_types'] = [cols_type_by_c_id_t_id(c_id, t_id, d, schema_dict) for (c_id, t_id) in tables_cols]
    
    print(f"[INFO]: generating based on pattern['query']: {pattern['query']}")
    keys = set(schema_dict['primary_keys'])
    for fk in schema_dict['foreign_keys']:
        keys.update(set(fk))
    keys = sorted(list(keys))
    pattern['keys'] = keys
    aopct_pattern = pattern['proj_pattern_tuples']
    a_vals = [aopct[0][1] for aopct in aopct_pattern]
    length = len(aopct_pattern)
    # TODO
    # 1. shuffle_agg
    # 2. shuffle_math_operator 
    # 3. impl randomness in where clause
    
    if shuffle_math_operator:
        """
        temp_res = [[a_val, get_random_math_ops(aopct_p[1][1]), x[0], x[1], x[2], x[3]]
                    for a_val, x, aopct_p in zip(a_vals, generated_ct, aopct_pattern)]
        """
        aopct_pattern = [[a_val, get_random_math_ops(aopct_p[1][1]), aopct_p[2][1], aopct_p[3][1], aopct_p[4][1], aopct_p[5][1]]
                    for a_val, aopct_p in zip(a_vals, aopct_pattern)]
    else:
        aopct_pattern = [[a_val, aopct_p[1][1], aopct_p[2][1], aopct_p[3][1], aopct_p[4][1], aopct_p[5][1]]
                    for a_val, aopct_p in zip(a_vals, aopct_pattern)]
    # multi-table: cartesian product
    if len(table_set) > 1:
        generated_cts = cartesian_product(
            pattern, aopct_pattern, n_combinations)
        if len(generated_cts) > max_res:
            generated_cts = random.sample(generated_cts, max_res)
    # only 1 table: combinations
    else:
        tables_cols_copy = tables_cols[:]
        random.shuffle(tables_cols_copy)
        pattern_gen = combinations(tables_cols_copy, n_combinations)
        n_combi = comb(len(
            tables_cols), n_combinations)
        if n_combi > MAX_NUM:
            n_combi = MAX_NUM
        print(f"***n_combi = {n_combi}***")
        sample_idx = range(n_combi)
        if max_res < n_combi:
            sample_idx = sorted(random.sample(sample_idx, max_res))
        generated_cts = iter_random_combinations(
            pattern, n_combinations, max_res)
    res = []
    for generated_ct in generated_cts:
        type_list = []
        for f in generated_ct:
            """
            if len(f) == 2: # need to change
                c, t = f
                if c in pattern['keys']:
                    type_list.append(-1)
                else:
                    c_type = pattern['cols_types'][c]
                    if c_type in ACCOUNTABLE_TYPES:
                        type_list.append(1)
                    else:
                        type_list.append(0)
            """
            assert len(f) == 4
            c, t, c_2, t_2 = f
                # assert pattern['cols_types'][c] == pattern['cols_types'][c_2] and pattern['cols_types'][c] in ACCOUNTABLE_TYPES
            if c != c_2 or t != t_2:
                assert pattern['cols_types'][tables_cols.index((c, t))] == pattern['cols_types'][tables_cols.index((c_2, t_2))] and pattern['cols_types'][tables_cols.index((c, t))] in ACCOUNTABLE_TYPES
                type_list.append(1)
            else:
                if pattern['cols_types'][tables_cols.index((c, t))] in ACCOUNTABLE_TYPES:
                    type_list.append(1)
                else:
                    type_list.append(-1)
            
        # print('***type_list***', type_list)
        if shuffle_agg and A_VALS_COUNTER > 0:
            a_vals = random_value_aggr(list(A.grammar_dict.keys(
                )), counter=A_VALS_COUNTER, length=length, type_list=type_list)
            
        temp_res = [[a_val, aopct_p[1], x[0], x[1], x[2], x[3]]
                            for a_val, x, aopct_p in zip(a_vals, generated_ct, aopct_pattern)]
        res.append([item for sublist in temp_res for item in sublist])
    return res

def get_random_math_ops(m_ops, start=0, end=4, exclusive=False):
    pool = [*range(start, end+1)]
    if exclusive:
        pool = list(set(pool)-m_ops)
    return random.sample(pool, 1)[0]


def iter_random_combinations(pattern, proj_pattern, n_combinations, n_sample):
    table_cols_dict = {}
    table_set = pattern['table_set']
    tables_cols = pattern['tables_cols']
    col_types = pattern['cols_types']
    # proj_pattern = pattern['proj_pattern_tuples']

    intermediate_res = _iter_random_combinations(
        tables_cols, n_combinations, n_sample)
    res = []
    for ele in intermediate_res:
        temp = []
        for (c, t), pat in zip(ele, proj_pattern):
            if pat[1] > 0:
                (c, t, c_2, t_2) = get_an_exclusive_random_col(
                        pattern, (c, t), k=2, types=ACCOUNTABLE_TYPES)
                temp.append((c, t, c_2, t_2))
            else:
                (c_2, t_2) = get_an_exclusive_random_col(
                    pattern, (c, t), types=None)
                temp.append((c_2, t_2, c_2, t_2))
        res.append(temp)
    return res


def _iter_random_combinations(input_list: list, n_combi: int, n_sample: int, seed=None) -> list:
    if seed:
        random.seed(seed)
    n_list = len(input_list)
    n_combinations = comb(n_list, n_combi)
    if n_sample > n_combinations:
        print(f"WARNING: sample size [{n_sample}] is bigger than size of combinations [{n_combinations}]")
        n_sample = n_combinations
    combi_iter = combinations(input_list, n_combi)
    rand_samples = sorted(random.sample(range(n_combinations), n_sample))
    res = []
    i = -1
    while any(rand_samples):
        idx = rand_samples.pop(0)
        for combi in combi_iter:
            i += 1
            if i == idx:
                res.append(combi)
                break
    assert i == idx
    assert len(res) == n_sample
    return res


def random_value_aggr(rang, counter, length, type_list=None, def_val=0):
    # init
    if type_list == None:
        type_list = [1]*length
    results = [def_val]*length
    assert len(results) == len(type_list)
    # choose pos to permute
    positions = sorted(random.sample(range(length), k=counter))
    # permute the res[pos]
    for pos in positions:
        results[pos] = random.sample(rang, k=1)[0]
    res = []
    for typ, result in zip(type_list, results):
        if typ == -1:
            result = result if result == 0 else 3  # 3 for count
        else:
            result = typ*result
        res.append(result)
    return res


def cols_type_by_c_id_t_id(c_id, t_id, d, table):
    column_names_idx = c_id2col_name_idx(c_id, t_id, d, table)
    col_types =  table['column_types']
    return col_types[column_names_idx]


def cartesian_product(pattern, proj_pattern, n_combinations, limit=1e6):
    '''
    For join multiple tables
    '''
    table_cols_dict = {}
    table_set = pattern['table_set']
    tables_cols = pattern['tables_cols']
    col_types = pattern['cols_types']
    # proj_pattern = pattern['proj_pattern_tuples']
    try:
        assert n_combinations >= len(
            table_set), f'n_combinations {n_combinations} is smaller than number of table_set {len(table_set)}'
        remains = n_combinations - len(table_set)
        # get all cartesian products
        for table_id in table_set:
            table_cols_dict[table_id] = [
                tc for tc in tables_cols if tc[1] == table_id]
        products = [*product(*table_cols_dict.values())]
        print(f"***n_combi(cartesian) = {len(products)}***")
        res = []
        
        for p in products:
            if len(res) >= limit:
                break
            start_idxs = [table_cols_dict[ele[1]].index(ele) for ele in p]
            remain_cols = []
            for start_idx, cols in zip(start_idxs, table_cols_dict.values()):
                remain_cols += cols[start_idx+1:]
            # adding the remaining combinations
            if len(remain_cols) >= remains:
                add_ons = combinations(remain_cols, remains)
                for add_on in add_ons:
                    res.append(sorted([*(p + add_on)], key=lambda x: x[1]))
        intermediate_res = [*map(list, res)]
        res = []
        for ele in intermediate_res:
            temp = []
            for (c, t), pat in zip(ele, proj_pattern):
                if pat[1] > 0:
                    [(c, t), (c_2, t_2)] = get_an_exclusive_random_col(
                            pattern, (c, t), k=2, types=ACCOUNTABLE_TYPES)
                    temp.append((c, t, c_2, t_2))
                else:
                    [(c_2, t_2)] = get_an_exclusive_random_col(
                            pattern, (c, t), types=None)
                    temp.append((c_2, t_2, c_2, t_2))
            res.append(temp)
        return res

    except AssertionError as msg:
        print(msg)


def get_an_exclusive_random_col(pattern, excepted_col, k=1, types=None, tables=None):
    if tables == None:
        tables = [excepted_col[1]]
    pool = [col for col in pattern['tables_cols']
            if col[1] in tables and col[0] != excepted_col[0]]
    if types:
        pool = [(c, t) for c, t in pattern['tables_cols']
                if pattern['cols_types'][pattern['tables_cols'].index((c, t))] in types]
    res =  random.sample(pool, k)
    # flatten res
    return res


def generate_projection_vars(data, table, max_res=100, shuffle_agg=False, shuffle_math_operator=False):
    for d in tqdm(data):
        ast = d['rule_label']
        ast_toks = ast.split(' ')
        # parser the toks
        pattern = ast_parser(ast_toks)
        pattern['query'] = d['query']
        # generate combinations
        d['pattern'] = pattern
        generated_vars = generate_vars_by_pattern(
            d, table['skyserver_dr16_2020_11_30'], max_res, shuffle_agg, shuffle_math_operator)
        
        d['generated_vars'] = generated_vars
    return data


def synthesize_sql(d, table):
    res = []
    generated_vars = d['generated_vars']
    for var_val in generated_vars:
        pattern = d['pattern']
        ops, op_vals = pattern['ops'], pattern['op_vals']
        assert len(ops) == len(op_vals)
        offset = op_vals.index(-1)
        tok = ''
        for idx, (op, op_val) in enumerate(zip(ops, op_vals)):
            if op_val == -1:
                tok += f'{op}({var_val[idx-offset]})'
            else:
                tok += f'{op}({op_val})'
        temp = semQL2sql(d, table['skyserver_dr16_2020_11_30'], tok)
        res.append(temp[0].strip())
    return res


def generate_sql(data, table):
    res = []
    for d in tqdm(data):
        res.extend(synthesize_sql(d, table))
    return res


def main():
    args = load_args()
    db_schema, seed_path = args.db_schema, args.seed_path
    output_path = args.output_path
    try:
        with open(db_schema, 'r') as db_input:
            _schema_dict = json.load(db_input)[0]
        schema_dict = {_schema_dict['db_id']: _schema_dict}
        with open(seed_path, 'r') as data_input:
            dataset = json.load(data_input)
        print("Starting generating AST...")
        data, table = sql2semQL(dataset=dataset, schema_dict=schema_dict, table_file=db_schema)
        print("Starting generating projection vars...")
        # print(semQL2sql(data[0], table['skyserver_dr16_2020_11_30'], data[0]['rule_label']))
        data = generate_projection_vars(
            data, table, max_res=args.max_per_seed, shuffle_agg=args.shuffle_agg, shuffle_math_operator=args.shuffle_math_operator)
        print("Starting generating SQL...")
        res = generate_sql(data, table)
        # print(*res, sep='\n')
        with open(output_path, 'w') as output:
            json.dump(res, output, indent=2)
    except FileExistsError as e:
        print(f"I/O ERROR: {e}")

def test():
    test_data = [{
        "question": "Find all magnitude values (u, and z) from photometric objects, which have type 6, value of magnitude g between 17 and 18 and redshift less than 0.05. ",
        "query": "select u, z from photoobj as p join specobj as s on s.bestobjid = p.objid where s.class = 'GALAXY' and p.g between 17 and 18 and s.z < 0.05",
        "db_id": "skyserver_dr16_2020_11_30"
    }]
    dir_path = os.path.dirname(__file__)
    par_path = os.path.dirname(dir_path)
    root_path = os.path.dirname(par_path)
    table_dir_path = os.path.join(root_path, 'data', 'skyserver_dr16_2020_11_30', 'original')
    table_file_name = 'tables.json'
    table_file = os.path.join(table_dir_path, table_file_name)
    data, table = load_dataSets(test_data, table_file)
    data, table = sql2semQL(data, table, table_file)
    data = generate_projection_vars(
            data, table, max_res=100, shuffle_agg=True, shuffle_math_operator=False)
    res = generate_sql(data, table)
    print(res)

if __name__ == '__main__':
    # main()
    test()

