import json, re, random, itertools
from datetime import datetime, timedelta
from preprocessing.sql2SemQL import Parser
from intermediate_representation.sem2sql.sem2SQL import transform
from intermediate_representation.semQL import Root1, Root, Sel, Sup, N, A, C, T, Filter, Order, V, Op
from tqdm import tqdm
import logging
import argparse
import re

logger = logging.getLogger(__name__)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--schema_path', help='The database schema files in JSON format', type=str)
    parser.add_argument(
        '--data_path', help='The seed files in JSON format', type=str)
    parser.add_argument(
        '--output_path', help='The path of output result', type=str)
    return parser.parse_args()


def load_datasets(data_file, table_file):
    with open(data_file, 'r') as in_file:
        data = json.load(in_file)
    with open(table_file, 'r') as in_file:
        table_data = json.load(in_file)
    output_tab = {}
    tables = {}
    table_name = set()
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
        table_name.add(db_name)
        table_name.add(db_name)
        table['schema_content'] = [col[1] for col in table['column_names']]
        table['col_table'] = [col[0] for col in table['column_names']]
        output_tab[db_name] = temp
        tables[db_name] = table
    for d in data:
        d['names'] = tables[d['db_id']]['schema_content']
        d['table_names'] = tables[d['db_id']]['table_names']
        d['col_set'] = tables[d['db_id']]['col_set']
        d['col_table'] = tables[d['db_id']]['col_table']
        d['col_names'] = tables[d['db_id']]['column_names']
        keys = {}
        for kv in tables[d['db_id']]['foreign_keys']:
            keys[kv[0]] = kv[1]
            keys[kv[1]] = kv[0]
        for id_k in tables[d['db_id']]['primary_keys']:
            keys[id_k] = id_k
        d['keys'] = keys
        # add cols_type_list
        d['col_types'] = tables[d['db_id']]['column_types']
    return data, tables

def sql2semQL(data, table):
    for row in tqdm(data):
        try:
            parser = Parser(build_value_list=True)
            parser.full_parse(row)
            row['values'] = parser.values
            parser = Parser(row['values'])
            semql_result = parser.full_parse(row)
            row['rule_label'] = ' '.join([str(x) for x in semql_result])
        except Exception as e:
            logging.warning(f"Unexpected {type(e)}: {e}")
    return data, table

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
            if n_combinations < 0:
                n_combinations = op_val + 1
            else:
                n_combinations += op_val + 1
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
    assert len([x for x in res['op_vals'] if x < 0]) == len([x for sublist in res['proj_pattern_tuples'] for x in sublist])
    return res

def ast_parser(d) -> dict:
    ast_toks = d['rule_label'].split(' ')
    regex = re.compile('(\w+)\((\d+)\)')
    combination_op_list = ['Op', 'A', 'C', 'T']
    table_set = set()
    filter_flag = False
    len_tokens = len(ast_toks)
    ops = [None]*len_tokens
    op_vals = [-1]*len_tokens  # init all vals to -1
    res = {'ops': ops, 'op_vals': op_vals}
    n_combinations = -1
    proj_patterns = []
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
        if op == 'Root':
            if any(proj_pattern_tuple):
                proj_pattern_tuples.append(proj_pattern_tuple)
                proj_pattern_tuple = []
                proj_patterns.append(proj_pattern_tuples)
                proj_pattern_tuples = []
            filter_flag = False
        if op in ['Filter', 'Order', 'Sup']:
            filter_flag = True
        if op == 'A' and not filter_flag:
            if any(proj_pattern_tuple):
                proj_pattern_tuples.append(proj_pattern_tuple)
                proj_pattern_tuple = []
        # add tables
        if op == 'T':
            table_set.add(op_val)
        # get number of combinations
        if op == 'N':
            if n_combinations < 0:
                n_combinations = op_val + 1
            else:
                n_combinations += op_val + 1
        
        if op not in combination_op_list or filter_flag:
            op_vals[idx] = op_val
        if op in combination_op_list and not filter_flag:
            proj_pattern_tuple.append((op, op_val))

    if any(proj_pattern_tuple):
        proj_pattern_tuples.append(proj_pattern_tuple)
        proj_patterns.append(proj_pattern_tuples)

    res['table_set'] = list(table_set)
    res['n_combinations'] = n_combinations
    res['proj_pattern_tuples'] = proj_patterns
    # assertion if proj_patterns are correctly annotated
    proj_sum = 0
    for tuples in res['proj_pattern_tuples']:
        for tup in tuples:
            proj_sum += len(tup)
    assert len([x for x in res['op_vals'] if x < 0]) == proj_sum

    # assert len([x for x in res['op_vals'] if x < 0]) == len([x for sublist in res['proj_pattern_tuples'] for x in sublist])
    d['ast'] = res
    return d

# ¨Return an AST that is slightly different from the original one
# This function shuffles the AST with very simple rules
# More complex rules can be added later 
def generate_new_ast(d):
    ast = d['ast']
    ast['new_proj_pattern_tuples'] = ast['proj_pattern_tuples'][:]
    d['old_values'] = d['values'][:]
    new_op_vals = ast['op_vals'][:] # copy the op_vals3
    # If we found any union, intersect or except
    if new_op_vals[ast['ops'].index('Root1')] < 3:
            # strategy: shuffle the Root1
        new_op_vals[ast['ops'].index('Root1')] = get_shuffle_id(0, 3, exclusive=[new_op_vals[ast['ops'].index('Root1')]])
    else:
        # if only one projection field
        if len(ast['proj_pattern_tuples']) == 1 and len(ast['proj_pattern_tuples'][0]) == 1:
            # and if no filter (means the query is very simple)
            if 'Filter' not in ast['ops']:
                # we only switch the aggregator,
                # no agg -> count or known agg -> no agg
                """
                0: 'A none Op',
                1: 'A max Op',
                2: "A min Op",
                3: "A count Op",
                4: "A sum Op",
                5: "A avg Op"
                """
                ast['new_proj_pattern_tuples'][0][0] = tuples_shuffle(ast['proj_pattern_tuples'][0][0], d, mask = [1, 0, 0])
            else:
                # we shuffle the Filter instead
                new_op_vals = filter_shuffle(ast, d)
        else:
            if 'Filter' not in ast['ops']: 
            # Only shuffle the proj_pattern_tuples in the first segment
                assert ast['proj_pattern_tuples'][0][0][0][0] == 'A'
                
                # if found an aggregator remove it
                if ast['proj_pattern_tuples'][0][0][0][1] != 3: # remove aggregator
                    ast['new_proj_pattern_tuples'][0][0] = tuples_shuffle(ast['proj_pattern_tuples'][0][0], d, mask = [1,0,0])
                else:
                    # replace by another column and try to use the same type
                    # check if "*"
                    # if so we try next one
                    tuples_candidates = []
                    for i, proj_pattern_tuples in enumerate(ast['proj_pattern_tuples']):
                        for j, pattern_tuples in enumerate(proj_pattern_tuples):
                            assert pattern_tuples[0][0] == 'A'
                            if pattern_tuples[0][1] == 0 and len(pattern_tuples) == 4 and pattern_tuples[2][1] != 0:
                                tuples_candidates.append([i, j])
                    if any(tuples_candidates):
                        i, j = tuples_candidates[0]
                        ast['new_proj_pattern_tuples'][i][j] = tuples_shuffle(ast['proj_pattern_tuples'][i][j], d, mask = [0,0,1])
                    else:
                        # check if contains Distinct, if contains just remove it
                        sels_indice = [(i, new_op_vals[i]) for i, ele in enumerate(ast['ops']) if new_op_vals[i]==1 and ast['ops'][i] == 'Sel']
                        if any(sels_indice):
                            # just change the first Distinct
                            new_op_vals[sels_indice[0][0]] = 0
                        else:
                            raise ValueError("Cannot shuffle the SQL! Need more Rules!!!")
            else:
                # we shuffle the Filter instead
                # TODO: impl filter_shuffle
                new_op_vals = filter_shuffle(ast, d)
    d['generated_ast'] = synthesize_ast(ast, new_op_vals)
    return d

# TODO need to change this tuples shuffle, since the leaf node of new AST has always a lenth of 6 (A Op C T C T)
def tuples_shuffle(_tuples: list, d: list, mask, agg_only=False):
    """
    input is a list of tuples len = 4 or 6
    mask is a list with lengeh of 3 to represent the 
    possible shuffling part in [A, Op, CT] respectively
    the digits in mask can be either 1 or 0
    1 means shuffling and 0 for keeping
    shuffle order : C -> Op -> A  
    """
    # copy _tuples
    tuples = _tuples[:]
    if mask[2]:
        if len(tuples) == 4:
            assert tuples[2][0] == 'C'
            c_id = tuples[2][1]
            assert tuples[3][0] == 'T'
            t_id = tuples[3][1]
            new_c_id = get_random_col(c_id, t_id, d, exclusive=[c_id])
            if new_c_id < 0:
                # when encounter an "*", we first check A
                assert tuples[0][0] == 'A'
                if tuples[0][1]==0: # no aggr
                    t_id = tuples[3][1]
                    # TODO
                    #  need to change! 
                    # use get_random_col
                    col_pool = [i for i, ele in filter(lambda ele: ele[1] == t_id, enumerate(d['col_table']))]
                    new_c_id = random.sample(col_pool,k=1)[0]
                else: 
                    # just remove the aggr
                    # and shut down the mask for A
                    mask[0] = 0
                    tuples[0] = ('A', 0)
            tuples[2] = ('C', new_c_id)
        elif len(tuples) == 6:
            assert tuples[2][0] == 'C' and tuples[4][0] == 'C'
            c1_id, c2_id = tuples[2][1], tuples[4][1]
            assert tuples[3][0] == 'T' and tuples[5][0] == 'T'
            t1_id, t2_id = tuples[3][1], tuples[5][1]
            new_c1_id = get_random_col(c1_id, t1_id, d, exclusive=[c1_id], same_type=True)
            new_c2_id = get_random_col(c2_id, t2_id, d, exclusive=[c2_id], same_type=True)
            # No need to consider * case here
            tuples[2] = ('C', new_c1_id)
            tuples[4] = ('C', new_c2_id)
        else:
            raise ValueError("Bad length of tuples")
        
    if mask[1]:
        # shuffle the Op
        if len(tuples) == 6:
            tuples = _ops_shuffle(tuples)
        else:
            logging.warning(f"Not enough operands, length of tuples is {len(tuples)} | {tuples}")
    if mask[0]:
        # shuffle the Aggregator
        assert tuples[0][0] == 'A'
        tuples = _aggregator_shuffle(tuples, agg_only)
        
    return tuples

# TODO 
# need also t_id
def _get_random_cols(c_id, t_id, d, exclusive=[], same_type=True, k=1):
    col_name = d['col_set'][c_id]
    if col_name == '*':
        # if found '*' we don't shuffle
        # just return the c_id back
        return [-1] # wrap to list
    else:
        all_col_names_types = [(col_name, col_type) for (table_id, col_name), col_type in filter(lambda ele: ele[0][0] == t_id, zip(d['col_names'], d['col_types']))]
        # copy all in case of fallback 
        filtered_col_names_types = all_col_names_types[:]
        if same_type:
            col_type = [*filter(lambda ele: ele[0] == col_name, all_col_names_types)]
            assert len(col_type) == 1
            col_type = col_type[0][1]
            filtered_col_names_types = [*filter(lambda ele: ele[1] == col_type, filtered_col_names_types)]
        cols_pool = [d['col_set'].index(ele[0]) for ele in filtered_col_names_types]
        # fall back if len(cols_pool) = 0
        for ele in exclusive:
            cols_pool.remove(ele)
        if len(cols_pool) == 0:
            logging.warning(f"Fallback to any types of columns from table {d['table_names'][t_id]}, since no same type {col_type} of column {col_name} exists!")
            cols_pool = [d['col_set'].index(ele[0]) for ele in all_col_names_types]
            for ele in exclusive:
                cols_pool.remove(ele)
            assert len(cols_pool) > 0
        return random.sample(cols_pool, k)

def get_random_col(c_id, t_id, d, exclusive=[], same_type=True):
    return _get_random_cols(c_id, t_id, d, exclusive, same_type)[0]

def _aggregator_shuffle(tuples, agg_only=False):
    if agg_only:
        assert tuples[0][1] != 0
        # most simple rule: random sample [1,2,3,4,5]
        tuples[0] = ('A', random.sample(range(1, 6), k=1)[0])
    else:
        if tuples[0][1] == 0:
            tuples[0] = ('A', 3)
        else:
            tuples[0] = ('A', 0)
    return tuples


def _get_filters_pos(ast):
    ops = ast['ops']
    res = []
    for idx, op in enumerate(ops):
        if op == 'Filter':
            res.append(idx)
    return res

def filter_shuffle(ast, d, at_most=1):
    # return a new op_vals back
    ops = ast['ops']
    op_vals = ast['op_vals']
    new_op_vals = op_vals[:] # copy op_vals
    filter_indices = [idx for idx, _ in filter(lambda x: x[1] == 'Filter', enumerate(ops))]
    at_most = min(at_most, len(filter_indices))
    
    for filter_idx in filter_indices[:at_most]:
        filter_id = op_vals[filter_idx]
        temp_filter_id = _shuffle_filter(filter_id)
        if temp_filter_id >= 0:
            new_op_vals[filter_idx] = temp_filter_id
        else: # case of Filter between A V V
            # Locate the filter position and find the next closest V
            if temp_filter_id == -1:
                # shuffle any existed aggregator
                tuples_candidates = []
                for i, proj_pattern_tuples in enumerate(ast['proj_pattern_tuples']):
                    for j, pattern_tuples in enumerate(proj_pattern_tuples):
                        assert pattern_tuples[0][0] == 'A'
                        if pattern_tuples[0][1] != 0 and len(pattern_tuples) == 4 and pattern_tuples[2][1] != 0:
                            tuples_candidates.append([i, j])
                if any(tuples_candidates):
                    i, j = tuples_candidates[0]
                    ast['new_proj_pattern_tuples'][i][j] = tuples_shuffle(ast['proj_pattern_tuples'][i][j], d, mask = [1,0,0], agg_only = True)
            elif temp_filter_id == -2:
                v1_idx = ops[filter_idx:].index('V') + filter_idx
                v2_idx = v1_idx + 1
                assert ops[v1_idx] == 'V' and ops[v2_idx] == 'V'
                v1_op_val = new_op_vals[v1_idx]
                v2_op_val = new_op_vals[v2_idx]
                new_v1_value, new_v2_value = _value_schuffle(d['values'][v1_op_val], d['values'][v2_op_val], 'between')
                d['values'][v1_op_val] = new_v1_value
                d['values'][v2_op_val] = new_v2_value
            else:
                raise ValueError (f'Unknown implemented action for Filter Rule : {temp_filter_id}')
    return new_op_vals

def _shuffle_filter(filter_id,):
    _rules = [[0, 1], [2, 3], [4, 7], [9, 10], [11, 14], [16, 17], [18, 19]]
    rules = [[*range(rule[0], rule[1]+1)] for rule in _rules]
    """
    # Rule_1
    0: 'Filter and Filter Filter',
    1: 'Filter or Filter Filter',
    
    # Rule_2
    2: 'Filter = A V',
    3: 'Filter != A V',

    # Rule_3
    4: 'Filter < A V',
    5: 'Filter > A V',
    6: 'Filter <= A V',
    7: 'Filter >= A V',
    
    # Rule_4: shuffle Values
    8: 'Filter between A V V',
    
    # Rule_5
    9: 'Filter like A V',
    10: 'Filter not_like A V',
    
    # Rule_6
    # now begin root
    11: 'Filter = A Root',
    12: 'Filter < A Root',
    13: 'Filter > A Root',
    14: 'Filter != A Root',

    # Rule_7
    15: 'Filter between A Root', # raise error
    
    # Rule_8
    
    16: 'Filter >= A Root',
    17: 'Filter <= A Root',
    
    # Rule_9
    # now for In
    18: 'Filter in A Root',
    19: 'Filter not_in A Root'
    """
    # get the list of Filter indice
    if filter_id != 8:
        if filter_id != 15:
            for rule in rules:
                if filter_id in rule:
                    return _get_shuffle_ids(rule[0], rule[-1], exclusive=[filter_id], k=1)[0]
        else:
            logging.warning(f"*** GOT Filter BETWEEN A Root")
            return -1
    else:
        return -2
    raise ValueError(f"Unknown filter_id: {filter_id}")

def _value_schuffle(v1_val, v2_val, constraint='between'):
    type_v1 = _type_of_values(v1_val)
    type_v2 = _type_of_values(v2_val)
    if v2_val is not None:
        assert type_v1 == type_v2, f"value type not identical! v1_val: {type_v1}, v2_val: {type_v2}"

    if type_v1 == 'num':
        return _num_value_shuffle(v1_val, v2_val, constraint='between')
    elif type_v2 == 'date':
        return _date_value_shuffle(v1_val,v2_val, constraint='between')
    else:
        raise ValueError(f"type {_type_of_values(v1_val)} not implemented")
    
def _num_value_shuffle(v1_val, v2_val, constraint='between'):
    new_v1_val, new_v2_val = '0', '0'
    n_decimals = _num_of_decimals(v1_val)
    if constraint == 'between':
        assert v2_val is not None, f"v2_val not assigned!"
        # new_v1_val must be smaller than new_v2_val
        # use normal distribution to get more realistic random values.
        assert float(v1_val) <= float(v2_val), f"v1_val {v1_val} must less than v2_val {v2_val}"
        n_decimals = max(_num_of_decimals(v1_val), _num_of_decimals(v2_val))
        # here we apply normal distribution for generating random value
        new_v1_val, new_v2_val = _gaussian_random(v1_val, v2_val, multiplier=0.5)
    elif constraint is None:
        new_v1_val = _uniform_random(v1_val)
        new_v2_val = None
        if v2_val is not None:
            new_v2_val = _uniform_random(v2_val)
    else:
        raise ValueError(f"constraint: '{constraint}' not implemented")
    return new_v1_val, new_v2_val

def _date_value_shuffle(v1_val, v2_val=None, constraint='between'):
    date_1 = _str_to_date(v1_val)
    date_2 = None
    if v2_val is not None:
        date_2 = _str_to_date(v2_val)
    if constraint == 'between':
        assert date_2 is not None, f"v2_val not assigned!"
        # date_1 must be smaller than date_2
        offset = max(2, (date_2-date_1).days)
        offset_pool = [*range(-1*offset, 0)] + [*range(1, offset+1)]
        new_offset = random.sample(offset_pool, k=1)[0]
        new_date_1 = date_1 + timedelta(days=new_offset)
        new_date_2 = date_2 + timedelta(days=new_offset)
    elif constraint == None:
        # -/+10 days uniform distribution without 0
        offset = 10
        offset_pool = [*range(-1*offset, 0)] + [*range(1, offset+1)]
        new_offset = random.sample(offset_pool, k=1)[0]
        new_date_1 = date_1 + timedelta(days=new_offset)
        new_date_2 = None
        if v2_val is not None:
            new_date_2 = date_2 + timedelta(days=new_offset)
    else:
        raise ValueError(f"constraint: '{constraint}' not implemented")
    new_v1_val = new_date_1.strftime("%Y-%m-%d")
    new_v2_val = None
    if new_date_2 is not None:
        new_v2_val = new_date_2.strftime("%Y-%m-%d")
    return new_v1_val, new_v2_val


def _str_to_date(date_string, fmt='%Y-%m-%d'):
    return datetime.strptime(date_string, fmt)

def _type_of_values(str_val):
    if str_val is None:
        return None
    if len(str_val.split('-')) > 2:
        return 'date'
    else:
        return 'num'

def _num_of_decimals(str_num):
    # if num is integer, return 0,
    # else return the number of digits after '.'
    n_decimals = 0
    if '.' in str_num:
        n_decimals = len(str_num.split('.')[1])
    return n_decimals

def _gaussian_random(v1_val, v2_val, multiplier=0.5, asc_sorted=True):
    numeric = int
    n_decimals = max(_num_of_decimals(v1_val), _num_of_decimals(v2_val))
    if n_decimals > 0:
        numeric = float
    v1_val = numeric(v1_val)
    v2_val = numeric(v2_val)
    mu = (v1_val + v2_val) / 2 # mean
    sigma = mu * multiplier # set default deviation = half of mean
    _v1 = v1_val
    _v2 = v2_val
    while _v1 == v1_val or _v1 == v2_val:
        _v1 = round(random.gauss(mu, sigma), n_decimals)
    while _v2 == v2_val or _v2 == v1_val:
        _v2 = round(random.gauss(mu, sigma), n_decimals)
    if asc_sorted:
        if _v1 > _v2:
            _v1, _v2 = _v2, _v1
    return str(_v1), str(_v2)

def _uniform_random(str_num, offset=0.55):
    numeric = int
    n_decimals = _num_of_decimals(str_num)
    if n_decimals > 0:
        numeric = float
    num = numeric(str_num)
    # special case
    if num == 0:
        return 1
    a = num*(1-0.55)
    b = num*(1+0.55)
    if a > b:
        a, b = b, a
    _num = num
    while _num == num:
        _num = round(random.uniform(a, b), _num_of_decimals)
    return str(_num)

# TODO
# No Ops in Spidder. Not necessary to impl now.
def _ops_shuffle():
    pass

def _get_shuffle_ids(start, end, exclusive=[], k=1):
    pool = [*range(start, end+1)]
    for ele in exclusive:
        pool.remove(ele)
    return random.sample(pool, k)

def get_shuffle_id(start, end, exclusive=[], k=1):
    return _get_shuffle_ids(start, end, exclusive, k)[0]

def synthesize_ast(ast, op_vals=None):
    """
    this function only synthsize the ast by its ops, op_vals 
    and proj_pattern_tuples
    if no proj_pattern_tuples is specified, 
    ast['proj_pattern_tuples'] would be parser-in.
    """
    if op_vals is None:
        op_vals = ast['op_vals']
    proj_pattern_tuples = ast['new_proj_pattern_tuples']
    ops = ast['ops']
    # flatten the proj_pattern_tuples
    flat_tuples = list(itertools.chain(*list(itertools.chain(*proj_pattern_tuples))))
    start_idx = 0
    res = []
    for op, val in zip(ops, op_vals):
        if val == -1:
            assert op == flat_tuples[start_idx][0]
            res.append(f"{op}({flat_tuples[start_idx][1]})")
            start_idx += 1
        else:
            res.append(f"{op}({val})")
    new_ast = ' '.join(res)
    return new_ast

def test():
    # 8659 spider data samples
    data_file = '/Users/justin/Projects/fraunhofer/valuenet/data/spider/original/train.json'

    # 167 spider databases
    table_file = '/Users/justin/Projects/fraunhofer/valuenet/data/spider/original/tables.json'
    data, tables = load_datasets(data_file, table_file)
    data, tables = sql2semQL(data, tables)
    
    res = []
    for i, d in enumerate(data):
        # Parse AST
        temp = ast_parser(d)
        print(i, d['rule_label'])
        print(i, d['query'])
        # generate AST
        d = generate_new_ast(temp)
        # translate semql to sql
        print(i, d['generated_ast'])
        d['generated_query'] = semQL2sql(d, tables[d['db_id']], origin=d['generated_ast'])[0]
        print(i, d['generated_query'])
        assert d['generated_ast'] != d['rule_label'] or d['values'] != d['old_values'], f"{i} is wrong! \n***\n{d}"
        res.append(d)
    print("test succeed!")

def upper_all_keywords(sql):
    keywords = [' des ', ' asc ', ' and ', ' or ', ' sum\(', ' min\(', ' max\(', ' avg\(', ' between ', ' like ', ' not like '] + \
[' in ', ' not ', ' not in ', ' count\(', ' intersect ', ' union ', ' except ', ' distinct ']
    kw_pattern = re.compile('|'.join(keywords))
    new_sql = re.sub(kw_pattern, lambda m: m.group(0).upper(), sql)
    return new_sql
    

def main():
    args = load_args()
    schema_path, data_path = args.schema_path, args.data_path
    output_path = args.output_path
    try:
        data, tables = load_datasets(data_path, schema_path)
        data, tables = sql2semQL(data, tables)
        res = []
        for i, d in enumerate(data):
            d = ast_parser(d)
            temp = {
                'db_id': d['db_id'],
                'question': d['question'],
                'original_AST': d['rule_label'],
                # 'original_values': d['old_values'],
                # 'generated_ast': d['generated_ast'],
                # 'generated_values': d['values']
            }
            temp['original_query'] = upper_all_keywords(semQL2sql(d, tables[d['db_id']], origin=d['rule_label'])[0].strip())
            d = generate_new_ast(d)
            temp['generated_query'] = upper_all_keywords(semQL2sql(d, tables[d['db_id']], origin=d['generated_ast'])[0].strip())
            temp['original_values'] = d['old_values']
            temp['generated_ast'] = d['generated_ast']
            temp['generated_values'] = d['values']
            assert d['generated_ast'] != d['rule_label'] or d['values'] != d['old_values'], f"{i} is wrong! \n***\n{d}"
            keywords = [' des ', ' asc ', ' and ', ' or ', ' sum(', ' min(', ' max(', ' avg(', ' between ', ' like '] + \
[' in ', ' not ', ' count(', ' intersect ', ' union ', ' except ', ' distinct ']
            for kw in keywords:
                assert kw not in temp['original_query']
                assert kw not in temp['generated_query']
            res.append(temp)
        with open(output_path, 'w') as json_ouput:
            json.dump(res, json_ouput, sort_keys=True, indent=2)
    except FileExistsError as e:
        print(f"I/O ERROR: {e}")


if __name__ == '__main__':
    # test()
    main()
