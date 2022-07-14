import psycopg2, json, re
from collections import defaultdict
filepath = '/home/ubuntu/fraunhofer/valuenet/data/skyserver_dr16_2020_11_30/original/tables.json'
reserved_attr_filepath = '/home/ubuntu/fraunhofer/valuenet/data/skyserver_dr16_2020_11_30/original/reserved_attributes.json'
extra_fk_filepath = None
# extra_fk_filepath = '/home/ubuntu/fraunhofer/valuenet/data/skyserver_dr16_2020_11_30/original/extra_fk.json'
tbl_schema = None

with open(filepath, 'r') as json_file:
    tbl_schema = json.load(json_file)[0]
# print(tbl_schema)
print(tbl_schema.keys())
tbl_original_names = tbl_schema['table_names_original']
col_original_names = tbl_schema['column_names_original']
pks = tbl_schema['primary_keys'] # [1,2,3]
fks = tbl_schema['foreign_keys'] # [[1,2],[3,4]]

# get reserved attr index, apply it as exclusive elements to prune_col_index
with open(reserved_attr_filepath, 'r') as reserved_attr_json_file:
    reserved_attr = json.load(reserved_attr_json_file)
tbl2idx_dict = {}
for idx, tbl in enumerate(tbl_original_names):
    tbl2idx_dict[tbl] = idx
for ele in reserved_attr:
    ele[0] = tbl2idx_dict[ele[0]]
# print(reserved_attr)
reserved_attr_idx = [i for i, ele in enumerate(col_original_names) if ele in reserved_attr]

# get extra forignkey
extra_fks_list = []
if extra_fk_filepath:
    with open(extra_fk_filepath, 'r') as extra_fk_json_file:
        extra_fks = json.load(extra_fk_json_file)

    for fk_1, fk_2 in extra_fks:
        tbl_1, col_1 = fk_1[0],fk_1[1]
        tbl_2, col_2 = fk_2[0],fk_2[1]
        tbl_1_idx = tbl_original_names.index(tbl_1) 
        tbl_2_idx = tbl_original_names.index(tbl_2)
        extra_fks_list.append([col_original_names.index([tbl_1_idx, col_1]), col_original_names.index([tbl_2_idx, col_2])])

    fks += extra_fks_list 

# all cols bound with pk or fk, we need reserve all these keys and got their information to get 
key_col_list = set(pks)
for [c1,c2] in fks:
    key_col_list.add(c1)
    key_col_list.add(c2)
key_col_list = [*key_col_list]
key_col_dict = {}
for key_col in key_col_list:
    v = tbl_schema['column_names_original'][key_col]
    key_col_dict[key_col] = v
print(key_col_dict)

print("***key_col_list***", key_col_list)
pks_cols = [key_col_dict[pk] for pk in pks]
fks_cols = [[key_col_dict[fk1],key_col_dict[fk2]] for fk1, fk2 in fks]
# Connect to an existing database

conn = psycopg2.connect(
    database="skyserver_dr16_2020_11_30",
    user="postgres",
    host="testbed.inode.igd.fraunhofer.de",
    password="vdS83DJSQz2xQ",
    port=18001
)


# Open a cursor to extract database relations labels
cur = conn.cursor()
# Query the database and obtain data as Python objects
cur.execute("SELECT * FROM translation_relation_node_labels;")
tbl_mapping = cur.fetchall()

tbl_dict = {}
for _, tbl_original_name, tbl_name in tbl_mapping:
    tbl_dict[tbl_original_name] = tbl_name
for i, tbl_original_name in enumerate(tbl_original_names):
    if tbl_dict.get(tbl_original_name, None):
        tbl_schema['table_names'][i] = tbl_dict[tbl_original_name]
# Close communication with the database
cur.close()

# Open a cursor to extract database attributes labels
cur = conn.cursor()
cur.execute("SELECT * FROM translation_attribute_node_labels;")
# [(1, 'photoobj', 'objid', 'object id'), (2, 'photoobj', 'b', 'galactic latitude'), (3, 'photoobj', 'l', 'galactic longitude'), (4, 'photoobj', 'mjd', 'modified julien date'), (5, 'photoobj', 'tai_u', 'time of observation u')]
col_mapping = cur.fetchall()
col_dict = defaultdict(dict)
for _, tbl_original_name, col_original_name, col_name in col_mapping:
    col_dict[tbl_original_name][col_original_name] = col_name

# print([tbl_schema['column_names_original'].index(x) for x in pks_cols])
for i, (tbl_idx, col_original_name) in enumerate(col_original_names):
    if tbl_idx >= 0:
        if col_dict[tbl_original_names[tbl_idx]].get(col_original_name, None):
            tbl_schema['column_names'][i] = [tbl_idx, col_dict[tbl_original_names[tbl_idx]][col_original_name]]

# experiment: pruning all cols without re-defined names. i.e. 'column_names' != 'column_names_original'
prune_col_idx = []
for i, ((tbl_idx,col_name), (_, col_original_name)) in enumerate(zip(tbl_schema['column_names_original'], tbl_schema['column_names'])):
        if (col_name == col_original_name or col_name ==re.sub(r' ',  '_', col_original_name)) and tbl_idx > -1 and tbl_idx:
            prune_col_idx.append(i)

reserved_col_list = sorted(list(set(reserved_attr_idx+key_col_list)))
# print("***reserved col list***", reserved_col_list, "\n", len(reserved_col_list))
prune_col_idx = [col_idx for col_idx in prune_col_idx if col_idx not in reserved_col_list]

# print([(tbl_schema['column_names_original'][i], tbl_schema['column_names'][i], tbl_schema['column_types'][i]) for i in prune_col_idx])
print("*** total count ***", len(prune_col_idx))
tbl_schema['column_names_original'] = [ele for i, ele in enumerate(tbl_schema['column_names_original']) if i not in prune_col_idx]
tbl_schema['column_names'] = [ele for i, ele in enumerate(tbl_schema['column_names']) if i not in prune_col_idx]
tbl_schema['column_types'] = [ele for i, ele in enumerate(tbl_schema['column_types']) if i not in prune_col_idx]
tbl_schema['primary_keys'] = [tbl_schema['column_names_original'].index(pk_col) for pk_col in pks_cols]
tbl_schema['foreign_keys'] = [[tbl_schema['column_names_original'].index(fk_col_1), tbl_schema['column_names_original'].index(fk_col_2)] for [fk_col_1, fk_col_2] in fks_cols]

# check the consistency
for (tbl_idx_original,_), (tbl_idx, _) in zip(tbl_schema['column_names_original'], tbl_schema['column_names']):
    assert(tbl_idx_original==tbl_idx)
cur.close()
conn.close()


with open(filepath, 'wt') as out:
    json.dump([tbl_schema], out, sort_keys=True, indent=2, separators=(',', ': '))


