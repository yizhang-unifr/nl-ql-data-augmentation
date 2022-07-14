import json
import os
import re
import random
from math import factorial
from itertools import combinations, product, islice
from spacy.lang.en import English
from manual_inference.helper import get_schema_sdss
from intermediate_representation.semQL import Root1, Root, Sel, Sup, N, A, C, T, Filter, V, Op
from data_augmentation.sql_generator import sql2semQL
from intermediate_representation.sem2sql.sem2SQL import transform


dataset = [
    {
        "db_id": "skyserver_dr16_2020_11_30",
        "query": "select p.objid, s.specobjid from photoobj as p join specobj as s on s.bestobjid = p.objid where p.cModelMag_g > 0 and p.cModelMag_g < 23 or p.cModelMagErr_g < 0.2 and p.clean = 1 and s.z <= 0.1 and s.class = 'GALAXY'",
        "question": "DUMMY"
    }
    #{
    #    "db_id": "skyserver_dr16_2020_11_30",
    #    "query": "select p.objid, s.specobjid from photoobj as p join specobj as s on s.bestobjid = p.objid where p.cModelMag_g > 0 and p.cModelMag_g < 23 and p.cModelMagErr_g < 0.2 and p.cModelMagErr_r < 0.2 and p.cModelMag_u - p.cModelMag_g > 0.3 and p.cModelMag_r - p.cModelMag_g > 0.5 and p.cModelMag_r - p.cModelMag_i < 1.0 and p.cModelMag_i - p.cModelMag_g > 0.5 and (p.cModelMag_r - p.cModelMag_g > 0.7 or p.cModelMag_i - p.cModelMag_g > 1.0) and p.petrorad_g < 5 and p.petroR90_r < 5.0 and p.clean = 1 and s.z <= 0.1 and s.class = 'GALAXY'",
    #    "question": "DUMMY"
    #}
    
    #{
    #    "db_id": "skyserver_dr16_2020_11_30",
    #    "query": "select t1.objid, t1.u + t1.g, sum(t1.u + t1.g) from photoobj as t1 where t1.type = 6 and (t1.u - t1.g) < 0.4 and (t1.g + t1.r) < 0.7 and (t1.r * t1.i) > 0.4 and (t1.i / t1.z) > 0.4 order by t1.u + t1.g",
    #    "question": "Find Cataclysmic Variables from photometry"
    #}
    
]

dir_path = os.path.dirname(__file__)
par_path = os.path.dirname(dir_path)
root_path = os.path.dirname(par_path)
data_path = os.path.join(root_path,'data')
schema_dict_path = "skyserver_dr16_2020_11_30/original/tables.json"
schema_dict_path = os.path.join(data_path, schema_dict_path)
with open(schema_dict_path, 'r') as input:
    _schema_dict = json.load(input)[0]
schema_dict = {_schema_dict['db_id']: _schema_dict}


def main():

    data, table = sql2semQL(dataset=dataset, schema_dict=schema_dict, table_file=schema_dict_path)
    print(data[0]['sql']['select'])
    print()
    print(data[0]['sql']['where'])
    print()
    print(data[0]['sql']['orderBy'])
    print()
    print(data[0]['rule_label'])
    res = []
    
    for d in data:
        res.append(
            transform(d, table['skyserver_dr16_2020_11_30'], origin=d['rule_label']))
    print(*res, sep='\n')
    


if __name__ == '__main__':
    main()


"""
Root1(3) Root(3) Sel(0) N(1) A(0) Op(0) C(8) T(1) A(0) Op(0) C(8) T(4) Filter(0) Filter(5) A(0) Op(0) C(75) T(1) V(0) Filter(0) Filter(1) Filter(4) A(0) Op(0) C(75) T(1) V(1) Filter(4) A(0) Op(0) C(80) T(1) V(2) Filter(0) Filter(2) A(0) Op(0) C(15) T(1) V(3) Filter(0) Filter(6) A(0) Op(0) C(124) T(4) V(4) Filter(2) A(0) Op(0) C(127) T(4) V(5)

values: ['0', '23', '0.2', '1', '0.1', 'GALAXY']
"""