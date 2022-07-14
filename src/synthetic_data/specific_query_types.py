from typing import Dict

def sdss_query_types() -> Dict[str, int]:
    """
    A list of common query types, which usually work well with GPT-3. This schemas have been painfully hand-selected
    based on the analysis done in group_pairs_to_find_templates.py and manual testing with GPT-3.

    Some special indexing values:
    - if we use -1 for a column index, we refer to the '*' column (eg. in a COUNT(*) query)
    - if we use -2 or less for a column index, we required this column to be a numeric/date column. This is a special case for the SemQL Sup() operator
    - if we use 100 or greater for a column index, we require this column to be a categorical column. 
    - all column indices unique, except for any arbitary column with the name, e.g. if c(0)T(0) is 'table_0.id', then c(0)T(1) is 'table_1.id'
    
    For any other index number, we just sample a column/table/value. Be aware though, that the use the same sampled value for
    the same index (so 0 --> column_A will stay like this throughout the whole query)


    @return: A list of query types (SemQL), including a multiplier for the number of queries to generate.
    """

    
    # all templates with between, like will be ignored
    res = {
    'Root1(3) Root(3) Sel(0) N(0) A(0) Op(0) C(0) T(0) C(0) T(0) Filter(0) Filter(2) A(0) Op(0) C(100) T(0) C(100) T(0) V(0) Filter(2) A(0) Op(0) C(101) T(0) C(101) T(0) V(1)': 3,
    'Root1(3) Root(3) Sel(0) N(0) A(0) Op(0) C(0) T(0) C(0) T(0) Filter(2) A(0) Op(0) C(1) T(1) C(1) T(1) V(0)': 2,
    'Root1(3) Root(3) Sel(0) N(0) A(0) Op(0) C(0) T(0) C(0) T(0) Filter(0) Filter(1) Filter(2) A(0) Op(0) C(1) T(0) C(1) T(0) V(0) Filter(2) A(0) Op(0) C(1) T(0) C(1) T(0) V(1) Filter(2) A(0) Op(0) C(2) T(0) C(2) T(0) V(2)': 1,
    'Root1(3) Root(3) Sel(0) N(0) A(0) Op(0) C(0) T(0) C(0) T(0) Filter(0) Filter(2) A(0) Op(0) C(1) T(0) C(1) T(0) V(0) Filter(1) Filter(2) A(0) Op(0) C(2) T(0) C(2) T(0) V(1) Filter(2) A(0) Op(0) C(2) T(0) C(2) T(0) V(2)': 1,
    'Root1(3) Root(3) Sel(0) N(0) A(3) Op(0) C(-1) T(0) C(-1) T(0) Filter(2) A(0) Op(0) C(100) T(0) C(100) T(0) V(0)': 1,
    'Root1(3) Root(3) Sel(0) N(1) A(3) Op(0) C(-1) T(0) C(-1) T(0) A(0) Op(0) C(100) T(1) C(100) T(1) Filter(2) A(0) Op(0) C(100) T(1) C(100) T(1) V(0)': 1,
    'Root1(3) Root(3) Sel(0) N(3) A(0) Op(0) C(0) T(0) C(0) T(0) A(0) Op(0) C(1) T(0) C(1) T(0) A(0) Op(0) C(2) T(0) C(2) T(0) A(0) Op(0) C(-2) T(0) C(-2) T(0) Filter(0) Filter(2) A(0) Op(0) C(100) T(0) C(100) T(0) V(0) Filter(0) Filter(5) A(0) Op(0) C(-2) T(0) C(-2) T(0) V(1) Filter(2) A(0) Op(0) C(101) T(0) C(101) T(0) V(2)': 1,
    'Root1(3) Root(3) Sel(0) N(3) A(0) Op(0) C(0) T(0) C(0) T(0) A(0) Op(0) C(1) T(0) C(1) T(0) A(0) Op(0) C(2) T(0) C(2) T(0) A(0) Op(0) C(3) T(0) C(3) T(0) Filter(0) Filter(2) A(0) Op(0) C(100) T(0) C(100) T(0) V(0) Filter(0) Filter(5) A(0) Op(0) C(-2) T(0) C(-2) T(0) V(1) Filter(4) A(0) Op(0) C(-3) T(0) C(-3) T(0) V(2)': 1,
    'Root1(3) Root(5) Sel(0) N(0) A(0) Op(0) C(-1) T(0) C(-1) T(0)': 1,
    'Root1(3) Root(5) Sel(0) N(1) A(0) Op(0) C(0) T(0) C(0) T(0) A(3) Op(0) C(-1) T(0) C(-1) T(0)': 1,
    'Root1(3) Root(3) Sel(0) N(0) A(0) Op(0) C(0) T(0) C(0) T(0) Filter(0) Filter(2) A(0) Op(0) C(100) T(0) C(100) T(0) V(0) Filter(0) Filter(2) A(0) Op(0) C(101) T(0) C(101) T(0) V(1) Filter(1) Filter(2) A(0) Op(0) C(102) T(0) C(102) T(0) V(2) Filter(2) A(0) Op(0) C(102) T(0) C(102) T(0) V(3)': 1,
    'Root1(3) Root(3) Sel(0) N(0) A(0) Op(0) C(0) T(0) C(0) T(0) Filter(0) Filter(1) Filter(2) A(0) Op(0) C(100) T(0) C(100) T(0) V(0) Filter(1) Filter(2) A(0) Op(0) C(100) T(0) C(100) T(0) V(1) Filter(1) Filter(2) A(0) Op(0) C(100) T(0) C(100) T(0) V(2) Filter(2) A(0) Op(0) C(100) T(0) C(100) T(0) V(3) Filter(2) A(0) Op(0) C(101) T(0) C(101) T(0) V(4)': 1,
    'Root1(3) Root(3) Sel(0) N(1) A(0) Op(0) C(0) T(0) C(0) T(0) A(0) Op(0) C(1) T(1) C(1) T(1) Filter(0) Filter(2) A(0) Op(0) C(100) T(1) C(100) T(1) V(0) Filter(0) Filter(4) A(0) Op(1) C(2) T(0) C(3) T(0) V(1) Filter(5) A(0) Op(1) C(2) T(0) C(3) T(0) V(2)': 1,
    'Root1(3) Root(3) Sel(0) N(1) A(0) Op(0) C(0) T(0) C(0) T(0) A(0) Op(0) C(1) T(0) C(1) T(0) Filter(2) A(0) Op(0) C(100) T(1) C(100) T(1) V(0)': 1,
    'Root1(3) Root(3) Sel(0) N(1) A(0) Op(0) C(0) T(0) C(0) T(0) A(0) Op(0) C(1) T(1) C(1) T(1) Filter(2) A(0) Op(0) C(100) T(1) C(100) T(1) V(0)': 1,
    'Root1(3) Root(3) Sel(0) N(0) A(1) Op(0) C(-2) T(0) C(-2) T(0) Filter(2) A(0) Op(0) C(100) T(0) C(100) T(0) V(0)': 1,
    'Root1(3) Root(3) Sel(0) N(0) A(2) Op(0) C(-2) T(0) C(-2) T(0) Filter(2) A(0) Op(0) C(100) T(0) C(100) T(0) V(0)': 1,
    'Root1(3) Root(3) Sel(0) N(0) A(0) Op(0) C(0) T(0) C(0) T(0) Filter(3) A(0) Op(0) C(100) T(0) C(100) T(0) V(0)': 1,
    'Root1(3) Root(3) Sel(0) N(0) A(0) Op(0) C(0) T(0) C(0) T(0) Filter(3) A(0) Op(0) C(100) T(1) C(100) T(1) V(0)': 1,
    'Root1(3) Root(3) Sel(0) N(0) A(3) Op(0) C(-1) T(0) C(-1) T(0) Filter(0) Filter(2) A(0) Op(0) C(100) T(0) C(100) T(0) V(0) Filter(2) A(0) Op(0) C(101) T(0) C(101) T(0) V(1)': 1,
    'Root1(3) Root(5) Sel(1) N(0) A(0) Op(0) C(100) T(0) C(100) T(0)': 1,
    'Root1(3) Root(5) Sel(0) N(0) A(3) Op(0) C(-1) T(0) C(-1) T(0)': 1,
    'Root1(3) Root(2) Sel(0) N(1) A(0) Op(0) C(1) T(0) C(1) T(0) A(0) Op(0) C(-2) T(1) C(-2) T(1) Sup(1) A(0) Op(0) C(-2) T(1) C(-2) T(1)': 1
    }

    """cordis templates
    # Modification for adapting to the extended SemQL grammar
    return {
        'Root1(3) Root(3) Sel(0) N(0) A(0) Op(0) C(0) T(0) C(0) T(0) Filter(2) A(0) Op(0) C(1) T(0) C(1) T(0) V(0)': 3,  # Simple select and filter, no join. Example: SELECT document_status_description FROM Ref_Document_Status WHERE document_status_code = "working"
        'Root1(3) Root(3) Sel(0) N(0) A(0) Op(0) C(0) T(0) C(0) T(0) Filter(2) A(0) Op(0) C(1) T(1) C(1) T(1) V(0)': 1,  # Select, filter and one join. Example: SELECT T2.name FROM Flight AS T1 JOIN Aircraft AS T2 ON T1.aid  =  T2.aid WHERE T1.flno  =  99
        'Root1(3) Root(5) Sel(0) N(1) A(0) Op(0) C(0) T(0) C(0) T(0) A(3) Op(0) C(-1) T(0) C(-1) T(0)': 1,  # two selects: a simple column and a count, both on the same table. Using a group by. Example: SELECT payment_method_code ,  count(*) FROM INVOICES GROUP BY payment_method_code
        'Root1(3) Root(5) Sel(0) N(0) A(3) Op(0) C(-1) T(0) C(-1) T(0)': 1,  # a simple count on a full table. Example: SELECT count(*) FROM railway
        # GPT-3 can't handle double selects it seems 'Root1(3) Root(5) Sel(0) N(1) A(0) C(0) T(0) A(0) C(1) T(0)',  # a double select from the same table. Example: SELECT first_name ,  last_name FROM Customers;
        # GPT-3 can't handle double selects it seems 'Root1(3) Root(5) Sel(0) N(1) A(0) C(0) T(0) A(0) C(1) T(1)',  # a double select from two different tables. Example: SELECT customer_id ,  last_name FROM Customers JOIN Persons ON Customers.cid  =  Persons.cid;
        # GPT-3 can't handle double selects it seems 'Root1(3) Root(3) Sel(0) N(1) A(0) C(0) T(0) A(0) C(1) T(0) Filter(2) A(0) C(2) T(0) V(0)',  # a double select and a filter, all from one table. Example: SELECT Customer_Phone ,  Customer_Email_Address FROM CUSTOMERS WHERE Customer_Name  =  "Harold",
        # GPT-3 can't handle it! 'Root1(3) Root(2) Sel(0) N(0) A(0) C(0) T(0) Sup(0) A(3) C(-1) T(0)' # a superlative (count and limit 1) Example: SELECT country FROM stadium GROUP BY country ORDER BY count(*) DESC LIMIT 1
        'Root1(3) Root(3) Sel(0) N(0) A(3) Op(0) C(-1) T(0) C(-1) T(0) Filter(2) A(0) Op(0) C(0) T(0) C(0) T(0) V(0)': 2,  # a count but restricted on a certain filter: SELECT count(*) FROM campuses WHERE county  =  "Los Angeles"
        'Root1(3) Root(2) Sel(0) N(0) A(0) Op(0) C(1) T(0) C(1) T(0) Sup(0) A(0) Op(0) C(-2) T(0) C(-2) T(0)': 1,  # a superlative on a numeric/date column Example: SELECT Player_name FROM player ORDER BY Votes DESC LIMIT 1
        #  GPT-3 can't handle it! 'Root1(3) Root(4) Sel(0) N(0) A(0) C(1) T(0) Order(1) A(0) C(-2) T(0)',
        'Root1(3) Root(3) Sel(0) N(0) A(0) Op(0) C(0) T(0) C(0) T(0) Filter(0) Filter(2) A(0) Op(0) C(1) T(0) C(1) T(0) V(0) Filter(2) A(0) Op(0) C(2) T(0) C(2) T(0) V(1)': 1,  # works only semi-well... a simple select and two filter with AND operator. No join. Example: SELECT t1.campusfee FROM csu_fees AS t1 JOIN campuses AS t2 ON t1.campus  =  t2.id WHERE t2.campus  =  "San Jose State University" AND t1.year  =  2000
        # GPT-3 seems not to be able to handle DISTINCT properly... 'Root1(3) Root(5) Sel(1) N(0) A(0) C(0) T(0)',  # a distinct select. Example: SELECT DISTINCT Visit_Date FROM VISITS
        'Root1(3) Root(3) Sel(0) N(0) A(0) Op(0) C(0) T(0) C(0) T(0) Filter(1) Filter(2) A(0) Op(0) C(1) T(0) C(1) T(0) V(0) Filter(2) A(0) Op(0) C(2) T(0) C(2) T(0) V(1)': 1,  # a select and an OR filter (on the same column). Example: SELECT Name FROM phone WHERE Carrier  =  "Sprint" OR Carrier  =  "TMobile",
        'Root1(3) Root(3) Sel(0) N(0) A(0) Op(0) C(0) T(0) C(0) T(0) Filter(3) A(0) Op(0) C(1) T(0) C(1) T(0) V(0)': 2,  # simple select and a filter with an unequal operator. Example: SELECT name FROM channel WHERE OWNER != 'CCTV'
        'Root1(3) Root(3) Sel(0) N(0) A(0) Op(0) C(0) T(0) C(0) T(0) Filter(3) A(0) Op(0) C(1) T(1) C(1) T(1) V(0)': 2,  # simple select and a filter with an unequal operator. Join over two tables. Example: SELECT T1.Name FROM people AS T1 JOIN perpetrator AS T2 ON T1.People_ID  =  T2.People_ID WHERE T2.Country != "China"
    }
    """

    return res

def cordis_query_types() -> Dict[str, int]:
    """
    A list of common query types, which usually work well with GPT-3. This schemas have been painfully hand-selected
    based on the analysis done in group_pairs_to_find_templates.py and manual testing with GPT-3.

    Some special indexing values:
    - if we use -1 for a column index, we refer to the '*' column (eg. in a COUNT(*) query)
    - if we use -2 or less for a column index, we required this column to be a numeric/date column. This is a special case for the SemQL Sup() operator
    - if we use 100 or greater for a column index, we require this column to be a categorical column. 
    - all column indices unique, except for any arbitary column with the name, e.g. if c(0)T(0) is 'table_0.id', then c(0)T(1) is 'table_1.id'
    
    For any other index number, we just sample a column/table/value. Be aware though, that the use the same sampled value for
    the same index (so 0 --> column_A will stay like this throughout the whole query)


    @return: A list of query types (SemQL), including a multiplier for the number of queries to generate.
    """

    
    # all templates with between, like will be ignored
    res = {
    "Root1(3) Root(3) Sel(0) N(0) A(5) Op(0) C(-2) T(0) C(-2) T(0) Filter(0) Filter(2) A(0) Op(0) C(0) T(1) C(0) T(1) V(0) Filter(2) A(0) Op(0) C(1) T(0) C(1) T(0) V(1)": 1,
    "Root1(3) Root(3) Sel(0) N(0) A(5) Op(0) C(0) T(0) C(0) T(0) Filter(0) Filter(3) A(0) Op(0) C(1) T(1) C(1) T(1) V(0) Filter(2) A(0) Op(0) C(-2) T(0) C(-2) T(0) V(1)" :1,
    "Root1(3) Root(3) Sel(1) N(0) A(0) Op(0) C(0) T(0) C(0) T(0) Filter(2) A(0) Op(0) C(1) T(1) C(1) T(1) V(0)": 1,
    "Root1(3) Root(3) Sel(0) N(0) A(5) Op(0) C(-2) T(0) C(-2) T(0) Filter(0) Filter(2) A(0) Op(0) C(1) T(1) C(1) T(1) V(0) Filter(3) A(0) Op(0) C(2) T(2) C(2) T(2) V(1)": 1,
    "Root1(3) Root(3) Sel(0) N(0) A(4) Op(0) C(0) T(0) C(0) T(0) Filter(0) Filter(2) A(0) Op(0) C(1) T(1) C(1) T(1) V(0) Filter(5) A(0) Op(0) C(-2) T(0) C(-2) T(0) V(1)": 1,
    "Root1(3) Root(3) Sel(1) N(0) A(3) Op(0) C(0) T(0) C(0) T(0) Filter(0) Filter(3) A(0) Op(0) C(1) T(1) C(1) T(1) V(0) Filter(2) A(0) Op(0) C(2) T(2) C(2) T(2) V(1)": 1,
    "Root1(3) Root(3) Sel(0) N(0) A(5) Op(0) C(0) T(0) C(0) T(0) Filter(0) Filter(2) A(0) Op(0) C(1) T(1) C(1) T(1) V(0) Filter(3) A(0) Op(0) C(2) T(0) C(2) T(0) V(1)": 1,
    "Root1(3) Root(3) Sel(0) N(0) A(4) Op(0) C(-2) T(0) C(-2) T(0) Filter(0) Filter(2) A(0) Op(0) C(0) T(1) C(0) T(1) V(0) Filter(2) A(0) Op(0) C(1) T(0) C(1) T(0) V(1)": 1,
    "Root1(3) Root(3) Sel(1) N(0) A(3) Op(0) C(0) T(0) C(0) T(0) Filter(3) A(0) Op(0) C(1) T(1) C(1) T(1) V(0)": 1,
    "Root1(3) Root(3) Sel(1) N(0) A(0) Op(0) C(0) T(0) C(0) T(0) Filter(7) A(0) Op(0) C(-2) T(1) C(-2) T(1) V(0)": 1,
    "Root1(3) Root(3) Sel(0) N(0) A(4) Op(0) C(-2) T(0) C(-2) T(0) Filter(0) Filter(3) A(0) Op(0) C(0) T(1) C(0) T(1) V(0) Filter(2) A(0) Op(0) C(1) T(2) C(1) T(2) V(1)": 1,
    "Root1(3) Root(3) Sel(0) N(0) A(5) Op(0) C(-2) T(0) C(-2) T(0) Filter(0) Filter(3) A(0) Op(0) C(0) T(1) C(0) T(1) V(0) Filter(0) Filter(2) A(0) Op(0) C(1) T(0) C(1) T(0) V(1) Filter(2) A(0) Op(0) C(2) T(2) C(2) T(2) V(2)": 1,
    "Root1(3) Root(3) Sel(0) N(0) A(5) Op(0) C(-2) T(0) C(-2) T(0) Filter(0) Filter(3) A(0) Op(0) C(0) T(1) C(0) T(1) V(0) Filter(2) A(0) Op(0) C(1) T(2) C(1) T(2) V(1)": 1,
    "Root1(3) Root(3) Sel(0) N(0) A(5) Op(0) C(-2) T(0) C(-2) T(0) Filter(0) Filter(3) A(0) Op(0) C(0) T(1) C(0) T(1) V(0) Filter(3) A(0) Op(0) C(1) T(0) C(1) T(0) V(1)": 1,
    "Root1(3) Root(3) Sel(0) N(0) A(4) Op(0) C(-2) T(0) C(-2) T(0) Filter(0) Filter(2) A(0) Op(0) C(0) T(1) C(0) T(1) V(0) Filter(2) A(0) Op(0) C(1) T(2) C(1) T(2) V(1)": 1,
    "Root1(3) Root(3) Sel(1) N(0) A(0) Op(0) C(0) T(0) C(0) T(0) Filter(0) Filter(2) A(0) Op(0) C(1) T(1) C(1) T(1) V(0) Filter(3) A(0) Op(0) C(2) T(2) C(2) T(2) V(1)": 1,
    "Root1(3) Root(3) Sel(1) N(0) A(3) Op(0) C(0) T(0) C(0) T(0) Filter(0) Filter(2) A(0) Op(0) C(1) T(1) C(1) T(1) V(0) Filter(0) Filter(2) A(0) Op(0) C(2) T(2) C(2) T(2) V(1) Filter(2) A(0) Op(0) C(3) T(3) C(3) T(3) V(2)": 1,
    "Root1(3) Root(3) Sel(1) N(0) A(0) Op(0) C(0) T(0) C(0) T(0) Filter(0) Filter(2) A(0) Op(0) C(1) T(1) C(1) T(1) V(0) Filter(0) Filter(2) A(0) Op(0) C(2) T(2) C(2) T(2) V(1) Filter(2) A(0) Op(0) C(3) T(0) C(3) T(0) V(2)": 1,
    "Root1(3) Root(3) Sel(1) N(0) A(3) Op(0) C(0) T(0) C(0) T(0) Filter(0) Filter(3) A(0) Op(0) C(1) T(1) C(1) T(1) V(0) Filter(0) Filter(2) A(0) Op(0) C(2) T(2) C(2) T(2) V(1) Filter(3) A(0) Op(0) C(3) T(3) C(3) T(3) V(2)": 1,
    "Root1(3) Root(3) Sel(1) N(0) A(0) Op(0) C(0) T(0) C(0) T(0) Filter(0) Filter(2) A(0) Op(0) C(1) T(0) C(1) T(0) V(0) Filter(3) A(0) Op(0) C(2) T(1) C(2) T(1) V(1)": 1
    }

    return res

def oncomx_query_types():
    res = {
            "Root1(3) Root(3) Sel(0) N(0) A(0) Op(0) C(-1) T(0) C(-1) T(0) Filter(2) A(0) Op(0) C(0) T(0) C(0) T(0) V(0)": 1,
            "Root1(3) Root(3) Sel(0) N(0) A(3) Op(0) C(-1) T(0) C(-1) T(0) Filter(2) A(0) Op(0) C(0) T(0) C(0) T(0) V(0)": 1,
            "Root1(3) Root(5) Sel(0) N(0) A(0) Op(0) C(0) T(0) C(0) T(0)": 1,
            "Root1(3) Root(5) Sel(1) N(0) A(0) Op(0) C(0) T(0) C(0) T(0)": 1,
            "Root1(3) Root(5) Sel(0) N(1) A(0) Op(0) C(0) T(0) C(0) T(0) A(0) Op(0) C(1) T(1) C(1) T(1)": 1,
            "Root1(3) Root(3) Sel(1) N(1) A(0) Op(0) C(0) T(0) C(0) T(0) A(0) Op(0) C(1) T(1) C(1) T(1) Filter(0) Filter(2) A(0) Op(0) C(2) T(2) C(2) T(2) V(0) Filter(2) A(0) Op(0) C(1) T(1) C(1) T(1) V(1)": 1,
            "Root1(3) Root(3) Sel(1) N(0) A(0) Op(0) C(0) T(0) C(0) T(0) Filter(0) Filter(2) A(0) Op(0) C(1) T(1) C(1) T(1) V(1) Filter(4) A(0) Op(0) C(-2) T(1) C(-2) T(1) V(0)": 1,
            "Root1(3) Root(3) Sel(1) N(0) A(0) Op(0) C(0) T(0) C(0) T(0) Filter(0) Filter(2) A(0) Op(0) C(1) T(1) C(1) T(1) V(0) Filter(3) A(0) Op(0) C(2) T(2) C(2) T(2) V(1)": 1,
            "Root1(3) Root(3) Sel(0) N(0) A(0) Op(0) C(-1) T(0) C(-1) T(0) Filter(0) Filter(2) A(0) Op(0) C(0) T(1) C(0) T(1) V(0) Filter(2) A(0) Op(0) C(2) T(0) C(2) T(0) V(1)": 1,
            "Root1(3) Root(3) Sel(0) N(1) A(0) Op(0) C(0) T(0) C(0) T(0) A(0) Op(0) C(1) T(0) C(1) T(0) Filter(0) Filter(2) A(0) Op(0) C(2) T(1) C(2) T(1) V(0) Filter(2) A(0) Op(0) C(3) T(2) C(3) T(2) V(1)": 1,
            "Root1(3) Root(3) Sel(0) N(1) A(0) Op(0) C(0) T(0) C(0) T(0) A(0) Op(0) C(1) T(1) C(1) T(1) Filter(1) Filter(2) A(0) Op(0) C(1) T(1) C(1) T(1) V(0) Filter(2) A(0) Op(0) C(2) T(1) C(2) T(1) V(1)": 1,
            "Root1(3) Root(3) Sel(0) N(1) A(0) Op(0) C(0) T(0) C(0) T(0) A(0) Op(0) C(1) T(1) C(1) T(1) Filter(2) A(0) Op(0) C(1) T(1) C(1) T(1) V(0)": 1,
            "Root1(3) Root(3) Sel(0) N(0) A(0) Op(0) C(-1) T(0) C(-1) T(0) Filter(5) A(0) Op(0) C(0) T(1) C(0) T(1) V(0)": 1,
            "Root1(3) Root(5) Sel(0) N(0) A(3) Op(0) C(-1) T(0) C(-1) T(0)": 1,
            "Root1(3) Root(3) Sel(1) N(0) A(0) Op(0) C(0) T(0) C(0) T(0) Filter(2) A(0) Op(0) C(1) T(0) C(1) T(0) V(0)": 1,
            "Root1(3) Root(5) Sel(1) N(0) A(3) Op(0) C(0) T(0) C(0) T(0)": 1,
            "Root1(3) Root(2) Sel(0) N(0) A(0) Op(0) C(0) T(0) C(0) T(0) Sup(0) A(0) Op(0) C(-2) T(0) C(-2) T(0)": 1
        }
    return res