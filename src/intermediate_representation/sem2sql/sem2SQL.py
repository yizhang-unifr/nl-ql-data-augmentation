import argparse
import os
import traceback
import numbers

from intermediate_representation.graph import Graph
from intermediate_representation.sem2sql.infer_from_clause import infer_from_clause
from intermediate_representation.semQL import Sup, Sel, Order, Root, Filter, A, N, C, T, Root1, V, Op
from intermediate_representation.sem_utils import alter_column0


def split_logical_form(lf):
    indexs = [i+1 for i, letter in enumerate(lf) if letter == ')']
    indexs.insert(0, 0)
    components = list()
    for i in range(1, len(indexs)):
        components.append(lf[indexs[i-1]:indexs[i]].strip())
    return components


def pop_front(array):
    if len(array) == 0:
        return 'None'
    return array.pop(0)


def peek_front(array):
    if len(array) == 0:
        return 'None'
    return array[0]


def is_end(components, transformed_sql, is_root_processed):
    end = False
    c = pop_front(components)
    c_instance = eval(c)

    if isinstance(c_instance, Root) and is_root_processed:
        # intersect, union, except
        end = True
    elif isinstance(c_instance, Filter):
        if 'where' not in transformed_sql:
            end = True
        else:
            num_conjunction = 0
            for f in transformed_sql['where']:
                if isinstance(f, str) and (f == 'and' or f == 'or'):
                    num_conjunction += 1
            current_filters = len(transformed_sql['where'])
            valid_filters = current_filters - num_conjunction
            if valid_filters >= num_conjunction + 1:
                end = True
    elif isinstance(c_instance, Order):
        if 'order' not in transformed_sql:
            end = True
        elif len(transformed_sql['order']) == 0:
            end = False
        else:
            end = True
    elif isinstance(c_instance, Sup):
        if 'sup' not in transformed_sql:
            end = True
        elif len(transformed_sql['sup']) == 0:
            end = False
        else:
            end = True
    components.insert(0, c)
    return end


def _transform(components, transformed_sql, col_set, table_names, values, schema):
    processed_root = False
    current_table = schema

    while len(components) > 0:
        if is_end(components, transformed_sql, processed_root):
            break
        c = pop_front(components)
        c_instance = eval(c)
        if isinstance(c_instance, Root):
            processed_root = True

            # a list with only 2 elements - similar to the spider data structure
            # [0] is used for distinct true/false. [1] contains the array with selection columns
            transformed_sql['select'] = [False, list()]

            if c_instance.id_c == 0:
                transformed_sql['where'] = list()
                transformed_sql['sup'] = list()
            elif c_instance.id_c == 1:
                transformed_sql['where'] = list()
                transformed_sql['order'] = list()
            elif c_instance.id_c == 2:
                transformed_sql['sup'] = list()
            elif c_instance.id_c == 3:
                transformed_sql['where'] = list()
            elif c_instance.id_c == 4:
                transformed_sql['order'] = list()
        elif isinstance(c_instance, Sel):
            if c_instance.id_c == 1:
                # The statement is distinct
                transformed_sql['select'][0] = True
        elif isinstance(c_instance, N):
            for i in range(c_instance.id_c + 1):
                agg = eval(pop_front(components))
                op = eval(pop_front(components))
                column = eval(pop_front(components))
                _table = pop_front(components)
                table = eval(_table)
                if not isinstance(table, T):
                    table = None
                    components.insert(0, _table)
                assert isinstance(agg, A) and isinstance(
                    column, C) and isinstance(op, Op)

                if table is not None:
                    col, _ = replace_col_with_original_col(
                        col_set[column.id_c], table_names[table.id_c], current_table)
                else:
                    col = col_set[column.id_c]
                # add another C T

                column_2 = eval(pop_front(components))
                table_2 = eval(pop_front(components))
                assert isinstance(column_2, C) and isinstance(
                    table_2, T)
                col_2, _ = replace_col_with_original_col(
                    col_set[column_2.id_c], table_names[table_2.id_c], current_table)

                # if we have an aggregation (e.g. COUNT(xy)) and the query as a whole is a distinct, we also use distinct
                # inside the aggregation (e.g. COUNT(DISTINCT xy)). While this is an oversimplification, it works for most cases
                # as it is hard to formulate a question in a way that some things are distinct and some are not.
                use_distinct = False
                if agg.id_c != 0 and transformed_sql['select'][0]:
                    use_distinct = True

                temp = (
                    agg.production.split()[1],
                    op.production.split()[1],
                    col,
                    table_names[table.id_c] if table is not None else table,
                    col_2,
                    table_names[table_2.id_c] if table_2 is not None else table_2,
                    use_distinct
                )
                if op.id_c == 0:
                    assert column.id_c == column_2.id_c
                    if table is not None:
                        table.id_c == table_2.id_c
                    else:
                        table is None and table_2 is None
                transformed_sql['select'][1].append(temp)

        elif isinstance(c_instance, Sup):
            transformed_sql['sup'].append(c_instance.production.split()[1])
            agg = eval(pop_front(components))
            op = eval(pop_front(components))
            column = eval(pop_front(components))
            _table = pop_front(components)
            table = eval(_table)
            if not isinstance(table, T):
                table = None
                components.insert(0, _table)
            assert isinstance(agg, A) and isinstance(
                column, C) and isinstance(op, Op)

            transformed_sql['sup'].append(agg.production.split()[1])
            transformed_sql['sup'].append(op.production.split()[1])
            if table:
                column_final, _ = replace_col_with_original_col(
                    col_set[column.id_c], table_names[table.id_c], current_table)
            else:
                column_final = col_set[column.id_c]
                raise RuntimeError('not found table !!!!')
            transformed_sql['sup'].append(column_final)
            transformed_sql['sup'].append(
                table_names[table.id_c] if table is not None else table)

            column_2 = eval(pop_front(components))
            table_2 = eval(pop_front(components))
            if table_2:
                column_final_2, _ = replace_col_with_original_col(
                    col_set[column_2.id_c], table_names[table_2.id_c], current_table)
            else:
                column_final_2 = col_set[column_2.id_c]
                raise RuntimeError('not found table 2 !!!!')
            transformed_sql['sup'].append(column_final_2)
            transformed_sql['sup'].append(
                table_names[table_2.id_c] if table_2 is not None else table_2)

        elif isinstance(c_instance, Order):
            transformed_sql['order'].append(c_instance.production.split()[1])
            agg = eval(pop_front(components))
            op = eval(pop_front(components))
            column = eval(pop_front(components))
            _table = pop_front(components)
            table = eval(_table)
            if not isinstance(table, T):
                table = None
                components.insert(0, _table)
            assert isinstance(agg, A) and isinstance(
                column, C) and isinstance(op, Op)
            transformed_sql['order'].append(agg.production.split()[1])
            transformed_sql['order'].append(op.production.split()[1])
            transformed_sql['order'].append(replace_col_with_original_col(
                col_set[column.id_c], table_names[table.id_c], current_table)[0])
            transformed_sql['order'].append(
                table_names[table.id_c] if table is not None else table)

            column_2 = eval(pop_front(components))
            _table_2 = pop_front(components)
            table_2 = eval(_table_2)
            if not isinstance(table_2, T):
                table_2 = None
                components.insert(0, _table_2)
            transformed_sql['order'].append(replace_col_with_original_col(
                col_set[column_2.id_c], table_names[table_2.id_c], current_table)[0])
            transformed_sql['order'].append(
                table_names[table_2.id_c] if table_2 is not None else table_2)

        elif isinstance(c_instance, Filter):
            op = c_instance.production.split()[1]
            # to handel"not_like", "not in"
            op = ' '.join(op.split('_'))
            if op == 'and' or op == 'or':
                transformed_sql['where'].append(op)
            else:
                # No sub-query
                agg = eval(pop_front(components))
                m_op = eval(pop_front(components))
                column = eval(pop_front(components))
                _table = pop_front(components)
                table = eval(_table)
                if not isinstance(table, T):
                    table = None
                    components.insert(0, _table)
                assert isinstance(agg, A) and isinstance(
                    column, C) and isinstance(m_op, Op)

                column_2 = eval(pop_front(components))
                _table_2 = pop_front(components)
                table_2 = eval(_table_2)
                if not isinstance(table_2, T):
                    table_2 = None
                    components.insert(0, _table_2)
                assert isinstance(column_2, C)
                if m_op.id_c == 0:
                    assert column.id_c == column_2.id_c
                    if table is not None:
                        table.id_c == table_2.id_c
                    else:
                        table is None and table is None

                # here we verify if there is a sub- query
                component = peek_front(components)
                if isinstance(eval(component), V):
                    if table and table_2:
                        column_final, column_final_idx = replace_col_with_original_col(
                            col_set[column.id_c], table_names[table.id_c], current_table)
                        column_final_2, column_final_2_idx = replace_col_with_original_col(
                            col_set[column_2.id_c], table_names[table_2.id_c], current_table)
                    else:
                        column_final = col_set[column.id_c]
                        column_final_2 = col_set[column_2.id_c]
                        raise RuntimeError('Table not found!')

                    second_value = None
                    # now we handle the values - we also handle data types properly.
                    value_obj = eval(pop_front(components))
                    value = format_value_given_datatype(
                        column_final_idx, schema, values[value_obj.id_c], c_instance)

                    # there is a few special cases where we have to deal with multiple values - e.g. in the "X BETWEEN Y AND Z" case.
                    if isinstance(eval(peek_front(components)), V):
                        second_value_obj = eval(pop_front(components))
                        second_value = format_value_given_datatype(
                            column_final_idx, schema, values[second_value_obj.id_c], c_instance)
                    temp = (
                        op,
                        agg.production.split()[1],
                        m_op.production.split()[1],
                        column_final,
                        table_names[table.id_c] if table is not None else table,
                        column_final_2,
                        table_names[table_2.id_c] if table_2 is not None else table_2,
                        value,
                        second_value,
                    )
                    transformed_sql['where'].append(temp)
                else:
                    # sub-query
                    new_dict = dict()
                    new_dict['sql'] = transformed_sql['sql']
                    temp = (
                        op,
                        agg.production.split()[1],
                        m_op.production.split()[1],
                        replace_col_with_original_col(
                            col_set[column.id_c], table_names[table.id_c], current_table)[0],
                        table_names[table.id_c] if table is not None else table,
                        replace_col_with_original_col(
                            col_set[column_2.id_c], table_names[table_2.id_c], current_table)[0],
                        table_names[table_2.id_c] if table_2 is not None else table_2,
                        _transform(components, new_dict, col_set,
                                   table_names, values, schema),
                        None
                    )
                    transformed_sql['where'].append(temp)

    return transformed_sql


def transform(query, schema, origin=None, readable_alias=False):
    preprocess_schema(schema)
    if origin is None:
        lf = query['model_result_replace']
    else:
        lf = origin
    # lf = query['rule_label']
    col_set = query['col_set']
    table_names = query['table_names']
    values = query['values']
    current_table = schema

    current_table['schema_content_clean'] = [x[1]
                                             for x in current_table['column_names']]
    current_table['schema_content'] = [x[1]
                                       for x in current_table['column_names_original']]

    components = split_logical_form(lf)

    transformed_sql = dict()
    transformed_sql['sql'] = query
    c = pop_front(components)
    c_instance = eval(c)
    assert isinstance(c_instance, Root1)
    if c_instance.id_c == 0:
        transformed_sql['intersect'] = dict()
        transformed_sql['intersect']['sql'] = query

        _transform(components, transformed_sql,
                   col_set, table_names, values, schema)
        _transform(
            components, transformed_sql['intersect'], col_set, table_names, values, schema)
    elif c_instance.id_c == 1:
        transformed_sql['union'] = dict()
        transformed_sql['union']['sql'] = query
        _transform(components, transformed_sql,
                   col_set, table_names, values, schema)
        _transform(
            components, transformed_sql['union'], col_set, table_names, values, schema)
    elif c_instance.id_c == 2:
        transformed_sql['except'] = dict()
        transformed_sql['except']['sql'] = query
        _transform(components, transformed_sql,
                   col_set, table_names, values, schema)
        _transform(
            components, transformed_sql['except'], col_set, table_names, values, schema)
    else:
        _transform(components, transformed_sql,
                   col_set, table_names, values, schema)

    parse_result = to_str(transformed_sql, 1, schema, readable_alias=readable_alias)

    parse_result = parse_result.replace('\t', '')
    return [parse_result]


def col_to_str(agg, op, col, tab, col_2, tab_2, table_names, N=1, is_distinct=False, readable_alias=False):
    _col = col.replace(' ', '_')
    _col_2 = col_2.replace(' ', '_')
    distinct_str = 'DISTINCT ' if is_distinct else ''
    if tab not in table_names:
        if readable_alias:
            table_names[tab] = tab.replace(' ', '_')
        else:
            table_names[tab] = 'T' + str(len(table_names) + N)
    table_alias = table_names[tab]

    if agg == 'none':

        if op == 'none':
            if col == '*':
                return '*'
            return '%s.%s' % (table_alias, _col)
        else:  # with ops
            if tab_2 not in table_names:
                if readable_alias:
                    table_names[tab_2] = tab_2.replace(' ', '_')
                else:
                    table_names[tab_2] = 'T' + str(len(table_names) + N)
            table_alias_2 = table_names[tab_2]

            if col_2 != 'none' and tab_2 != 'none':
                return '%s.%s %s %s.%s' % (table_alias, _col, op, table_alias_2, _col_2)
            else:
                raise AssertionError(f"invalid col_2 or tab_2")
    else:  # with aggr.
        if op == 'none':
            if col == '*':
                return '%s(%s%s)' % (agg, distinct_str, _col)
            else:
                if tab is not None and tab not in table_names:
                    if readable_alias:
                        table_names[tab] = tab.replace(' ', '_')
                    else:
                        table_names[tab] = 'T' + str(len(table_names) + N)
                table_alias = table_names[tab]
                return '%s(%s %s.%s)' % (agg, distinct_str, table_alias, _col)
        else:  # with ops
            if tab_2 not in table_names:
                if readable_alias:
                    table_names[tab_2] = tab_2.replace(' ', '_')
                else:
                    table_names[tab_2] = 'T' + str(len(table_names) + N)
            table_alias_2 = table_names[tab_2]
            return '%s(%s %s.%s %s %s.%s)' % (agg, distinct_str, table_alias, _col, op, table_alias_2, _col_2)


def replace_col_with_original_col(query, col, current_table):
    # print(query, col)
    if query == '*':
        return query, None

    cur_table = col
    cur_col = query
    single_final_col = None
    single_final_col_idx = None
    # print(query, col)
    for col_ind, col_name in enumerate(current_table['schema_content_clean']):
        if col_name == cur_col:
            assert cur_table in current_table['table_names']
            if current_table['table_names'][current_table['col_table'][col_ind]] == cur_table:
                single_final_col = current_table['column_names_original'][col_ind][1]
                single_final_col_idx = col_ind
                break

    assert single_final_col
    # if query != single_final_col:
    #     print(query, single_final_col)
    return single_final_col, single_final_col_idx


def build_graph(schema):
    relations = list()
    foreign_keys = schema['foreign_keys']
    for (fkey, pkey) in foreign_keys:

        fkey_table = schema['table_names_original'][schema['column_names'][fkey][0]]
        fkey_original_name = schema['column_names_original'][fkey][1]

        pkey_table = schema['table_names_original'][schema['column_names'][pkey][0]]
        pkey_original_name = schema['column_names_original'][pkey][1]

        relations.append((fkey_table, fkey_original_name,
                         pkey_table, pkey_original_name))
        relations.append((pkey_table, pkey_original_name,
                         fkey_table, fkey_original_name))

    return Graph(relations)


def preprocess_schema(schema):
    tmp_col = []
    for cc in [x[1] for x in schema['column_names']]:
        if cc not in tmp_col:
            tmp_col.append(cc)
    schema['col_set'] = tmp_col
    # print table
    schema['schema_content'] = [col[1] for col in schema['column_names']]
    schema['col_table'] = [col[0] for col in schema['column_names']]
    graph = build_graph(schema)
    schema['graph'] = graph


def format_value_given_datatype(column_final_idx, schema, value, filter_instance):
    # there is some special cases on the training set where the user asks for an "empty" history or
    # similar. In that case, we need to really add an empty string.
    if value == '':
        return "\'\'"

    # before we handle the values, we wanna find out what data-type the value has (based on the schema)
    if column_final_idx is not None:
        data_type = schema['column_types'][column_final_idx]
        if data_type == 'text':
            use_quotes = True
        elif data_type == 'time':
            use_quotes = True
        else:
            use_quotes = True
            try:
                _ = float(value)
                use_quotes = False
            except Exception as e:
                print(e)
    else:
        # this means we are comparing the value with an aggregation - e.g. a COUNT(*). So it must be a number.
        use_quotes = False

    # sometimes a column is text, but the value in it is a number. We then have to remove the floating point, as
    # otherwise, the comparison is wrong. Example: a boolean column (as VARCHAR(1), you have compare 1 with '1' and not with '1.0'
    # for every other numeric comparison, the float/int difference is handled properly by SQL.
    if use_quotes and isinstance(value, numbers.Number):
        value = int(value)

    # filter 9 is 'LIKE' or 'NOT LIKE' - we need to add the wildcards to the value.
    # TODO: introduce new filter actions for LIKE_FUZZY_BEGINNING and LIKE_FUZZY_ENDING
    if filter_instance.id_c == 9 or filter_instance.id_c == 10:
        value = f'%{value}%'

    value_formatted = "\"{}\"".format(value) if use_quotes else value

    return value_formatted


def to_str(sql_json, N_T, schema, pre_table_names=None, readable_alias=False):
    all_columns = list()
    select_clause = list()
    table_names = dict()
    current_table = schema

    whole_select_distinct = 'DISTINCT ' if sql_json['select'][0] else ''
    for sql_json_sel in sql_json['select'][1]:
        if len(sql_json_sel) == 7:
            agg, op, col, tab, col_2, tab_2, distinct = sql_json_sel
        else:
            raise AssertionError(
                f"unrecognized select query!  ***{sql_json_sel}***")
        all_columns.append((agg, col, tab))
        if op != 'none':
            # TODO assert
            all_columns.append((agg, col_2, tab_2))
        select_clause.append(col_to_str(
            agg, op, col, tab, col_2, tab_2, table_names, N_T, distinct, readable_alias=readable_alias))

    select_clause_str = 'SELECT ' + whole_select_distinct + \
        ', '.join(select_clause).strip()

    sup_clause = ''
    order_clause = ''
    direction_map = {"des": 'DESC', 'asc': 'ASC'}

    if 'sup' in sql_json:
        if len(sql_json['sup']) == 7:
            (direction, agg, op, col, tab, col_2, tab_2) = sql_json['sup']
        else:
            raise AssertionError(f"unrecognized sup: {sql_json['sup']}")
        all_columns.append((agg, col, tab))
        if op != 'none':
            # TODO assert
            all_columns.append((agg, col_2, tab_2))

        subject = col_to_str(agg, op, col, tab, col_2, tab_2, table_names, N_T, readable_alias=readable_alias)
        # TODO change LIMIT 1 to LIMIT V
        sup_clause = ('ORDER BY %s %s LIMIT 1' %
                      (subject, direction_map[direction])).strip()
    # TODO add Op support
    elif 'order' in sql_json:
        if len(sql_json['order']) == 7:
            (direction, agg, op, col, tab, col_2, tab_2) = sql_json['order']
        else:
            raise AssertionError(f"unrecgnized order: {sql_json['order']}")
        all_columns.append((agg, col, tab))
        if op != 'none':
            all_columns.append((agg, col_2, tab_2))
        subject = col_to_str(agg, op, col, tab, col_2, tab_2, table_names, N_T, readable_alias=readable_alias)
        order_clause = ('ORDER BY %s %s' %
                        (subject, direction_map[direction])).strip()

    has_group_by = False
    where_clause = ''
    have_clause = ''

    if 'where' in sql_json:
        conjunctions = list()
        filters = list()
        # print(sql_json['where'])
        for f in sql_json['where']:
            if isinstance(f, str):
                conjunctions.append(f)
            else:
                if len(f) == 9:
                    op, agg, m_op, col, tab, col_2, tab_2, value, value2 = f
                all_columns.append((agg, col, tab))
                if m_op != 'none':
                    all_columns.append((agg, col_2, tab_2))
                subject = col_to_str(agg, m_op, col, tab,
                                     col_2, tab_2, table_names, N_T, readable_alias=readable_alias)

                # here we detect the difference between a simple value or a value which refers to a subquery
                if not isinstance(value, dict):
                    values_combined = value
                    if value2 is not None:
                        # right now the only case where two values are allowed is a BETWEEN X AND Y statement.
                        values_combined = "{} AND {}".format(value, value2)

                    filters.append('%s %s %s' % (subject, op, values_combined))
                else:
                    value['sql'] = sql_json['sql']

                    # This is kind of a style-change: instead of "xy IN (SELECT z FROM a)" one can also rewrite the query to a simple JOIN (this is done with adding the table).
                    # While it is critical for Exact Matching (where joins are checked), it is also necessary for Execution Accuracy. The reason is simply that a "EXISTS (SELECT ID FROM ...)" or a "IN (SELECT ID FROM ...)"
                    # behave slightly different than a JOIN, especially when it comes to their shortcut-behaviour and therefore also the natural order by (read more here) https://blog.jooq.org/2016/03/09/sql-join-or-exists-chances-are-youre-doing-it-wrong/

                    # as we choose to model simple joins (see model_simple_joins_as_filter.py) with a filter, we use the opportunity here to change the filter back to a normal join - as in the ground truth expected.
                    # Obviously this would be an issue if a query tries to exactly use "IN (SELECT ID FROM XY)". There is a few examples where this is the case, e.g.
                    # SELECT avg(age) FROM Dogs WHERE dog_id IN ( SELECT dog_id FROM Treatments ) (look for this in ground truth)

                    number_of_selects = len(value['select'][1])
                    first_select_aggregation = value['select'][1][0][0]

                    if op == 'in' and number_of_selects == 1 \
                            and first_select_aggregation == 'none' \
                            and 'where' not in value \
                            and 'order' not in value \
                            and 'sup' not in value:

                        if value['select'][1][0][3] not in table_names:
                            if readable_alias:
                                table_names[value['select'][1][0][3]] = value['select'][1][0][3].replace(' ', '_')
                            else:
                                table_names[value['select'][1][0][3]] = 'T' + str(len(table_names) + N_T)
                        # This is necessary to avoid incorrect queries: if there is an "and/or" conjunction at the end of the filter, we need to put a next filter to avoid an invalid query.
                        # If we though apply a join instead of an "IN ()" statement, we need to remove that conjunction.
                        if len(conjunctions) > 0:
                            conjunctions.pop()

                        filters.append(None)

                    else:
                        # for every other sub-query we use a recursion to transform it to a string. By using a high NT-value we avoid conflicts with table-aliases.
                        filters.append('%s %s %s' % (
                            subject, op, '(' + to_str(value, len(table_names) + N_T + 20, schema, readable_alias=readable_alias) + ')'))
                if len(conjunctions):
                    filters.append(conjunctions.pop())

        aggs = ['count(', 'avg(', 'min(', 'max(', 'sum(']
        having_filters = list()
        idx = 0
        while idx < len(filters):
            _filter = filters[idx]
            if _filter is None:
                idx += 1
                continue
            for agg in aggs:
                if _filter.startswith(agg):
                    having_filters.append(_filter)
                    filters.pop(idx)
                    # print(filters)
                    if 0 < idx and (filters[idx - 1] in ['and', 'or']):
                        filters.pop(idx - 1)
                        # print(filters)
                    break
            else:
                idx += 1
        if len(having_filters) > 0:
            have_clause = 'HAVING ' + ' '.join(having_filters).strip()
        if len(filters) > 0:
            # print(filters)
            filters = [_f for _f in filters if _f is not None]
            conjun_num = 0
            filter_num = 0
            for _f in filters:
                if _f in ['or', 'and']:
                    conjun_num += 1
                else:
                    filter_num += 1
            if conjun_num > 0 and filter_num != (conjun_num + 1):
                # assert 'and' in filters
                idx = 0
                while idx < len(filters):
                    if filters[idx] == 'and':
                        if idx - 1 == 0:
                            filters.pop(idx)
                            break
                        if filters[idx - 1] in ['and', 'or']:
                            filters.pop(idx)
                            break
                        if idx + 1 >= len(filters) - 1:
                            filters.pop(idx)
                            break
                        if filters[idx + 1] in ['and', 'or']:
                            filters.pop(idx)
                            break
                    idx += 1
            if len(filters) > 0:
                where_clause = 'WHERE ' + ' '.join(filters).strip()
                where_clause = where_clause.replace('not in', 'NOT IN')
            else:
                where_clause = ''

        if len(having_filters) > 0:
            has_group_by = True

    for agg in ['count(', 'avg(', 'min(', 'max(', 'sum(']:
        if (len(sql_json['select'][1]) > 1 and agg in select_clause_str)\
                or agg in sup_clause or agg in order_clause:
            has_group_by = True
            break

    group_by_clause = ''
    if has_group_by:
        if len(table_names) == 1:
            # check none agg
            is_agg_flag = False
            for f in sql_json['select'][1]:
                if len(f) == 7:
                    (agg, m_op, col, tab, col_2, tab_2, _) = f
                else:
                    raise AssertionError(f"unrecognized select: {f}")
                if agg == 'none':  #  only the last projection without agg in group by
                    group_by_clause = 'GROUP BY ' + \
                        col_to_str(agg, m_op, col, tab, col_2,
                                   tab_2, table_names, N_T, readable_alias=readable_alias)
                else:
                    is_agg_flag = True

            # if all has no aggr flag, concate all projection fields in group by
            #  question by Yi: why len(group_by_clause) > 5
            if is_agg_flag is False and len(group_by_clause) > 5:
                group_by_clause = "GROUP BY"
                for f in sql_json['select'][1]:
                    if len(f) == 7:
                        (agg, m_op, col, tab, col_2, tab_2, _) = f
                    else:
                        raise AssertionError(f"unrecognized select: {f}")
                    group_by_clause = group_by_clause + ' ' + \
                        col_to_str(agg, m_op, col, tab, col_2,
                                   tab_2, table_names, N_T, readable_alias=readable_alias) + ','
                # remove the last comma
                group_by_clause = group_by_clause[:-1]

            if len(group_by_clause) < 5:
                if 'count(*)' in select_clause_str:
                    current_table = schema
                    for primary in current_table['primary_keys']:
                        if current_table['table_names'][current_table['col_table'][primary]] in table_names:
                            group_by_clause = 'GROUP BY ' + col_to_str('none', 'none', current_table['schema_content'][primary],
                                                                       current_table['table_names'][
                                                                           current_table['col_table'][primary]], 'none', 'none',
                                                                       table_names, N_T, readable_alias=readable_alias)
        else:
            # if only one select
            if len(sql_json['select'][1]) == 1:
                f = sql_json['select'][1][0]
                if len(f) == 7:
                    agg, m_op, col, tab, col_2, tab_2, _ = f
                else:
                    raise AssertionError(f"unrecognized select: {f}")
                group_by_clause = 'GROUP BY ' + \
                    col_to_str(agg, m_op, col, tab, col_2,
                               tab_2, table_names, N_T, readable_alias=readable_alias)

            else:
                # check if there are only one non agg
                non_agg, non_agg_count = None, 0
                non_lists = []
                for f in sql_json['select'][1]:
                    if len(f) == 7:
                        agg, m_op, col, tab, col_2, tab_2, _ = f
                    else:
                        raise AssertionError(f"unrecognized select: {f}")
                    if agg == 'none':
                        non_agg = (agg, m_op, col, tab, col_2, tab_2)
                        non_lists.append(tab)
                        non_agg_count += 1

                non_lists = list(set(non_lists))
                # print(non_lists)
                if non_agg_count == 1:
                    group_by_clause = 'GROUP BY ' + \
                        col_to_str(*non_agg, table_names=table_names, N=N_T, readable_alias=readable_alias)
                elif non_agg:
                    find_flag = False
                    fix_flag = False
                    find_primary = None
                    if len(non_lists) <= 1:
                        for key, value in table_names.items():
                            if key not in non_lists:
                                non_lists.append(key)
                    if len(non_lists) > 1:
                        a = non_lists[0]
                        b = None
                        for non in non_lists:
                            if a != non:
                                b = non
                        if b:
                            for pair in current_table['foreign_keys']:
                                t1 = current_table['table_names'][current_table['col_table'][pair[0]]]
                                t2 = current_table['table_names'][current_table['col_table'][pair[1]]]
                                if t1 in [a, b] and t2 in [a, b]:
                                    if pre_table_names and t1 not in pre_table_names:
                                        assert t2 in pre_table_names
                                        t1 = t2
                                    group_by_clause = 'GROUP BY ' + col_to_str('none',
                                                                               'none',
                                                                               current_table['schema_content'][pair[0]],
                                                                               t1,
                                                                               'none',
                                                                               'none',
                                                                               table_names, N_T, readable_alias=readable_alias)
                                    fix_flag = True
                                    break
                    tab = non_agg[3]
                    assert tab in current_table['table_names']

                    for primary in current_table['primary_keys']:
                        if current_table['table_names'][current_table['col_table'][primary]] == tab:
                            find_flag = True
                            find_primary = (
                                current_table['schema_content'][primary], tab)
                    if fix_flag is False:
                        if find_flag is False:
                            # rely on count *
                            foreign = []
                            for pair in current_table['foreign_keys']:
                                if current_table['table_names'][current_table['col_table'][pair[0]]] == tab:
                                    foreign.append(pair[1])
                                if current_table['table_names'][current_table['col_table'][pair[1]]] == tab:
                                    foreign.append(pair[0])

                            for pair in foreign:
                                if current_table['table_names'][current_table['col_table'][pair]] in table_names:
                                    group_by_clause = 'GROUP BY ' + col_to_str('none', 'none', current_table['schema_content'][pair],
                                                                               current_table['table_names'][current_table['col_table'][pair]],
                                                                               'none', 'none',
                                                                               table_names, N_T, readable_alias=readable_alias)
                                    find_flag = True
                                    break
                            if find_flag is False:
                                for f in sql_json['select'][1]:
                                    if len(f) == 7:
                                        (agg, m_op, col, tab, col_2, tab_2, _) = f
                                    else:
                                        raise AssertionError(
                                            f"unrecognized select: {f}")
                                    if 'id' in col.lower() and m_op == 'none':
                                        group_by_clause = 'GROUP BY ' + \
                                            col_to_str(
                                                agg, m_op, col, tab, col_2, tab_2, table_names, N_T, readable_alias=readable_alias)
                                        break
                                if len(group_by_clause) > 5:  # WHY > 5?
                                    pass
                                else:
                                    raise RuntimeError('fail to convert')
                        else:
                            group_by_clause = 'GROUP BY ' + col_to_str('none', 'none', find_primary[0],
                                                                       find_primary[1],
                                                                       'none',
                                                                       'none',
                                                                       table_names, N_T, readable_alias=readable_alias)
    intersect_clause = ''
    if 'intersect' in sql_json:
        sql_json['intersect']['sql'] = sql_json['sql']
        intersect_clause = 'INTERSECT ' + \
            to_str(sql_json['intersect'], len(
                table_names) + 1, schema, table_names, readable_alias=readable_alias)
    union_clause = ''
    if 'union' in sql_json:
        sql_json['union']['sql'] = sql_json['sql']
        union_clause = 'UNION ' + \
            to_str(sql_json['union'], len(
                table_names) + 1, schema, table_names, readable_alias=readable_alias)
    except_clause = ''
    if 'except' in sql_json:
        sql_json['except']['sql'] = sql_json['sql']
        except_clause = 'EXCEPT ' + \
            to_str(sql_json['except'], len(
                table_names) + 1, schema, table_names, readable_alias=readable_alias)

    # print(current_table['table_names_original'])
    table_names_replace = {}
    for a, b in zip(current_table['table_names_original'], current_table['table_names']):
        table_names_replace[b] = a
    new_table_names = {}
    for key, value in table_names.items():
        if key is None:
            continue
        new_table_names[table_names_replace[key]] = value
    from_clause = infer_from_clause(
        new_table_names, schema['graph'], all_columns, readable_alias=readable_alias).strip()

    sql = ' '.join([select_clause_str, from_clause, where_clause, group_by_clause, have_clause, sup_clause, order_clause,
                    intersect_clause, union_clause, except_clause]).strip()

    return sql


def transform_semQL_to_sql(schemas, sem_ql_prediction, output_dir):

    # TODO: find out if this adds any benefit for the trained models. If we run it with the ground truth (so no prediction, just SQL -> SemQL -> SQL) it is even slightly better without it.
    # alter_not_in(sem_ql_prediction, schemas=schemas)
    # alter_inter(sem_ql_prediction)
    alter_column0(sem_ql_prediction)

    index = range(len(sem_ql_prediction))
    count = 0
    exception_count = 0
    with open(os.path.join(output_dir, 'output.txt'), 'w', encoding='utf8') as d, open(os.path.join(output_dir, 'ground_truth.txt'), 'w', encoding='utf8') as g:
        for i in index:
            try:
                result = transform(
                    sem_ql_prediction[i], schemas[sem_ql_prediction[i]['db_id']])
                d.write(result[0] + '\n')
                g.write("%s\t%s\t%s\n" % (replace_tabs_linebreaks(i, sem_ql_prediction),
                        sem_ql_prediction[i]["db_id"], sem_ql_prediction[i]["question"]))
                count += 1
            except Exception as e:
                # This origin seems to be the fallback-query. Not sure how we come up with it, most probably it's just a dummy query to fill in a result for each example.
                result = transform(sem_ql_prediction[i], schemas[sem_ql_prediction[i]
                                   ['db_id']], origin='Root1(3) Root(5) Sel(0) N(0) A(3) Op(0) C(0) T(0) C(0) T(0)')
                exception_count += 1
                d.write(result[0] + '\n')
                g.write("%s\t%s\t%s\n" % (replace_tabs_linebreaks(i, sem_ql_prediction),
                        sem_ql_prediction[i]["db_id"], sem_ql_prediction[i]["question"]))
                count += 1
                # print(e)
                print('Exception')
                print(traceback.format_exc())
                print(sem_ql_prediction[i]['question'])
                print(replace_tabs_linebreaks(i, sem_ql_prediction))
                print(sem_ql_prediction[i]['db_id'])
                print('===\n\n')

    return count, exception_count


def replace_tabs_linebreaks(i, sem_ql_prediction):
    return sem_ql_prediction[i]['query'].replace('\t', ' ').replace('\n', ' ')
