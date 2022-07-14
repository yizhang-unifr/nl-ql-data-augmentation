import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional


class GenerativeSchema:
    def __init__(self, generative_schema_path: Path) -> None:
        with open(generative_schema_path) as f:
            self.schema = json.load(f)

    @property
    def tables(self) -> List[str]:
        return [table['name'] for table in self.schema]

    def get_math_ops_tables(self) -> Optional[List]:
        return [t for t in self.schema if t.get('math_ops_columns', None)]

    def get_schema_for_math_ops_columns(self, table: str) -> Optional[Dict]:
        return [t for t in self.schema if t['name'] == table][0].get('math_ops_columns', None)
    
    def get_schema_for_math_operands_columns(self, table: str, column: str) -> Optional[Dict]:
        table_schema = self.get_schema_for_table(table)
        return [c for c in table_schema['columns'] if c['name'] == column][0].get('math_operands', None)

    def get_schema_for_table(self, table: str) -> Dict:
        return [t for t in self.schema if t['name'] == table][0]

    def get_schema_for_column(self, table: str, column: str) -> Dict:
        table_schema = self.get_schema_for_table(table)
        return [c for c in table_schema['columns'] if c['name'] == column][0]

    def all_columns_of_table(self, table: str) -> List[str]:
        table_schema = self.get_schema_for_table(table)
        return [column['name'] for column in table_schema['columns']]

    def get_original_table_name(self, table: str) -> str:
        return self.get_schema_for_table(table)['original_name']

    def get_logical_table_name(self, table: str) -> str:
        return self.get_schema_for_table(table)['logical_name']


def transform(original_schema_path: Path, new_schema_path: Path, tables_of_interest: List[str], columns_of_interest = None, db_id = None): # Add columns_of_interest  if needed
    original_schema = None
    with open(original_schema_path) as f:
        _original_schema = json.load(f)

    # assuming there is only one schema per file
    if db_id is None:
        original_schema = _original_schema[0]
    else:
        for schema in _original_schema:
            if schema['db_id'] == db_id:
                original_schema = schema
                break
    assert isinstance(original_schema, Dict)
    if not any(tables_of_interest): # in case no tables explicitly specified for generative schema, we use all tables
        tables_of_interest = original_schema['table_names_original']

    new_schema = []
    for table_idx, table_original in enumerate(original_schema['table_names_original']):

        # we don't consider all tables but just the one of interest. This normally does not include simple
        # connection tables.
        if table_original not in tables_of_interest:
            continue

        table = {
            "name": original_schema['table_names'][table_idx],
            "original_name": table_original,
            "logical_name": table_original,
            "columns": []
        }

        for column_idx, column in enumerate(original_schema['column_names_original']):
            column_table_idx = column[0]

            # we loop over all columns, but only care about the columns of the current table.
            if table_idx == column_table_idx:
                column = {
                    "name": original_schema['column_names'][column_idx][1],
                    "original_name": column[1],
                    "logical_name": column[1],
                    "original_datatype": original_schema['column_types'][column_idx],
                    "logical_datatype": original_schema['column_types'][column_idx],
                }

                table["columns"].append(column)

        new_schema.append(table)

    # TODO: do we need to add relationship (PK/FK) information?
    if not new_schema_path.exists():
        with open(new_schema_path, 'wt') as out:
            json.dump(new_schema, out, indent=2, )
            print(f"{str(new_schema_path)} generated succsessfully.")
    else:
        print(f"{str(new_schema_path)} exsited. Generating schema interupted.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', default = 'oncomx', type=str)
    
    # parser.add_argument('--original_schema', default='data/cordis/original/tables.json', type=str)
    # parser.add_argument('--new_schema', default='data/cordis/generative/generative_schema.json', type=str)

    args = parser.parse_args()
    original_schema_path = Path('.').joinpath('data', args.db_path, 'original', 'tables.json')
    new_schema_path = Path('.').joinpath('data', args.db_path, 'generative', 'generative_schema.json')
    Path('.').joinpath('data', args.db_path, 'generative').mkdir(parents=True, exist_ok=True)

    # we don't consider all tables but just the one of interest. This normally does not include simple connection tables.
    
    db_path_tables = {
        "cordis": [
            'projects',
            'people',
            'ec_framework_programs',
            'funding_schemes',
            'topics',
            'project_members',
            'subject_areas',
            'programmes',
            'erc_research_domains',
            'project_member_roles',
            'activity_types',
            'countries',
            'eu_territorial_units',
            'institutions'],
    }

    tables = db_path_tables.get(args.db_path, [])

    transform(original_schema_path, new_schema_path, tables)
