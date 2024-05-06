from pathlib import Path
import json
import pandas as pd
import argparse

def load_parser():
    parser = argparse.ArgumentParser(description="We use this file to analyze all possible cross-join possibility")
    parser.add_argument('--dataset_name', type=str, required=True, help='The dataset name must be the same as the corresponding subfolder name under data folder')
    parser.add_argument('--input_path', type=str, default='data_aug', help='The input file path')
    parser.add_argument('--output_path', type=str, default='data_aug', help='The output file path')


    return parser

def reduce_questions(d):
    questions = d['questions']
    if any(questions):
        temp, res = [], []
        for question in questions:
            if question not in temp:
                temp.append(question)
                res.append({
                    'db_id': d['db_id'],
                    'hops': d['hops'],
                    'id': d['id'],
                    'question': question,
                    'query': d['query'],
                })
        d['questions'] = temp
        return d, res
    raise ValueError('No questions found')

def reduce_data(data):
    _data, res_list = [], []
    for d in data:
        _d, _data_list = reduce_questions(d)
        _data.append(_d)
        res_list += _data_list
    return _data, res_list

def main():
    parser = load_parser()
    args = parser.parse_args()
    dataset_name = args.dataset_name
    input_path = args.input_path
    root_path = Path.cwd()
    data_path = Path(root_path / 'data')
    input_path = Path(data_path / dataset_name / args.input_path / f'generated_{dataset_name}_no_labels.json')
    output_path_json = Path(data_path / dataset_name / args.output_path / f'compact_{dataset_name}_no_labels.json')
    output_path_json_flat = Path(data_path / dataset_name / args.output_path / f'flat_{dataset_name}_no_labels.json')
    output_path_xls = Path(data_path / dataset_name / args.output_path / f'reduced_{dataset_name}_no_labels.xlsx')
    with open(input_path, 'r') as f_in:
        data = json.load(f_in)
    res_1, res_2 = reduce_data(data)
    len_1, len_2 = sum([len(ele['questions']) for ele in res_1]), len(res_2)
    assert len_1 == len_2, f'len_1: {len_1}, len_2: {len_2}'
    if not output_path_xls.exists():
        df_compact = pd.DataFrame(res_1)
        # need to process the questions to multii-line texts
        df_flat = pd.DataFrame(res_2)
        writer = pd.ExcelWriter(output_path_xls)
        df_compact['questions'] = df_compact.apply(lambda row: '\n'.join(row['questions']), axis=1)
        df_compact.to_excel(writer, sheet_name = 'compact', index=False)
        df_flat.to_excel(writer, sheet_name = 'flat', index=False)
        # add more sheets for different hops
        max_hops = max(df_flat['hops'])
        for i in range(1, max_hops+1):
            hops_filter = df_flat['hops'] == i
            df_flat[hops_filter].to_excel(writer, sheet_name = f'hops={i}', index=False)
        writer.save()
    if not (output_path_json.exists() and output_path_json_flat.exists()):
        with open(output_path_json, 'w') as f_out_compact, open(output_path_json_flat, 'w') as f_out_flat:
            json.dump(res_1, f_out_compact, indent=2, sort_keys=True)
            json.dump(res_2, f_out_flat, indent=2, sort_keys=True)

if __name__ == '__main__':
    main()