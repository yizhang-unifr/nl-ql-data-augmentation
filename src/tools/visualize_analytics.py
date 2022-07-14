from pathlib import Path
from tqdm import tqdm
from dash import Dash, html, dcc, Input, Output, dash_table
import pandas as pd
import plotly.express as px
import argparse

def load_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument('--data_path', default=None, help='path to input data, if not set, data/<dataset>/original/analysis.xlsx will be used')
    parser.add_argument('--dataset', default='spider')
    return parser.parse_args()

def load_data(data_path):
    with pd.ExcelFile(data_path) as f_in:
        total_df = pd.read_excel(f_in, sheet_name='detailed_total')
        train_db_df = pd.read_excel(f_in, sheet_name='db_train')
        dev_db_df = pd.read_excel(f_in, sheet_name='db_dev')
    return total_df, train_db_df, dev_db_df

def run_app(data_path):
    total_res, train_db_res, dev_db_res = load_data(data_path)
    total_db_res = pd.concat([train_db_res, dev_db_res])

    # external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    external_stylesheets = []
    fig = px.histogram(total_db_res, x = 'db_id', y='occurance', color='dataset', hover_data=total_db_res.columns).update_layout(
    xaxis_title="Database ID", yaxis_title="Counts of reference in dataset")
    data = total_res.to_dict('records')

    app = Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div([
        
        html.Div([
            dcc.Graph(
                id='total-histogram',
                figure=fig
            )
        ], style = {'width': '100%', 'display': 'inline-block', 'padding': '0 20'}),

        html.Div([
            dcc.Graph(
                id='hardness-pie',
                figure={}
            )
        ], style = {'width': '39%', 'display': 'inline-block', 'padding': '0 20'}),

        html.Div([
            dash_table.DataTable(columns=[
        {'name': 'db_id', 'id': 'db_id', 'type': 'text'},
        {'name': 'table_name', 'id': 'table_name', 'type': 'text'},
        {'name': 'column_name', 'id': 'column_name', 'type': 'text'},
        {'name': 'column_type', 'id': 'column_type', 'type': 'text'},
        {'name': 'is_key', 'id': 'is_key', 'type': 'numeric'},
        {'name': 'frequency', 'id': 'frequency', 'type': 'numeric', 'format': {'specifier': '.2f'}},
        {'name': 'dataset', 'id': 'dataset', 'type': 'text'}
        ],
        data = data,
        sort_action="native",
        sort_mode="multi",
        filter_action='native',
        page_action="native",
        page_current= 0,
        page_size= 10,

        style_table={
            'height': 400,
        },
        style_data={
            'minWidth': '10px', 'maxWidth': '150px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
        },
        id='data-table'),
        ], style = {'width': '59%', 'display': 'inline-block', 'padding': '0 20 0 20', "float": "right"}),

    ])

    @app.callback(
        Output('hardness-pie', 'figure'),
        Output('data-table', 'data'),
        Input('total-histogram', 'clickData')
    )
    def event_click_data(clickData):
        if clickData:
            db_id = clickData['points'][0]['x']
            dataset = 'train' if clickData['points'][0]['curveNumber'] == 0 else 'dev'
            train_df = total_db_res[(total_db_res['db_id']==db_id) & (total_db_res['dataset']== dataset)]
            temp_df = pd.DataFrame({'SPIDER Hardness': train_df.keys()[2:6], 'y': train_df.values[0][2:6]})
            fig = px.histogram(temp_df, x = 'SPIDER Hardness', y = 'y', color='SPIDER Hardness').update_layout(
    xaxis_title="SPIDER Hardness", yaxis_title="Counts in data samples")
            # fig = px.pie(temp_df,  values = 'y', color='x')

            # update data table
            data_tab_df = total_res[(total_res['db_id'] == db_id) & (total_res['dataset'] == dataset)]
            # print(data_tab_df)
            data = data_tab_df.to_dict('records')
        else:
            fig = {}
            data = total_res.to_dict('records')
        return fig, data
    app.run_server(debug=True)

def main():
    args= load_args()
    root_path = Path.cwd()
    dataset_path = root_path / 'data' / args.dataset
    data_path = args.data_path
    if not data_path:
        data_path = dataset_path / 'original' / 'analytics.xlsx'
    
    if Path(data_path).exists():
        run_app(data_path)
    else:
        print(f"input data_path {data_path} doesn't exist")
        return
    
if __name__ == '__main__':
    main()