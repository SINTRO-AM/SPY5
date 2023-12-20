# Import necessary libraries
import dash
from dash import html, dcc, Input, Output, dash_table
import plotly.express as px
import pandas as pd
import numpy as np
import datetime as dt
import os

# selfwritten functions
from SPY3_functions import start_end_date, create_performance_graph, fetch_and_process_data, get_clicked_button_id, create_price_and_var_graph, create_results_df, current_results_df, Alpha,maxDD


# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.Div(
    children=[
        html.H1(
            children='SPY3 Performance Dashboard',
            className='dashboard-header'
                )
            ],
            className='dashboard-container'
            ),
    html.Div([
        dcc.Dropdown(
            id='index-selector',
            options=[
                {'label': 'S&P 500 Index (SPY)', 'value': 'SPY'},
                {'label': 'DAX Index', 'value': '^GDAXI'},
                {'label': 'NASDAQ Index', 'value': '^NDX'},
                ],
                value='SPY',  # Default value
            className='styled-dropdown'
            ),
            # ... rest of your components
        ], className='your-custom-class-name'),

    html.Div([ 
            html.Div([
                html.Button('Live', id='btn-Live', n_clicks=0, className='time-filter-btn'),
                html.Button('YtD', id='btn-YtD', n_clicks=0, className='time-filter-btn'),
                html.Button('1 Jahr', id='btn-1yr', n_clicks=0, className='time-filter-btn'),
                html.Button('2 Jahre', id='btn-2yr', n_clicks=0, className='time-filter-btn'),
                html.Button('4 Jahre', id='btn-4yr', n_clicks=0, className='time-filter-btn'),
                html.Button('5 Jahre', id='btn-5yr', n_clicks=0, className='time-filter-btn'),
                html.Button('Reset', id='btn-max', n_clicks=0, className='time-filter-btn')
            ], className='button-container'),    
        html.Div([
            html.Div([dcc.Graph(id='performance-graph',style={'height': '75vh'})], className='graph-container'),
            html.Div([
                dash_table.DataTable(
                    id='table',
                    style_header={
                        'backgroundColor': '#0b1620',
                        'color': '#d9e0e8',
                        'fontWeight': 'bold',
                        'fontFamily': 'Segoe UI'
                    },
                    style_cell={
                        'textAlign': 'center',
                        'color': 'black',
                        'fontFamily': 'Segoe UI',
                         'marginBottom': '20px'
                    },
                    ),
                html.Div(style={'height': '20px'}),  
                
                dash_table.DataTable(
                    id='current-results-table',
                    style_header={
                        'backgroundColor': '#0b1620',
                        'color': '#d9e0e8',
                        'fontWeight': 'bold',
                        'fontFamily': 'Segoe UI'
                    },
                    style_cell={
                        'textAlign': 'center',
                        'color': 'black',
                        'fontFamily': 'Segoe UI',
                        'marginBottom': '20px'
                    }
                ),
                    html.Div(style={'height': '20px'}), 
                    dcc.Graph(id='Alpha',style={'height': '20vh'}),

                    html.Div(style={'height': '20px'}), 

                dcc.Graph(id='maxDD',style={'height': '22vh'}),
            ], className='table-container'),
        ], className='row-flex'),
    ], className='performance_container'),

    html.Div([ 


    dcc.Graph(id='price-and-var-graph', style={'height': '90vh'}),
])])

# Define the callback
@app.callback(
    [Output('performance-graph', 'figure'),
     Output('price-and-var-graph', 'figure'),
     Output('table', 'data'),  # Daten für die DataTable
     Output('table', 'columns'),
     Output('current-results-table', 'data'),  # Daten für die neue DataTable
     Output('current-results-table', 'columns'),
     Output('Alpha', 'figure'),
     Output('maxDD', 'figure')]     ,
    [Input('index-selector', 'value'),
     Input('btn-Live', 'n_clicks'),
     Input('btn-YtD', 'n_clicks'),
     Input('btn-1yr', 'n_clicks'),
     Input('btn-2yr', 'n_clicks'),
     Input('btn-4yr', 'n_clicks'),
     Input('btn-5yr', 'n_clicks'),
     Input('btn-max', 'n_clicks')]
)

def update_graph(selected_index,btn_1yr, btn_2yr, btn_4yr, btn_5yr,btn_YtD, btn_max, Live):
    # Get the ID of the clicked button
    button_id = get_clicked_button_id()

    start_date = start_end_date(button_id)
    # Sample data processing (replace with actual logic)
    df = fetch_and_process_data(selected_index, start_date)

   # Create a Plotly figure for each graph
    figure_performance = create_performance_graph(df)
    figure_prices_and_var = create_price_and_var_graph(df)
    results_data, results_columns = create_results_df(df)

    current_results_data, current_results_columns = current_results_df(df)
    Alpha_figure = Alpha(df)
    max_dd_figure = maxDD(df)
    return [figure_performance,figure_prices_and_var,results_data, results_columns, current_results_data, current_results_columns, Alpha_figure, max_dd_figure]

# Run the app
if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
    server = app.server