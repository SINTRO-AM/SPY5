import pandas as pd
import plotly.graph_objects as go
from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output, State
import numpy as np
import yfinance as yf
from dash.exceptions import PreventUpdate
from graphs.overall_graphs import (
                    create_nav_figure, 
                    create_cumulative_return_figure,
                    create_max_drawdown_figure,
                    calculate_kpis
                    )
from graphs.strategy_graphs import (
                calculate_strategy_kpis
                    )
# Funktion zum Einlesen und Verarbeiten der CSV-Dateien
def process_nav_file(file_path):
    df = pd.read_csv(file_path, skiprows=3)
    
    # Dynamische Überprüfung und Anpassung der Spaltennamen
    date_col = [col for col in df.columns if 'Date' in col or 'Datum' in col][0]
    nav_col = [col for col in df.columns if 'NAV' in col or 'Nav' in col or 'NAV-Wert' in col][0]
    df = df.rename(columns={date_col: 'Date', nav_col: 'NAV'})
    
    # Konvertierung des Datumsformats von JJJJMMTT in ein Datetime-Objekt
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d', errors='coerce')
    
    # Überprüfen und Entfernen von Zeilen mit ungültigen Datumswerten
    df = df.dropna(subset=['Date'])
    
    # Konvertierung der NAV-Spalte in numerische Werte
    df['NAV'] = pd.to_numeric(df['NAV'], errors='coerce')
    
    # Überprüfen und Entfernen von Zeilen mit ungültigen NAV-Werten
    df = df.dropna(subset=['NAV'])
    
    # Verwenden des frühesten Datums der CSV-Datei als Startpunkt
    start_date = df['Date'].min()
    
    # Berechnung der täglichen logarithmischen Renditen
    df['Log_Return'] = np.log(df['NAV'] / df['NAV'].shift(1))
    
    # Anpassung der Renditen bei Inflows (> 5% Veränderung des NAVs)
    df.loc[df['NAV'].pct_change().abs() > 0.05, 'Log_Return'] = 0
    
    # Berechnung der kumulierten logarithmischen Renditen
    df['Cumulative_Log_Return'] = df['Log_Return'].cumsum()
    df['Cumulative_NAV'] = np.exp(df['Cumulative_Log_Return'])
    df['Max_Drawdown'] = (df['Cumulative_NAV'] / df['Cumulative_NAV'].cummax()) - 1
    
    return df

# Pfade zu den CSV-Dateien der Einzelstrategien
file_paths = {
    'Overall': './front-end/NAV.csv',
    'SPY3': './front-end/SPY3.csv',
    'SPY4': './front-end/SPY4.csv',
    'SPY5': './front-end/SPY5.csv'
}

# Einlesen der Daten für die Einzelstrategien
data = {strategy: process_nav_file(file_paths[strategy]) for strategy in file_paths}

# Abrufen der historischen Preise des SPY ETFs von Yahoo Finance
start_date = data['Overall']['Date'].min()
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = data['Overall']['Date'].max().strftime('%Y-%m-%d')
spy_data = yf.download('SPY', start=start_date_str, end=end_date_str)
spy_data.reset_index(inplace=True)

spy_start_row = pd.DataFrame({'Date': [start_date], 'Close': [spy_data['Close'].iloc[0]]})
spy_data = pd.concat([spy_start_row, spy_data]).reset_index(drop=True)
spy_data['Log_Return'] = np.log(spy_data['Close'] / spy_data['Close'].shift(1))
spy_data['Cumulative_Log_Return'] = spy_data['Log_Return'].cumsum()
spy_data['Cumulative_NAV'] = np.exp(spy_data['Cumulative_Log_Return'])
spy_data['Max_Drawdown'] = (spy_data['Cumulative_NAV'] / spy_data['Cumulative_NAV'].cummax()) - 1

# Erstellen der Dash-App
app = Dash(__name__, external_stylesheets=['./front-end/custom.css'])

app.layout = html.Div([
    html.Div([
        html.H1('SINTRO Trading Strategy Dashboard', className='dashboard-header'),
        html.Div([
            html.Button('Aktueller NAV Wert', id='nav-button', className='btn'),
            html.Button('Kumulierte Overall Rendite', id='return-button', className='btn')
        ], className='button-container')
    ], className='dashboard-header-container'),
    
    html.Div([
        html.Div([
            dcc.Graph(id='overall-nav-graph', className='dash-graph'),
            dcc.Graph(id='overall-cumulative-return-graph', className='dash-graph'),
            dcc.Graph(id='overall-max-drawdown-graph', className='dash-graph')
        ], style={'display': 'flex', 'justify-content': 'space-around'}),
    ], className='dashboard-container'),

    html.Div([
        dash_table.DataTable(
            id='overall-kpi-table',
            columns=[
                {"name": "KPI", "id": "KPI"},
                {"name": "Overall", "id": "Overall"},
                {"name": "SPY ETF", "id": "SPY"}
            ],
            style_table={'width': '50%', 'margin': 'auto'},
            style_cell={'textAlign': 'center'},
            style_header={
                'backgroundColor': '#233a53',
                'color': 'white',
                'fontWeight': 'bold'
            }
        )
    ], className='dashboard-container'),

    html.Div(id='strategy-graphs', className='dashboard-container')
], className='dashboard-background')

@app.callback(
    [Output('overall-nav-graph', 'figure'),
     Output('overall-cumulative-return-graph', 'figure'),
     Output('overall-max-drawdown-graph', 'figure'),
     Output('overall-kpi-table', 'data')],
    [Input('overall-nav-graph', 'relayoutData')]
)
def update_overall_graphs(relayoutData):

    
    nav_fig = create_nav_figure(data['Overall'], 'Overall NAV Over Time')
    cumulative_return_fig = create_cumulative_return_figure(data['Overall'], spy_data, 'Overall Cumulative Log Return Over Time')
    max_drawdown_fig = create_max_drawdown_figure(data['Overall'], spy_data, 'Overall Max Drawdown Over Time')
    kpis = calculate_kpis(data['Overall'], spy_data)
    
    return nav_fig, cumulative_return_fig, max_drawdown_fig, kpis

@app.callback(
    Output('strategy-graphs', 'children'),
    [Input('overall-nav-graph', 'relayoutData')]
)
def update_strategy_graphs(relayoutData):


    strategy_layouts = []
    
    for strategy in ['SPY3', 'SPY4', 'SPY5']:
        nav_fig = create_nav_figure(data[strategy], f'{strategy} NAV Over Time')
        cumulative_return_fig = create_cumulative_return_figure(data[strategy], spy_data, f'{strategy} Cumulative Log Return Over Time')
        max_drawdown_fig = create_max_drawdown_figure(data[strategy], spy_data, f'{strategy} Max Drawdown Over Time')
        kpis = calculate_strategy_kpis(data[strategy], strategy, spy_data)
        
        strategy_layouts.append(html.Div([
            html.H2(f'{strategy} Strategy', className='strategy-header'),
            html.Div([
                dcc.Graph(figure=nav_fig, className='dash-graph'),
                dcc.Graph(figure=cumulative_return_fig, className='dash-graph'),
                dcc.Graph(figure=max_drawdown_fig, className='dash-graph')
            ], style={'display': 'flex', 'justify-content': 'space-around'}),
            dash_table.DataTable(
                columns=[
                    {"name": "KPI", "id": "KPI"},
                    {"name": strategy, "id": strategy},
                    {"name": "SPY ETF", "id": "SPY"}
                ],
                data=kpis,
                style_table={'width': '50%', 'margin': 'auto'},
                style_cell={'textAlign': 'center'},
                style_header={
                    'backgroundColor': '#233a53',
                    'color': 'white',
                    'fontWeight': 'bold'
                }
            )
        ], className='strategy-container'))
    
    return strategy_layouts

@app.callback(
    Output('nav-button', 'children'),
    [Input('nav-button', 'n_clicks')]
)
def update_nav_button(n_clicks):
    if n_clicks is None:
        raise PreventUpdate

    total_nav = sum([data[strategy]['NAV'].iloc[-1] for strategy in ['SPY3', 'SPY4', 'SPY5']])
    return f'Aktueller NAV Wert: {total_nav:.2f} EUR'

@app.callback(
    Output('return-button', 'children'),
    Output('return-button', 'style'),
    [Input('return-button', 'n_clicks')]
)
def update_return_button(n_clicks):
    if n_clicks is None:
        raise PreventUpdate

    overall_return = (data['Overall']['Cumulative_NAV'].iloc[-1] - 1) * 100
    color = 'darkgreen' if overall_return > 0 else 'darkred'
    return f'Kumulierte Overall Rendite: {overall_return:.2f}%', {'backgroundColor': color, 'color': 'white'}


if __name__ == '__main__':
    app.run_server(debug=True)
