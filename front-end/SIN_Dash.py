import pandas as pd
import plotly.graph_objects as go
from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output
import numpy as np
import yfinance as yf

# Pfad zur hochgeladenen CSV-Datei
file_path = './front-end/NAV.csv'

# Einlesen der CSV-Datei ab Zeile 4 und Auswahl der relevanten Spalten
df = pd.read_csv(file_path, skiprows=3, usecols=['Date', 'NAV'])

# Konvertierung des Datumsformats von JJJJMMTT in ein Datetime-Objekt
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

# Hinzufügen eines fiktiven Startpunkts am 31. Dezember des Vorjahres
start_date = df['Date'].min()
start_row = pd.DataFrame({'Date': [start_date], 'NAV': [df['NAV'].iloc[0]]})
df = pd.concat([start_row, df]).reset_index(drop=True)

# Abrufen der historischen Preise des SPY ETFs von Yahoo Finance
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = df['Date'].max().strftime('%Y-%m-%d')
spy_data = yf.download('SPY', start=start_date_str, end=end_date_str)
spy_data.reset_index(inplace=True)

# Hinzufügen eines fiktiven Startpunkts am 31. Dezember des Vorjahres für SPY
spy_start_row = pd.DataFrame({'Date': [start_date], 'Close': [spy_data['Close'].iloc[0]]})
spy_data = pd.concat([spy_start_row, spy_data]).reset_index(drop=True)

# Berechnung der täglichen logarithmischen Renditen der Strategie
df['Log_Return'] = np.log(df['NAV'] / df['NAV'].shift(1))

# Anpassung der Renditen bei Inflows (> 5% Veränderung des NAVs)
df.loc[df['NAV'].pct_change().abs() > 0.05, 'Log_Return'] = 0

# Berechnung der kumulierten logarithmischen Renditen der Strategie
df['Cumulative_Log_Return'] = df['Log_Return'].cumsum()

# Berechnung der täglichen logarithmischen Renditen des SPY ETFs
spy_data['Log_Return'] = np.log(spy_data['Close'] / spy_data['Close'].shift(1))
spy_data['Cumulative_Log_Return'] = spy_data['Log_Return'].cumsum()

# Berechnung des maximalen Drawdowns der Strategie und des SPY ETFs
df['Cumulative_NAV'] = np.exp(df['Cumulative_Log_Return'])
df['Max_Drawdown'] = (df['Cumulative_NAV'] / df['Cumulative_NAV'].cummax()) - 1

spy_data['Cumulative_NAV'] = np.exp(spy_data['Cumulative_Log_Return'])
spy_data['Max_Drawdown'] = (spy_data['Cumulative_NAV'] / spy_data['Cumulative_NAV'].cummax()) - 1

# Erstellen der Dash-App
app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H1('SINTRO Trading Strategy Dashboard', className='dashboard-header'),
    ], className='dashboard-container'),
    
    html.Div([
        dcc.Graph(id='nav-graph', style={'width': '32%', 'display': 'inline-block'}),
        dcc.Graph(id='cumulative-return-graph', style={'width': '32%', 'display': 'inline-block'}),
        dcc.Graph(id='max-drawdown-graph', style={'width': '32%', 'display': 'inline-block'}),
    ]),

    html.Div([
        dash_table.DataTable(
            id='kpi-table',
            columns=[
                {"name": "KPI", "id": "KPI"},
                {"name": "SPY3", "id": "SPY3"},
                {"name": "SPY ETF", "id": "SPY"}
            ],
            style_table={'width': '50%', 'margin': 'auto'},
            style_cell={'textAlign': 'center'},
            style_header={
                'backgroundColor': 'lightgray',
                'fontWeight': 'bold'
            }
        )
    ])
])

@app.callback(
    [Output('nav-graph', 'figure'),
     Output('cumulative-return-graph', 'figure'),
     Output('max-drawdown-graph', 'figure'),
     Output('kpi-table', 'data')],
    [Input('nav-graph', 'relayoutData')]
)
def update_graphs(relayoutData):
    def create_nav_figure(df, title):
        return {
            'data': [
                go.Scatter(x=df['Date'], y=df['NAV'], mode='lines', name='NAV')
            ],
            'layout': go.Layout(title=title, xaxis={'title': 'Date'}, yaxis={'title': 'NAV'})
        }
    
    def create_cumulative_return_figure(df, spy_data, title):
        return {
            'data': [
                go.Scatter(x=df['Date'], y=df['Cumulative_Log_Return'], mode='lines', name='Strategy Cumulative Log Return'),
                go.Scatter(x=spy_data['Date'], y=spy_data['Cumulative_Log_Return'], mode='lines', name='SPY Cumulative Log Return')
            ],
            'layout': go.Layout(title=title, xaxis={'title': 'Date'}, yaxis={'title': 'Cumulative Log Return'})
        }
    
    def create_max_drawdown_figure(df, spy_data, title):
        return {
            'data': [
                go.Scatter(x=df['Date'], y=df['Max_Drawdown'], mode='lines', name='Strategy Max Drawdown'),
                go.Scatter(x=spy_data['Date'], y=spy_data['Max_Drawdown'], mode='lines', name='SPY Max Drawdown')
            ],
            'layout': go.Layout(title=title, xaxis={'title': 'Date'}, yaxis={'title': 'Max Drawdown'})
        }
    
    def calculate_kpis(df, spy_data):
        # KPIs für die Strategie berechnen
        strategy_annual_return = (np.exp(df['Log_Return'].mean() * 252) - 1) * 100
        strategy_annual_vol = df['Log_Return'].std() * np.sqrt(252) * 100
        strategy_total_return = df['Cumulative_Log_Return'].iloc[-2] * 100
        strategy_sharpe_ratio = df['Log_Return'].mean() / df['Log_Return'].std() * (252 ** 0.5)
        strategy_max_drawdown = df['Max_Drawdown'].min() * 100
        strategy_positive_days = (df['Log_Return'] > 0).mean() * 100

        # KPIs für den SPY ETF berechnen
        spy_annual_return = (np.exp(spy_data['Log_Return'].mean() * 252) - 1) * 100
        spy_annual_vol = spy_data['Log_Return'].std() * np.sqrt(252) * 100
        spy_total_return = spy_data['Cumulative_Log_Return'].iloc[-1] * 100
        spy_sharpe_ratio = df['Log_Return'].mean() / df['Log_Return'].std() * (252 ** 0.5)
        spy_max_drawdown = spy_data['Max_Drawdown'].min() * 100
        spy_positive_days = (spy_data['Log_Return'] > 0).mean() * 100

        kpis = [
            {"KPI": "Annualized Return", "SPY3": f"{strategy_annual_return:.2f}%", "SPY": f"{spy_annual_return:.2f}%"},
            {"KPI": "Annualized Volatility", "SPY3": f"{strategy_annual_vol:.2f}%", "SPY": f"{spy_annual_vol:.2f}%"},
            {"KPI": "Sharpe Ratio", "Value": f"{strategy_sharpe_ratio:.2f}", "SPY": f"{spy_sharpe_ratio:.2f}"},
            {"KPI": "Total Return", "SPY3": f"{strategy_total_return:.2f}%", "SPY": f"{spy_total_return:.2f}%"},
            {"KPI": "Max Drawdown", "SPY3": f"{strategy_max_drawdown:.2f}%", "SPY": f"{spy_max_drawdown:.2f}%"},
            {"KPI": "Positive Days %", "SPY3": f"{strategy_positive_days:.2f}%", "SPY": f"{spy_positive_days:.2f}%"}
        ]
        
        return kpis
    
    nav_fig = create_nav_figure(df, 'NAV Over Time')
    cumulative_return_fig = create_cumulative_return_figure(df, spy_data, 'Cumulative Log Return Over Time')
    max_drawdown_fig = create_max_drawdown_figure(df, spy_data, 'Max Drawdown Over Time')
    kpis = calculate_kpis(df, spy_data)
    
    return nav_fig, cumulative_return_fig, max_drawdown_fig, kpis

if __name__ == '__main__':
    app.run_server(debug=True)