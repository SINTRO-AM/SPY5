from dash import Dash, html, dcc, callback, Output, Input, dash_table
import yfinance as yf
import datetime as dt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
import datetime
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, clear_output
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import flask

#Input
start_date = dt.datetime(2000, 1, 1)
start_dateB = dt.datetime(2000, 1, 1)
end_date = dt.datetime.today()

spx = yf.Ticker("SPY")
spx_hist = spx.history(start=start_date, end=end_date)
ust = yf.Ticker("SHY")
ust_hist = ust.history(start=start_dateB, end=end_date)

df = pd.DataFrame({'Close': spx_hist['Close']})
df['CloseB'] = ust_hist['Close']
df['Return'] = np.log(df['Close'] / df['Close'].shift(1))
df['ReturnB'] = np.log(df['CloseB'] / df['CloseB'].shift(1))

# überprüfe, ob es leere Zellen in "CloseB" gibt
if df['CloseB'].isnull().values.any():
    # Fülle leere Zellen in "ReturnB" mit 0
    df['ReturnB'].fillna(0, inplace=True)

# Berechne den 30-Tage-gleitenden Durchschnitt
df['30D_MA'] = df['Close'].rolling(window=30).mean()
df['200D_MA'] = df['Close'].rolling(window=198).mean()

# Berechne die taegliche Standardabweichung ueber die letzten 50 Tage
df['STD'] = df['Return'].rolling(window=50).std()

# Berechne die taegliche logarithmische Rendite
df['Return'] = np.log(df['Close'] / df['Close'].shift(1))

# Berechne die annualisierte Standardabweichung
df['50d_STD'] = df['STD'] * np.sqrt(252)

# Berechne das taegliche VaR auf Basis der letzten 50 returns
df['VaR_1d'] = -df['50d_STD'] * norm.ppf(0.99) * np.sqrt(1/252)*-1

# Berechne die kumulierte Rendite des SPX
df['SPX_Total_Return'] = (df['Return']).cumsum()
df['UST_Total_Return'] = (df['ReturnB']).cumsum()

# Signal
conditions = (df['VaR_1d'] < 0.05) & ((df['30D_MA'] > df['200D_MA']) | (df['VaR_1d'] < 0.02) | (df['Close'] < df['Close'].rolling(window=200).max()/1.3))
df.loc[conditions, 'Signal'] = 1
df['Signal'].fillna(0, inplace=True) 

# SignalB
df.loc[df['Signal'] == 0, 'SignalB'] = 1
df.loc[df['Signal'] == 1, 'SignalB'] = 0

# Time-lag & transaction costs
df['Portfolio_Return'] = df['Return'] * df['Signal'].shift(1) + df['ReturnB'] * df['SignalB'].shift(1)
previous_signal = df['Signal'].shift(2)
df.loc[df['Signal'] != previous_signal, 'Portfolio_Return'] -= 0.0001  # transaction costs of 1 basis points (0.0001)
df['Total_Return'] = (df['Portfolio_Return']).cumsum()
df = df.reset_index()
df.Date = pd.to_datetime(df.Date)
df.Date = df.Date.dt.date

# Definition of minor charts
df['Alpha'] = (df['Portfolio_Return']) - (df['Return'])
df['Alpha_cum'] = (df['Alpha']).cumsum()
df['DD_SPY3'] = df['Total_Return'] - df['Total_Return'].cummax()
df['DD_SPY'] = df['SPX_Total_Return'] - df['SPX_Total_Return'].cummax()

# Calculate rolling performance (cumulative sum of returns)
df['Rolling_SPY'] = df['Return'].rolling(window=1260).sum()/5
df['Rolling_SPY3'] = df['Portfolio_Return'].rolling(window=1260).sum()/5

# Calculate rolling volatility (standard deviation of returns)
df['Rolling_SPY_Vol'] = df['Return'].rolling(window=252).std()
df['Rolling_SPY3_Vol'] = df['Portfolio_Return'].rolling(window=252).std()

date_today = datetime.datetime.now().date()

Result = pd.DataFrame()
Result['KPI'] = ['Total Return', 'Sharpe Ratio', 'Annualized Vol']
Result['Portfolio']  = df['Portfolio_Return']

# Calculations 
annual_return_bmk = round(np.mean(df['Return']) * 252 *100, 2)
annual_std_bmk = round(np.std(df['Return']) * np.sqrt(252)*100, 2)
sharpe_ratio_bmk = round(annual_return_bmk / annual_std_bmk,2)

annual_return_pf = round(np.mean(df['Portfolio_Return']) * 252 *100, 2)
annual_std_pf = round(np.std(df['Portfolio_Return']) * np.sqrt(252)*100, 2)
df['Rolling_Annual_Return_SPY'] = df['Return'].rolling(window=756).sum()/3
df['Rolling_Annual_Return_SPY3'] = df['Portfolio_Return'].rolling(window=756).sum()/3
df['Rolling_Max_DD_SPY3'] = (df['Total_Return'] - df['Total_Return'].cummax()).rolling(window=756).min()
df['Rolling_Max_DD_SPY'] = (df['SPX_Total_Return'] - df['SPX_Total_Return'].cummax()).rolling(window=756).min()
df['Rolling_Calmar_SPY'] = df['Rolling_Annual_Return_SPY'] / abs(df['Rolling_Max_DD_SPY'])
df['Rolling_Calmar_SPY3'] = df['Rolling_Annual_Return_SPY3'] / abs(df['Rolling_Max_DD_SPY3'])


sharpe_ratio_pf = round(annual_return_pf / annual_std_pf, 2)
max_DD_SPY3 = round(df['DD_SPY3'].min()*100,2)
max_DD_SPY = round(df['DD_SPY'].min()*100,2)
calmar_ratio_spy = annual_return_bmk / abs(max_DD_SPY)
calmar_ratio_spy3 = annual_return_pf / abs(max_DD_SPY3)

current_var_1d = round(df['VaR_1d'].iloc[-1],5)
df['Rolling_200D_High_Discount'] = df['Close'].rolling(window=200).max() / 1.3
p_delta = 1.3 * df['Close'].iloc[-1] / df['Close'].rolling(window=200).max().iloc[-1] -1 

# Signal und Faktoren für die Buttons abrufen
current_signal = df['Signal'].iloc[-1]
current_var_1d = round(df['VaR_1d'].iloc[-1],4)
current_30d_ma = df['30D_MA'].iloc[-1]
current_200d_ma = df['200D_MA'].iloc[-1]
ma_delta = current_30d_ma - current_200d_ma

# TEST: Probability of Signal Change
df['Target'] = (df['Signal'] != df['Signal'].shift(-20)).astype(int)
# 2. Eingabemerkmale definieren
X = df[['30D_MA', '200D_MA', 'VaR_1d', 'Rolling_200D_High_Discount']]
y = df['Target']
# Datenbereinigung: Entfernen von NaN Werten
X = X.dropna()
y = y[X.index]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 4. Training des Modells
clf = LogisticRegression()
clf.fit(X_train, y_train)
# 5. Vorhersagen und Evaluierung
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
# 6. Vorhersage für die Zukunft basierend auf aktuellen Werten
current_values = df[['30D_MA', '200D_MA', 'VaR_1d', 'Rolling_200D_High_Discount']].iloc[-1].values.reshape(1, -1)
probability_switch = clf.predict_proba(current_values)[0][1]
print(f"Probability of a signal change within the next 30 days: {probability_switch * 100:.2f}%")


# Calculate the 5y rolling annualized Sharpe Ratio for the portfolio and the benchmark (1y = 252 trading days)
df['Rolling_SPY_Sharpe'] = (df['Return'].rolling(window=1260).mean()*252) / (df['Return'].rolling(window=1260).std()*np.sqrt(252))
df['Rolling_SPY3_Sharpe'] = (df['Portfolio_Return'].rolling(window=1260).mean()*252) / (df['Portfolio_Return'].rolling(window=1260).std()*np.sqrt(252))

results_df = pd.DataFrame({
    'Metric': ['Annual Return in %', 'Annual STD in %', 'Sharpe Ratio','max DD in %'],
    'SPY3': [annual_return_pf, annual_std_pf, sharpe_ratio_pf, max_DD_SPY3],
    'SPY': [annual_return_bmk, annual_std_bmk, sharpe_ratio_bmk, max_DD_SPY]
})

# Calculate Monthly Returns
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.to_period('M')
monthly_returns_spy = df.groupby('Month')['Return'].sum()
monthly_returns_spy3 = df.groupby('Month')['Portfolio_Return'].sum()
monthly_returns_table = pd.DataFrame({
    'Month': monthly_returns_spy.index.astype(str),
    'SPY Monthly Returns': ["{:.2f}%".format(val * 100) for val in monthly_returns_spy.values],
    'SPY3 Monthly Returns': ["{:.2f}%".format(val * 100) for val in monthly_returns_spy3.values]
})

# Calculate Annual Returns
df['Year'] = df['Date'].dt.to_period('Y')
annual_returns_spy = df.groupby('Year')['Return'].sum()
annual_returns_spy3 = df.groupby('Year')['Portfolio_Return'].sum()
annual_returns_table = pd.DataFrame({
    'Year': annual_returns_spy.index.astype(str),
    'SPY Annual Returns': ["{:.2f}%".format(val * 100) for val in annual_returns_spy.values],
    'SPY3 Annual Returns': ["{:.2f}%".format(val * 100) for val in annual_returns_spy3.values]
})

# Create a bar chart with Plotly
annual_bar = go.Figure()
annual_bar.add_trace(go.Bar(
    x=annual_returns_table['Year'],
    y=annual_returns_table['SPY3 Annual Returns'].str.rstrip('%').astype('float'),
    name='SPY3 Annual Returns',
    marker_color='steelblue'  # Farbe für SPY3 Balken
))
annual_bar.add_trace(go.Bar(
    x=annual_returns_table['Year'],
    y=annual_returns_table['SPY Annual Returns'].str.rstrip('%').astype('float'),
    name='SPY Annual Returns',
    marker_color='lightgray'))

annual_bar.update_layout(
     font=dict(family="Segoe UI"),paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)' ,legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

def create_heatmap(data, title):
    """Erstellt eine Plotly Heatmap aus den bereitgestellten Daten."""
    heatmap = sns.heatmap(data, cmap="RdYlGn", annot=True, fmt=".2%")
    heatmap.get_figure().clf()
    return go.Figure(data=[
        go.Heatmap(
            z=data.values,
            x=data.columns,
            y=data.index,
            colorscale='RdYlGn',
            zmin=-0.12,
            zmax=0.12
        )
    ],
        layout=go.Layout( font=dict(family="Segoe UI"),title=title, xaxis=dict(title='Jahr',side='top', dtick=1), yaxis=dict(title='Month', autorange='reversed', dtick=1)))


heatmap_data_spy = df.pivot_table(index=df['Date'].dt.month, columns=df['Date'].dt.year, values='Return', aggfunc='sum')
heatmap_data_spy3 = df.pivot_table(index=df['Date'].dt.month, columns=df['Date'].dt.year, values='Portfolio_Return', aggfunc='sum')

heatmap_spy = create_heatmap(heatmap_data_spy, "Monthly Returns of the S&P500 Index")
heatmap_spy3 = create_heatmap(heatmap_data_spy3, "Monthly Returns of the SPY3 Model")

# Adjust the layout
annual_bar.update_layout(font=dict(family="Segoe UI"), title='Annual Returns in %', barmode='group')

external_stylesheets = ['./assets/custom.css']
app = Dash(_name_, external_stylesheets=external_stylesheets)
server = app.server

def determine_button_color(signal_value):
    return 'green' if signal_value == 1 else 'red'

def determine_button_label(signal_value):
    return 'Risk On' if signal_value == 1 else 'Risk Off'

def determine_risk_button_color(current_var_1d):
    if current_var_1d < 0.02:
        return 'green'
    elif 0.02 <= current_var_1d <= 0.05:
        return 'orange'
    else:
        return 'red'
    
def determine_button_color_class(current_signal):
    """
    Determines the button color class based on the current signal.
    
    Args:
    current_signal (str): The current signal that determines the button color.
    
    Returns:
    str: The class name corresponding to the current signal.
    """
    if current_signal == 'active':
        return 'green'  # This class would be defined with a green background in your CSS
    elif current_signal == 'inactive':
        return 'red'  # This class would be defined with a red background in your CSS
    else:
        return 'default'
    
    

def determine_risk_button_label(current_var_1d):
    if current_var_1d > 0.05:
        return 'High Volatility'
    elif 0.02 <= current_var_1d <= 0.05:
        return 'Neutral'
    else:
        return 'Low Market Risk'

def determine_momentum_button_color(current_30d_ma, current_200d_ma):
    return 'green' if current_30d_ma > current_200d_ma else 'orange'

def determine_momentum_button_label(current_30d_ma, current_200d_ma):
    return 'Trend-Following' if current_30d_ma > current_200d_ma else 'Counter-Trending'

def determine_meanreversion_button_color(p_delta):
    return 'green' if p_delta < 0 else 'Grey'

def determine_meanreversion_button_label(p_delta):
    return 'Mean-Reversion expected' if p_delta < 0 else 'Neutral'

def generate_shapes_for_signals(df):
    shapes = []
    start_date = None
    in_shape = False

    for i, row in df.iterrows():
        if row['Signal'] == 0 and not in_shape:
            start_date = row['Date']
            in_shape = True
        elif (row['Signal'] == 1 or i == len(df) - 1) and in_shape:
            end_date = row['Date']
            in_shape = False
            shapes.append({
                'type': 'rect',
                'xref': 'x',
                'yref': 'paper',
                'x0': start_date,
                'x1': end_date,
                'y0': 0,
                'y1': 1,
                'fillcolor': 'grey',
                'name': 'Risk Off',
                'opacity': 0.25,
                'line': {'width': 0},
            })

    return shapes

shapes = generate_shapes_for_signals(df)

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
        html.Div([  # Container for the Buttons and the Performance Graph

            # Central Container Div for the Buttons
            html.Div([

                # Signal Button
                html.Button(
                determine_button_label(current_signal), 
                id='signal-button', 
                className='signal-button ' + determine_button_color_class(current_signal)),

                # Risk Button
                html.Button(
                "{} (VaR = {})".format(determine_risk_button_label(current_var_1d), round(current_var_1d*100,4)), 
                id='risk-button', 
                style={
                                'background-color': determine_risk_button_color(current_var_1d),
                                'color': 'white',
                                'border': 'none',
                                'padding': '10px 20px',
                                'margin-right': '10px',
                                'fontWeight': 'bold',
                                'font-family': 'Segoe UI',
                                'display': 'inline-block'
                            }),

                # Momentum Button
            html.Button(
                "{} (SMA_Delta in USD = {:.2f})".format(determine_momentum_button_label(current_30d_ma, current_200d_ma), ma_delta), 
                id='momentum-button', 
                style={
                    'background-color': determine_momentum_button_color(current_30d_ma, current_200d_ma),
                    'color': 'white',
                    'border': 'none',
                    'padding': '10px 20px',
                    'margin-right': '10px',
                    'fontWeight': 'bold',
                    'font-family': 'Segoe UI',
                    'display': 'inline-block'
                }
            ),
              # Mean Reversion Button
            html.Button(
                "{} (MR_Delta in % = {:.2f})".format(determine_meanreversion_button_label(p_delta), p_delta), 
                id='MR-button', 
                style={
                    'background-color': determine_meanreversion_button_color(p_delta),
                    'color': 'white',
                    'border': 'none',
                    'padding': '10px 20px',
                    'fontWeight': 'bold',
                    'font-family': 'Segoe UI',
                    'display': 'inline-block'
                }
            ),
            ], style={'textAlign': 'center'}),  # This centralizes the button group

            dcc.Graph(id='performance-graph', style={'height': '90vh', 'clear': 'both'}),
            dcc.Graph(id='price-and-var-graph', style={'height': '90vh'}),

             # Annual Returns Table
        html.H4(children='Annual Returns', style={'textAlign': 'center','fontFamily': 'Segoe UI'}),
        dash_table.DataTable(
            id='annual-returns-table',
            columns=[{"name": i, "id": i} for i in annual_returns_table.columns],
            data=annual_returns_table.to_dict('records'),
            style_table={'height': '160px', 'overflowY': 'auto'},
            style_cell={'textAlign': 'center'},
            style_header={
                'backgroundColor': 'lightgray',
                'font-family': 'Segoe UI',
                'fontWeight': 'bold'
            }
        ),

        # Monthly Returns Table
        html.H4(children='Monthly Returns', style={'textAlign': 'center','fontFamily': 'Segoe UI'}),
        dash_table.DataTable(
            id='monthly-returns-table',
            columns=[{"name": i, "id": i} for i in monthly_returns_table.columns],
            data=monthly_returns_table.to_dict('records'),
            style_table={'height': '160px', 'overflowY': 'auto'},
            style_cell={'textAlign': 'center'},
            style_header={
                'backgroundColor': 'lightgray',
                'font-family': 'Segoe UI',
                'fontWeight': 'bold'
            }
        ),
            html.Div([
        dcc.Graph(figure=annual_bar)
    ]),
    html.Div([
        dcc.Graph(figure=heatmap_spy3),
        dcc.Graph(figure=heatmap_spy)
    ]),

        ], style={'width': '62%', 'display': 'inline-block', 'vertical-align': 'top'}),

        html.Div([  # Container for the Metrics Table, Alpha and Drawdown Graphs
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in results_df.columns],
                data=results_df.to_dict('records'),
                style_table={'height': '160px', 'overflowY': 'auto'},
                style_cell={'textAlign': 'center'},
                style_header={
                    'backgroundColor': 'lightgray',
                    'font-family': 'Segoe UI',
                    'fontWeight': 'bold'
                }
            ),
            dcc.Graph(id='alpha-time-series', style={'height': '40vh'}),
            dcc.Graph(id='drawdown-comparison', style={'height': '35vh'}),
            dcc.Graph(id='divergence', style={'height': '35vh'}),
            dcc.Graph(id='rolling-sharpe-ratio', style={'height': '50vh'}),
            dcc.Graph(id='rolling-performance', style={'height': '50vh'}),
            dcc.Graph(id='rolling-volatility', style={'height': '50vh'}),
            dcc.Graph(id='calmar-ratio-graph', style={'height': '50vh'}
    ),    
        ], style={'width': '38%', 'display': 'inline-block', 'vertical-align': 'top'})
    ])
])

@app.callback(
     [Output('performance-graph', 'figure'),
     Output('price-and-var-graph', 'figure'),
     Output('alpha-time-series', 'figure'),
     Output('drawdown-comparison', 'figure'),
     Output('divergence', 'figure'),
     Output('rolling-sharpe-ratio', 'figure'),
     Output('rolling-performance', 'figure'),
     Output('rolling-volatility', 'figure'),
     Output('calmar-ratio-graph', 'figure')],
    Input('performance-graph', 'relayoutData')
)
def update_graph(relayoutData):
    # Performance graph
    trace1 = go.Scatter(x=df['Date'], y=df['SPX_Total_Return'], mode='lines', name='S&P500 Total Return',line=dict(color="grey",width=2))
    trace2 = go.Scatter(x=df['Date'], y=df['Total_Return'], mode='lines', name='SPY3 Total Return',line=dict(color="#4472C4",width=2))
    trace3 = go.Scatter(x=df['Date'], y=df['UST_Total_Return'], mode='lines', name='US Treasury Total Return')
    layout_performance = go.Layout(title='Performance Overview: SPY3 vs. SPX',
                                   xaxis=dict(title='Date', range=[min(df['Date']), max(df['Date'])],dtick="M24"),
                                   yaxis=dict(title='Cumulative Return'),
                                   legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))  # Adjusted legend
    # New graph for Close prices, Moving averages, and VaR
    trace_spy_close = go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='SPY Close')
    trace_shy_close = go.Scatter(x=df['Date'], y=df['CloseB'], mode='lines', name='US Treasuries')
    trace_30d_ma = go.Scatter(x=df['Date'], y=df['30D_MA'], mode='lines', name='30 Day MA', line=dict(dash='dashdot'))
    trace_200d_ma = go.Scatter(x=df['Date'], y=df['200D_MA'], mode='lines', name='200 Day MA', line=dict(dash='dashdot'))
    trace_200d_high_discount = go.Scatter(x=df['Date'], y=df['Rolling_200D_High_Discount'], mode='lines', name='Mean-Reversion Threshold', line=dict(dash='dash', color="orange"))

# VaR with a second y-axis
    trace_var = go.Scatter(x=df['Date'], y=df['VaR_1d'], mode='lines', name='VaR', yaxis='y2',line=dict(color="#E088B0"))
    
    # VaR Thresholds
    green_line = {
        'type': 'line','xref': 'paper','yref': 'y2','x0': 0,'x1': 1,'y0': 0.02,'y1': 0.02,'line': {
            'color': 'green',
            'width': 1,
            'dash': 'dash'}
     }
    red_line = {
        'type': 'line', 'xref': 'paper','yref': 'y2','x0': 0,'x1': 1,'y0': 0.05,'y1': 0.05,'line': {'color':'red','width': 1, 'dash': 'dash' }
    }
    # Add the new lines to your existing shapes
    shapes.extend([green_line, red_line])
    
    #Inside the Spy3 model with all indicators (except, yet, of Mean Reversion factor)
    layout_prices_and_var = go.Layout(title='Inside the SPY3 Model: Prices, Moving Averages and VaR',
                                 xaxis=dict(title='Date',dtick="M24"),
                                 yaxis=dict(title='Price'),
                                 yaxis2=dict(title='VaR', overlaying='y', side='right'),
                                 legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)) 
    layout_prices_and_var['shapes'] = shapes  
    # Alpha time-series
    trace_alpha = go.Scatter(x=df['Date'], y=df['Alpha_cum'], mode='lines', name='Alpha', line=dict(color="#4D79C7", width=2), fill='tozeroy', fillcolor='rgba(77, 121, 199, 0.35)')
    layout_alpha = go.Layout(title='Excess Return over Time', xaxis=dict(title='Date'), yaxis=dict(title='Alpha'))
    
    # Drawdown comparison 
    trace_dd_spy3 = go.Scatter(x=df['Date'], y=df['DD_SPY3'], mode='lines', name='Max Drawdown SPY3',line=dict(color="#4472C4",width=2))
    trace_dd_spy = go.Scatter(x=df['Date'], y=df['DD_SPY'], mode='lines', name='Max Drawdown SPY',line=dict(color="grey",width=2))
    layout_dd = go.Layout(title='Maximum Drawdown Comparison: SPY3 vs. SPY', 
                          xaxis=dict(title='Date'),
                          yaxis=dict(title='Max Drawdown'),
                          legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))  # Adjusted legend

    # Divergence time-series (not really important, nur eine Spielerei)
    trace_divergence = go.Scatter(x=df['Date'], y=df['Alpha'], mode='lines', name='Divergence')
    layout_divergence = go.Layout(title='Divergence over Time', xaxis=dict(title='Date'), yaxis=dict(title='BMK Delta'))

    # Rolling Sharpe Ratio graph
    valid_sharpe = df.dropna(subset=['Rolling_SPY_Sharpe', 'Rolling_SPY3_Sharpe'])
    trace_spy_sharpe = go.Scatter(x=valid_sharpe['Date'], y=valid_sharpe['Rolling_SPY_Sharpe'], mode='lines', name='SPY Rolling Sharpe Ratio', line=dict(color="grey", width=2))
    trace_spy3_sharpe = go.Scatter(x=valid_sharpe['Date'], y=valid_sharpe['Rolling_SPY3_Sharpe'], mode='lines', name='SPY3 Rolling Sharpe Ratio', line=dict(color="#4472C4", width=2))
    layout_sharpe = go.Layout(title='5y Rolling Sharpe Ratio: SPY3 vs. SPX',
                          xaxis=dict(title='Date'),
                          legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                          yaxis=dict(title='Sharpe Ratio'))
    
      # Rolling Performance graph
    valid_perf = df.dropna(subset=['Rolling_SPY', 'Rolling_SPY3'])
    trace_roll_perf_spy = go.Scatter(x=valid_perf['Date'], y=valid_perf['Rolling_SPY'], mode='lines', name='S&P500 Rolling Performance', line=dict(color="grey", width=2))
    trace_roll_perf_spy3 = go.Scatter(x=valid_perf['Date'], y=valid_perf['Rolling_SPY3'], mode='lines', name='SPY3 Rolling Performance', line=dict(color="#4472C4", width=2))
    layout_roll_perf = go.Layout(title='5y Rolling Annual Performance: SPY3 vs. SPX', 
                             xaxis=dict(title='Date'), 
                             yaxis=dict(title='Annualized Return'), 
                             legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))

    # Rolling Volatility graph
    trace_roll_vol_spy = go.Scatter(x=df['Date'], y=df['Rolling_SPY_Vol'], mode='lines', name='S&P500 Rolling Volatility',line=dict(color="grey",width=2))
    trace_roll_vol_spy3 = go.Scatter(x=df['Date'], y=df['Rolling_SPY3_Vol'], mode='lines', name='SPY3 Rolling Volatility',line=dict(color="#4472C4",width=2))
    layout_roll_vol = go.Layout(title='Rolling Volatility: SPY3 vs. SPX', xaxis=dict(title='Date'), yaxis=dict(title='Volatility (1-year)'), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    
    trace_calmar_spy = go.Scatter(x=df['Date'], y=df['Rolling_Calmar_SPY'], mode='lines', name='Calmar Ratio SPY', line=dict(color="grey", width=2))
    trace_calmar_spy3 = go.Scatter(x=df['Date'], y=df['Rolling_Calmar_SPY3'], mode='lines', name='Calmar Ratio SPY3', line=dict(color="#4472C4", width=2))
    layout_calmar = go.Layout(title='Rolling Calmar Ratios for SPY & SPY3', xaxis=dict(title='Date'), yaxis=dict(title='calmar-ratio-graph'), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1), showlegend=True)

    figure_performance = {"data": [trace1, trace2], "layout": layout_performance}  #add 2nd graph with analysis tools and more traces
    figure_prices_and_var = {"data": [trace_spy_close, trace_shy_close, trace_30d_ma, trace_200d_ma, trace_var, trace_200d_high_discount], "layout": layout_prices_and_var}
    figure_alpha = {"data": [trace_alpha], "layout": layout_alpha}
    figure_dd = {"data": [trace_dd_spy3, trace_dd_spy], "layout": layout_dd}
    figure_divergence = {"data": [trace_divergence], "layout": layout_divergence}
    figure_sharpe = {"data": [trace_spy_sharpe, trace_spy3_sharpe], "layout": layout_sharpe}
    figure_roll_perf = {"data": [trace_roll_perf_spy, trace_roll_perf_spy3], "layout": layout_roll_perf}
    figure_roll_vol = {"data": [trace_roll_vol_spy, trace_roll_vol_spy3], "layout": layout_roll_vol}
    figure_calmar = {"data": [trace_calmar_spy, trace_calmar_spy3], "layout": layout_calmar}
    return figure_performance, figure_prices_and_var, figure_alpha, figure_dd, figure_divergence,figure_sharpe, figure_roll_perf, figure_roll_vol, figure_calmar

print("Current VaR =",round(current_var_1d*100,4),"%")
print('End')

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
    server = app.server
    