import pandas as pd
import numpy as np
import datetime as dt

from dash import callback_context
import dash
import plotly.graph_objs as go
from scipy.stats import norm
import yfinance as yf



def start_end_date(button_id):
    """
    Determines the start date based on the button clicked in the Dash app.
    The function calculates the start date for different time ranges: 
    1 year, 2 years, 4 years, 5 years, Year to Date, or a default start date.
    :param button_id: The ID of the button that was clicked.
    :return: The calculated start date.
    """
    # Mapping button IDs to their corresponding time deltas
    button_map = {
        'btn-1yr': dt.timedelta(days=365),
        'btn-2yr': dt.timedelta(days=365 * 2),
        'btn-4yr': dt.timedelta(days=365 * 4),
        'btn-5yr': dt.timedelta(days=365 * 5),
        'btn-YtD': 'ytd',
        'btn-Live': 'Live',
        'btn-max': 'total'  # Total period since 01-01-2000
    }

    # Calculate the start date based on the clicked button
    if button_id in button_map:
        if button_map[button_id] == 'ytd':
            now = dt.datetime.now()
            start_date = dt.datetime(now.year, 1, 1)  # First day of current year
        elif button_map[button_id] == 'Live':
            start_date = dt.datetime(2022, 4, 1)  # Inception Date
        elif button_map[button_id] == 'total':
            start_date = dt.datetime(2000, 1, 1)  # January 1, 2000
        else:
            start_date = dt.datetime.now() - button_map[button_id]
    else:
        # Default start date (e.g., 1 year back if no button is clicked)
        start_date = dt.datetime(2000, 1, 1)
    return start_date



def get_clicked_button_id():
    """
    Determines which button was clicked in the Dash app.
    Uses Dash's callback_context to find the ID of the clicked button.
    Returns 'default' if no button was clicked.
    """
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'default'
    return button_id




# Funktion 1: Datenabruf
def fetch_data(ticker_symbol, initial_date):
    end_date = dt.datetime.now()
    ticker = yf.Ticker(ticker_symbol)
    return ticker.history(start=initial_date, end=end_date)

# Funktion 2: Berechnung der Basiswerte
def calculate_base_values(px_hist, ust_hist,second = None):
    if second is None:

        df = pd.DataFrame()
        df["Close"] = px_hist['Close']
        df['Date'] = px_hist.index
    else:
        df  = second

    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.to_period('Y')
    df['Month'] = df['Date'].dt.to_period('M')

    df['CloseB'] = ust_hist['Close']

    df['Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['ReturnB'] = np.log(df['CloseB'] / df['CloseB'].shift(1))

    if df['CloseB'].isnull().values.any():
        df['ReturnB'].fillna(0, inplace=True)

    return df

# Funktion 3: Berechnung der rollierenden Werte und Kennzahlen
def calculate_rolling_values_and_kpis(df):
    df['SPX_Total_Return'] = (df['Return']).cumsum()
    df['UST_Total_Return'] = (df['ReturnB']).cumsum()
    # Berechnung der gleitenden Durchschnitte
    df['30D_MA'] = df['Close'].rolling(window=30).mean()
    df['200D_MA'] = df['Close'].rolling(window=198).mean()
    df['STD'] = df['Return'].rolling(window=50,min_periods = 1).std()

  # Berechnung der annualisierten Standardabweichung
    df['50d_STD'] = df['STD'] * np.sqrt(252)

    # Berechnung des täglichen Value at Risk (VaR) auf Basis der letzten 50 Returns
    df['VaR_1d'] = -df['50d_STD'] * norm.ppf(0.99) * np.sqrt(1/252)*-1

    # Signal
    df.loc[df['VaR_1d'] > 0.05, 'Signal'] = 0
    
    # Wenn eine der gegebenen Bedingungen zutrifft, setze Signal auf 1
    conditions = ((df['VaR_1d'] <= 0.05) & ((df['30D_MA'] > df['200D_MA']) | 
        (df['VaR_1d'] < 0.02) | 
        (df['Close'] < df['Close'].rolling(window=200, min_periods = 1).max()/1.3))
    )
    df.loc[conditions, 'Signal'] = 1

    
    # In allen anderen Fällen setze Signal auf 0
    df['Signal'].fillna(0, inplace=True)

    # SignalB
    df.loc[df['Signal'] == 0, 'SignalB'] = 1
    df.loc[df['Signal'] == 1, 'SignalB'] = 0

    # Time-lag & transaction costs
    df['Portfolio_Return'] = df['Return'] * df['Signal'].shift(1) + df['ReturnB'] * df['SignalB'].shift(1)
    previous_signal = df['Signal'].shift(2)
    df.loc[df['Signal'] != previous_signal, 'Portfolio_Return'] -= 0.0001  # transaction costs of 1 basis points (0.0001)
    df['Total_Return'] = (df['Portfolio_Return']).cumsum()
    df[['Total_Return',"SPX_Total_Return"]] =df[["Total_Return","SPX_Total_Return"]].fillna(0)

    # Berechnung des Rolling 200 Day High Discount
    df['Rolling_200D_High_Discount'] = df['Close'].rolling(window=200, min_periods=1).max() / 1.3

    # Kumulierte Rendite berechnen
    df['Cumulative_Return'] = df['Return'].cumsum()

    df['Portfolio_Return'] = df['Return'] * df['Signal'].shift(1) + df['ReturnB'] * df['SignalB'].shift(1)

    # Berechnung der rollierenden Performance (kumulative Summe der Renditen)
    df['Rolling_SPY'] = df['Return'].rolling(window=1260, min_periods=1).sum() / 5
    df['Rolling_SPY3'] = df['Portfolio_Return'].rolling(window=1260, min_periods=1).sum() / 5

    # Berechnung der rollierenden Volatilität (Standardabweichung der Renditen)
    df['Rolling_SPY_Vol'] = df['Return'].rolling(window=252, min_periods=1).std()
    df['Rolling_SPY3_Vol'] = df['Portfolio_Return'].rolling(window=252, min_periods=1).std()

    # Berechnung des rollierenden Sharpe-Verhältnisses
    df['Rolling_SPY_Sharpe'] = (df['Return'].rolling(window=1260).mean() * 252) / (df['Return'].rolling(window=1260).std() * np.sqrt(252))
    df['Rolling_SPY3_Sharpe'] = (df['Portfolio_Return'].rolling(window=1260).mean() * 252) / (df['Portfolio_Return'].rolling(window=1260).std() * np.sqrt(252))

    df['DD_SPY3'] = df['Total_Return'] - df['Total_Return'].cummax()
    df['DD_SPY'] = df['SPX_Total_Return'] - df['SPX_Total_Return'].cummax()

    df['Alpha'] = (df['Portfolio_Return']) - (df['Return'])
    df['Alpha_cum'] = (df['Alpha']).cumsum()


    return df


# Funktion 4: Filterfunktion
def filter_data(df, start_date):
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    start_date = pd.to_datetime(start_date).tz_localize(None)

    df = df[df['Date'] >= start_date]
    return df

# Hauptfunktion
def fetch_and_process_data(selected_index, start_date):
    initial_date = dt.datetime(2000, 1, 1)
    
    # Abrufen der U.S. Treasuries Daten
    ust_hist = fetch_data("SHY", initial_date)

    # Auswählen des Tickers
    if selected_index == '^GDAXI':
        ticker_symbol = "^GDAXI"
    elif selected_index == '^NDX':
        ticker_symbol = "^NDX"
    else:
        ticker_symbol = "SPY"

    # Abrufen der Indexdaten
    px_hist = fetch_data(ticker_symbol, initial_date)

    # Basiswerte berechnen
    df = calculate_base_values(px_hist, ust_hist,second = None)

    # Rollende Werte und KPIs berechnen
    df = calculate_rolling_values_and_kpis(df)

    # Daten basierend auf dem ausgewählten Zeitraum filtern
    df = filter_data(df, start_date)

    # df = calculate_base_values(px_hist, ust_hist,second = df)
    df = calculate_rolling_values_and_kpis(df)
    


    return df

# Verwendung der Hauptfunktion
# Hier beginnen die Funktionen der Graphen
#


def create_performance_graph(df):
    """
    Enhanced version with stylish elements.
    """
    # Creating the figure and adding traces with updated style
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SPX_Total_Return'], mode='lines', name='S&P500 Total Return', line=dict(color="darkgrey", width=2.5)))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Total_Return'], mode='lines', name='SPY3 Total Return', line=dict(color="#1f77b4", width=2.5)))  # dark blue

    # Update the layout with style elements
    fig.update_layout(
        xaxis=dict(titlefont=dict(size=14), gridcolor='lightgrey'),
        yaxis=dict(title='Cumulative Return (%)', titlefont=dict(size=14), gridcolor='lightgrey', tickformat=".0%"),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        title=dict(text='Portfolio Performance Comparison', x=0.5, xanchor='center')
    )
    return fig

def Alpha(df):
    # Assuming df['Date'], df['Portfolio_Return'], and df['Return'] are already defined in df
    df['Alpha'] = (df['Portfolio_Return']) - (df['Return'])
    df['Alpha_cum'] = (df['Alpha']).cumsum()

    trace_alpha = go.Scatter(x=df['Date'], y=df['Alpha_cum'], mode='lines', name='Alpha', line=dict(color="#4D79C7", width=2), fill='tozeroy', fillcolor='rgba(77, 121, 199, 0.35)')

    layout_alpha = go.Layout(
        xaxis=dict(titlefont=dict(size=14), gridcolor='lightgrey'), 
        yaxis=dict(title='Alpha',titlefont=dict(size=14), gridcolor='lightgrey',tickformat=".0%"),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        margin=dict(l=20, r=20, t=10, b=20)
    )
    return go.Figure(data=[trace_alpha], layout=layout_alpha)

def maxDD(df):
    # Define traces for maximum drawdowns
    trace_max_dd_spy3 = go.Scatter(x=df['Date'], y=df['DD_SPY3'], mode='lines', name='Max Drawdown SPY3', line=dict(color="#4472C4"))
    trace_max_dd_spy = go.Scatter(x=df['Date'], y=df['DD_SPY'], mode='lines', name='Max Drawdown SPY', line=dict(color="grey"))

    # Define the layout with style elements
    layout_max_dd = go.Layout(
        xaxis=dict(titlefont=dict(size=14), gridcolor='lightgrey'),
        yaxis=dict(title='Max Drawdown', titlefont=dict(size=14), gridcolor='lightgrey',tickformat=".0%"),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        margin=dict(l=20, r=20, t=10, b=20)
    )
    return go.Figure(data=[trace_max_dd_spy3, trace_max_dd_spy], layout=layout_max_dd)

##### Bar chart of annual returns
def create_annual_bar_chart(df):
    # Calculate Annual Returns
    df['Year'] = pd.to_datetime(df['Date']).dt.to_period('Y')
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
        marker_color='steelblue'
    ))
    annual_bar.add_trace(go.Bar(
        x=annual_returns_table['Year'],
        y=annual_returns_table['SPY Annual Returns'].str.rstrip('%').astype('float'),
        name='SPY Annual Returns',
        marker_color='lightgray'
    ))

    annual_bar.update_layout(
        font=dict(family="Segoe UI"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return annual_bar

def create_price_and_var_graph(df):
    """
    Creates a graph for Close prices, Moving averages, and VaR (Value at Risk).
    :param df: DataFrame containing the financial data.
    :return: Plotly graph object (Figure) for the dashboard.
    """
    # Define traces for Close prices, Moving Averages, and VaR
    trace_spy_close = go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='SPY Close', line=dict(color="#1f77b4"))  # dark blue
    trace_shy_close = go.Scatter(x=df['Date'], y=df['CloseB'], mode='lines', name='US Treasuries', line=dict(color="darkgrey"))
    trace_30d_ma = go.Scatter(x=df['Date'], y=df['30D_MA'], mode='lines', name='30 Day MA', line=dict(dash='dashdot', color='green'))
    trace_200d_ma = go.Scatter(x=df['Date'], y=df['200D_MA'], mode='lines', name='200 Day MA', line=dict(dash='dashdot', color='grey'))
    trace_200d_high_discount = go.Scatter(x=df['Date'], y=df['Rolling_200D_High_Discount'], mode='lines', name='Mean-Reversion Threshold', line=dict(dash='dash', color="#FFBF00"))
    trace_var = go.Scatter(x=df['Date'], y=df['VaR_1d'], mode='lines', name='VaR', yaxis='y2', line=dict(color="#E088B0"))
    trace_risk_off = go.Scatter(
        x=[None], y=[None],
        mode='lines',
        name='Risk Off',
        line=dict(color='rgba(128, 128, 128, 0.3)', width = 10),
        showlegend=True
    )

      # Grüne Linie bei y=2%
    green_line = {'type': 'line','xref': 'paper','x0': 0,'x1': 1,'yref': 'y2','y0': 0.02,'y1': 0.02,
        'line': {
            'color': 'green',
            'width': 2,
            'dash': 'dashdot',
        },
    }

    # Rote Linie bei y=5%
    red_line = {'type': 'line','xref': 'paper','x0': 0,'x1': 1,'yref': 'y2','y0': 0.05,'y1': 0.05,
        'line': {
            'color': 'red',
            'width': 2,
            'dash': 'dashdot',
        },
    }
    # Generate shapes for signals
    signal_shapes = generate_shapes_for_signals(df)

    # Define the layout with style elements
    layout_prices_and_var = go.Layout(
        title='Price and VaR Analysis',
        xaxis=dict(title='Date', titlefont=dict(size=14), gridcolor='lightgrey'),
        yaxis=dict(title='Price', titlefont=dict(size=14), gridcolor='lightgrey'),
        yaxis2=dict(title='VaR', titlefont=dict(size=14), overlaying='y', side='right', gridcolor='lightgrey'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        shapes=[green_line, red_line] + signal_shapes   # Add signal shapes to the layout
    )

    # Create and return the figure
    return go.Figure(data=[trace_spy_close, trace_shy_close, trace_30d_ma, trace_200d_ma, trace_200d_high_discount, trace_var,trace_risk_off], layout=layout_prices_and_var)

def generate_shapes_for_signals(df):
    shapes = []
    df_copy =df.copy()
    df_copy.reset_index(inplace = True, drop = True)
    
    for i, row in df_copy.iterrows():
        # Set the color to light grey if 'Signal' is 1, otherwise do nothing
        if row['Signal'] == 0:
            shape_color = 'grey'  # light grey for signals of 1
        else:
            shape_color = 'rgba(0, 0, 0, 0)'  # fully transparent for other signals

        # Calculate the end date for the shape
        if i < len(df_copy) - 1:
            next_date = df_copy.iloc[i + 1]['Date']
        else:
            next_date = row['Date']

        # Append the shape dictionary to the list
        shapes.append({
            'type': 'rect',
            'xref': 'x',
            'yref': 'paper',
            'x0': row['Date'],
            'x1': next_date,
            'y0': 0,
            'y1': 1,
            'fillcolor': shape_color,
            'opacity': 0.3,  # slightly increased opacity for better visibility
            'line': {'width': 0},
        })
    
    return shapes


def create_results_df(df):
    annual_return_bmk = round(np.mean(df['Return']) * 252 * 100, 2)
    annual_std_bmk = round(np.std(df['Return']) * np.sqrt(252) * 100, 2)
    sharpe_ratio_bmk = round(annual_return_bmk / annual_std_bmk, 2)

    annual_return_pf = round(np.mean(df['Portfolio_Return']) * 252 * 100, 2)
    annual_std_pf = round(np.std(df['Portfolio_Return']) * np.sqrt(252) * 100, 2)
    sharpe_ratio_pf = round(annual_return_pf / annual_std_pf, 2)

    DD_SPY3 = round(df['DD_SPY3'].min() * 100, 2) if 'DD_SPY3' in df else None
    DD_SPY = round(df['DD_SPY'].min() * 100, 2) if 'DD_SPY' in df else None

    results_df = pd.DataFrame({
        'Metric': ['Annual Return in %', 'Annual STD in %', 'Sharpe Ratio', 'max DD in %'],
        'SPY3': [annual_return_pf, annual_std_pf, sharpe_ratio_pf, DD_SPY3],
        'SPY': [annual_return_bmk, annual_std_bmk, sharpe_ratio_bmk, DD_SPY]
    })

    # Umwandlung des DataFrames in ein Format, das für die Dash DataTable geeignet ist
    results_data = results_df.to_dict('records')
    results_columns = [{"name": i, "id": i} for i in results_df.columns]

    return results_data, results_columns

def current_results_df(df):
    Inception_price = 464
    Buy_Price = 435  # Durchschnittlicher Kaufpreis
    inception_date = dt.datetime(2022, 4, 1)
    end_date = dt.datetime.now()
    n_days_since_inception = (end_date - inception_date).days
    current_spy_price = df['Close'].iloc[-1]
    SPY3_return = round(((current_spy_price-1) / Buy_Price - 1) * 100,2)  # Rendite seit Kauf in %
    SPY_return = round((current_spy_price / Inception_price - 1) * 100,2)  # Rendite seit Beginn in %
    Annualized_SPY3_return = round((SPY3_return / n_days_since_inception) * 252,2)
    Annualized_SPY_return = round((SPY_return / n_days_since_inception) * 252 ,2)
    Current_NAV_SPY3 = round(current_spy_price * 25,2) - 1
    Current_NAV_SPY = round(current_spy_price * 25,2)
    formatted_NAV_SPY3 = "${:,.0f}".format(Current_NAV_SPY3)
    formatted_NAV_SPY = "${:,.0f}".format(Current_NAV_SPY)

    current_results_df = pd.DataFrame({
        'Since Inception': ['Current Return in %', 'Annualized Return in %', 'Current NAV in USD'],
        'SPY3': [SPY3_return, Annualized_SPY3_return, formatted_NAV_SPY3],
        'SPY': [SPY_return, Annualized_SPY_return, formatted_NAV_SPY]
    })

    # Umwandlung des DataFrames in ein Format, das für die Dash DataTable geeignet ist
    current_results_data = current_results_df.to_dict('records')
    current_results_columns = [{"name": i, "id": i} for i in current_results_df.columns]

    return current_results_data, current_results_columns