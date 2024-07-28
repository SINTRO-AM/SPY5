import plotly.graph_objects as go
import numpy as np


def create_nav_figure(df, title):
    return {
        'data': [go.Scatter(
            x=df['Date'],
            y=df['NAV'],
            mode='lines',
            name='NAV',
            line=dict(color='green', width=2.5),
            hovertemplate='<b>Date</b>: %{x}<br><b>NAV</b>: %{y:.2f}€<br><extra></extra>'
        )],
        'layout': go.Layout(
            title=title,
            xaxis={'title': 'Date'},
            yaxis={'title': 'NAV'},
            font=dict(color='#2c3e50')
        )
    }


def create_cumulative_return_figure(df, spy_data, title):
    return {
        'data': [
            go.Scatter(
                x=df['Date'],
                y=df['Cumulative_Log_Return'],
                mode='lines',
                name='Strategy Cumulative Log Return',
                line=dict(color='green', width=2.5),
                hovertemplate='<b>Date</b>: %{x}<br><b>Strategy Cumulative Log Return</b>: %{y:.2%}<extra></extra>'
            ),
            go.Scatter(
                x=spy_data['Date'],
                y=spy_data['Cumulative_Log_Return'],
                mode='lines',
                name='SPY Cumulative Log Return',
                hovertemplate='<b>Date</b>: %{x}<br><b>SPY Cumulative Log Return</b>: %{y:.2%}<extra></extra>'
            )
        ],
        'layout': go.Layout(
            title=title,
            xaxis={'title': 'Date'},
            yaxis={'title': 'Cumulative Log Return (%)', 'tickformat': '.2%'},
            font=dict(color='#2c3e50'),
            legend=dict(
                x=0,
                y=1,
                xanchor='left',
                yanchor='top',
                orientation='v'
            )
        )
    }


def create_max_drawdown_figure(df, spy_data, title):
    return {
        'data': [
            go.Scatter(
                x=df['Date'],
                y=df['Max_Drawdown'],
                mode='lines',
                name='Strategy Max Drawdown',
                line=dict(color='green', width=2.5),
                hovertemplate='<b>Date</b>: %{x}<br><b>Strategy Max Drawdown</b>: %{y:.2%}<extra></extra>'
            ),
            go.Scatter(
                x=spy_data['Date'],
                y=spy_data['Max_Drawdown'],
                mode='lines',
                name='SPY Max Drawdown',
                hovertemplate='<b>Date</b>: %{x}<br><b>SPY Max Drawdown</b>: %{y:.2%}<extra></extra>'
            )
        ],
        'layout': go.Layout(
            title=title,
            xaxis={'title': 'Date'},
            yaxis={'title': 'Max Drawdown (%)', 'tickformat': '.2%'},
            font=dict(color='#2c3e50'),
            legend=dict(
                x=0,
                y=0,
                xanchor='left',
                yanchor='bottom',
                orientation='v'
            )
        )
    }



def calculate_kpis(df, spy_data):
    strategy_annual_return = (np.exp(df['Log_Return'].mean() * 252) - 1) * 100
    strategy_annual_vol = df['Log_Return'].std() * np.sqrt(252) * 100
    strategy_total_return = (np.exp(df['Cumulative_Log_Return'].iloc[-1]) - 1) * 100
    strategy_sharpe_ratio = df['Log_Return'].mean() / df['Log_Return'].std() * (252 ** 0.5)
    strategy_max_drawdown = df['Max_Drawdown'].min() * 100
    strategy_positive_days = (df['Log_Return'] > 0).mean() * 100

    spy_annual_return = (np.exp(spy_data['Log_Return'].mean() * 252) - 1) * 100
    spy_annual_vol = spy_data['Log_Return'].std() * np.sqrt(252) * 100
    spy_total_return = (np.exp(spy_data['Cumulative_Log_Return'].iloc[-1]) - 1) * 100
    spy_sharpe_ratio = df['Log_Return'].mean() / df['Log_Return'].std() * (252 ** 0.5)
    spy_max_drawdown = spy_data['Max_Drawdown'].min() * 100
    spy_positive_days = (spy_data['Log_Return'] > 0).mean() * 100

    kpis = [
        {"KPI": "Annualized Return", "Overall": f"{strategy_annual_return:.2f}%", "SPY": f"{spy_annual_return:.2f}%"},
        {"KPI": "Annualized Volatility", "Overall": f"{strategy_annual_vol:.2f}%", "SPY": f"{spy_annual_vol:.2f}%"},
        {"KPI": "Total Return", "Overall": f"{strategy_total_return:.2f}%", "SPY": f"{spy_total_return:.2f}%"},
        {"KPI": "Sharpe Ratio", "Overall": f"{strategy_sharpe_ratio:.2f}", "SPY": f"{spy_sharpe_ratio:.2f}"},
        {"KPI": "Max Drawdown", "Overall": f"{strategy_max_drawdown:.2f}%", "SPY": f"{spy_max_drawdown:.2f}%"},
        {"KPI": "Positive Days %", "Overall": f"{strategy_positive_days:.2f}%", "SPY": f"{spy_positive_days:.2f}%"}
    ]
    
    return kpis