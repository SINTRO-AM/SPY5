import plotly.graph_objects as go
import numpy as np

def calculate_strategy_kpis(df,strategy, spy_data):
    strategy_annual_return = (np.exp(df['Log_Return'].mean() * 252) - 1) * 100
    strategy_annual_vol = df['Log_Return'].std() * np.sqrt(252) * 100
    strategy_total_return = (np.exp(df['Cumulative_Log_Return'].iloc[-1]) - 1) * 100
    strategy_sharpe_ratio = df['Log_Return'].mean() / df['Log_Return'].std() * (252 ** 0.5)
    strategy_max_drawdown = df['Max_Drawdown'].min() * 100
    strategy_positive_days = (df['Log_Return'] > 0).mean() * 100

    spy_annual_return = (np.exp(spy_data['Log_Return'].mean() * 252) - 1) * 100
    spy_annual_vol = spy_data['Log_Return'].std() * np.sqrt(252) * 100
    spy_total_return = (np.exp(spy_data['Cumulative_Log_Return'].iloc[-1]) - 1) * 100
    spy_sharpe_ratio = spy_data['Log_Return'].mean() / spy_data['Log_Return'].std() * (252 ** 0.5)
    spy_max_drawdown = spy_data['Max_Drawdown'].min() * 100
    spy_positive_days = (spy_data['Log_Return'] > 0).mean() * 100

    kpis = [
        {"KPI": "Annualized Return", strategy: f"{strategy_annual_return:.2f}%", "SPY": f"{spy_annual_return:.2f}%"},
        {"KPI": "Annualized Volatility", strategy: f"{strategy_annual_vol:.2f}%", "SPY": f"{spy_annual_vol:.2f}%"},
        {"KPI": "Total Return", strategy: f"{strategy_total_return:.2f}%", "SPY": f"{spy_total_return:.2f}%"},
        {"KPI": "Sharpe Ratio", strategy: f"{strategy_sharpe_ratio:.2f}", "SPY": f"{spy_sharpe_ratio:.2f}"},
        {"KPI": "Max Drawdown", strategy: f"{strategy_max_drawdown:.2f}%", "SPY": f"{spy_max_drawdown:.2f}%"},
        {"KPI": "Positive Days %", strategy: f"{strategy_positive_days:.2f}%", "SPY": f"{spy_positive_days:.2f}%"}
    ]
    return kpis