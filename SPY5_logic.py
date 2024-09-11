import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
from scipy.stats import norm


def preparing_df():

    start_date = dt.datetime(2000, 1, 1)
    start_dateB = dt.datetime(2000, 1, 1)
    end_date = dt.datetime.today()

    spx = yf.Ticker("SPY")
    spx_hist = spx.history(start=start_date, end=end_date)
    ust = yf.Ticker("GOVT")
    ust_hist = ust.history(start=start_dateB, end=end_date)

    df = pd.DataFrame({'close_SPX': spx_hist['Close']})
    df['close_govB'] = ust_hist['Close']
    df['Return_SPX'] = np.log(df['close_SPX'] / df['close_SPX'].shift(1))
    df['Return_govB'] = np.log(df['close_govB'] / df['close_govB'].shift(1))

    # Check if there are empty cells in "CloseB"
    if df['close_govB'].isnull().values.any():
        # Fill empty cells in "ReturnB" with 0
        df['Return_govB'].fillna(0, inplace=True)

    df.index = df.index.date 
    return df
def get_date_today(df):

    # #können wir hierfür nicht auch einfach den letzten verfügbaren Tag nehmen?
    # date_today = dt.datetime.now().date()-dt.timedelta(days = 1)

    # if date_today.weekday() >= 5:  # 5 steht für Samstag, 6 für Sonntag
    #     while date_today.weekday() != 4:  # 4 steht für Freitag
    #         date_today -= dt.timedelta(days = 2) # aber warum hier -2?????
    
    return df.index.max()

def SMA_signal(df, date_today):
    """
    This function calculates the Simple Moving Averages (SMA) for the provided time series data df over two windows: 

    - short_window (default is 50 days) 
    - and long_window (default is 200 days). 

    The function returns two new series representing the short and long term SMAs. 
    An upward trend is identified when the SMA over the short window is higher than the SMA over the long window, 
    and vice versa for downward trends. 
    The function thus provides a mechanism for identifying and following medium-term market trends, 
    while also providing signals for avoiding downside trends.
    """
    short_window = 30
    long_window = 200
    # Calculate the 50-day moving average
    df['30D_MA'] = df['close_SPX'].rolling(window=short_window).mean()
    df['200D_MA'] = df['close_SPX'].rolling(window=long_window).mean()

    if df.at[date_today, "200D_MA"] < df.at[date_today, "30D_MA"]:
        return True
    else:
        return False

def VaR_signal(df, date_today):
    """
   Description: This function calculates the daily Value-at-Risk (VaR) of the provided time series 
   data df at a given confidence_level (default is 99%) over a window of days (default is 50 days). 
   It assumes the returns are normally distributed and uses the standard deviation of returns over the window to estimate the VaR. 
   
   The VaR calculated by this function indicates the potential loss that could occur with a (1 - confidence_level) probability. 
   The function also annualizes the standard deviation of returns using the annualize_factor, 
   typically 252, representing the average number of trading days in a year.
    
    """

    # Daily log return
    df['log_return_SPX'] = np.log(df['close_SPX'] / df['close_SPX'].shift(1))
    # calculate the daily standard deviation over the past 50 days
    df['STD'] = df['log_return_SPX'].rolling(window=50).std()

    # Annualized Standard deviation
    df['50d_STD'] = df['STD'] * np.sqrt(252)

    # daily VaR (0.01 confidence interval and 50-day history)
    df['VaR_1d'] = (df['50d_STD'] * norm.ppf(0.99) * np.sqrt(1/252))

    value_at_risk = df.at[date_today, "VaR_1d" ]
    if  value_at_risk < 0.02:
        return  True
    else:
        return False

def VaR_threshold(df, date_today):

    # Daily log return
    df['log_return_SPX'] = np.log(df['close_SPX'] / df['close_SPX'].shift(1))
    # calculate the daily standard deviation over the past 50 days
    df['STD'] = df['log_return_SPX'].rolling(window=30).std()

    # Annualized Standard deviation
    df['30d_STD'] = df['STD'] * np.sqrt(252)

    # daily VaR (0.01 confidence interval and 50-day history)
    df['VaR_1d'] = (df['50d_STD'] * norm.ppf(0.99) * np.sqrt(1/252))
    if  df.at[date_today, "VaR_1d" ] < 0.05:
        return True
    else:
        return False    

def Rebound_signal(df, date_today):
    """
    This function calculates the Rebound Factor for the provided time series data df by checking if the annualized returns of the prior 100days is less than -15%
    It returns a boolean series where True indicates a buy signal, i.e., the annualized ("forward-looking") return is less than -15%, 
    suggesting a potential for mean reversion. This factor is designed to exploit market exaggerations 
    and herd behavior by identifying opportunities for buying at significant price discounts 
    and capturing potential gains when markets rally and rebound in the short term.
    """

   
    # Berechnung für den Fwd-Faktor, die annualisierte Rendite der letzten 100 Tage
    df['Fwd'] = (((df['close_SPX'] / df['close_SPX'].shift(101)) - 1) / 100) * 252
  
    # calculate the asset price min 30% below its 200d high
    Fwd = df.at[date_today, 'Fwd'] 

    if Fwd < -0.15 :
        return True
    else:
        return False
    

if __name__ == "__main__" :
    print("This script is not running any functions!")