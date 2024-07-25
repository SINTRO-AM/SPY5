from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import *
from threading import Thread
import pandas as pd
import time

class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.historicalData = []

    def historicalData(self, reqId, bar):
        self.historicalData.append([bar.date, bar.close])
        print("HistoricalData. ReqId:", reqId, "Date:", bar.date, "Close:", bar.close)

    def historicalDataEnd(self, reqId, start, end):
        print("HistoricalDataEnd. ReqId:", reqId, "from", start, "to", end)
        self.disconnect()

def run_loop():
    app.run()

app = IBapi()
app.connect('127.0.0.1', 4001, 1)

api_thread = Thread(target=run_loop, daemon=True)
api_thread.start()

time.sleep(1)  # Sleep interval to allow time for connection to server

# Define the contract for which to request historical data
contract = Contract()
contract.symbol = "AAPL"  # Example symbol for your portfolio
contract.secType = "STK"
contract.exchange = "SMART"
contract.currency = "USD"

# Request historical data
endDateTime = ''
durationStr = '1 Y'  # Adjust as needed
barSizeSetting = '1 day'
whatToShow = 'ADJUSTED_LAST'
useRTH = 1
formatDate = 1
keepUpToDate = False
chartOptions = []

app.reqHistoricalData(1, contract, endDateTime, durationStr, barSizeSetting, whatToShow, useRTH, formatDate, keepUpToDate, chartOptions)

time.sleep(5)  # Sleep interval to allow time for data to be returned

# Convert the historical data to a DataFrame
df = pd.DataFrame(app.historicalData, columns=["Date", "NAV"])
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
print(df)

app.disconnect()
