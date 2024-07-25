import yfinance as yf
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.common import *
from ibapi.contract import Contract
from ibapi.account_summary_tags import AccountSummaryTags
from ibapi.execution import Execution, ExecutionFilter
from threading import Thread
import pandas as pd
import time
import os
import yaml

current_directory = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_directory, "config", "config.yml")
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

daw_path = os.path.join(current_directory, "DaW.csv")
daw = pd.read_csv(daw_path)
daw_acc_sums = daw.groupby("Account").sum()[["Amount"]]
acc_ID_mapping = config['ACC_IDs']

class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.accountSummaryData = {}  # Dictionary to store account summary data
        self.positions = {}  # Dictionary to store position details
        self.executions = []  # List to store execution details

    def execDetails(self, reqId: int, contract: Contract, execution: Execution):
        print("ExecDetails. ReqId:", reqId, "Symbol:", contract.symbol, "SecType:", contract.secType, "Currency:", contract.currency, execution)
        self.executions.append({
            'order_id': execution.orderId,
            'shares': execution.shares,
            'price': execution.price,
            'time': execution.time
        })

    def execDetailsEnd(self, reqId: int):
        super().execDetailsEnd(reqId)
        print("Received all execution details")

    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        super().accountSummary(reqId, account, tag, value, currency)
        if account not in self.accountSummaryData:
            self.accountSummaryData[account] = {}
        self.accountSummaryData[account][tag] = value
        print(f"Received account summary for {account}: {tag} = {value}")

    def accountSummaryEnd(self, reqId: int):
        super().accountSummaryEnd(reqId)
        print("Account Summary End")

    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        print(f"Received position for account {account}, {contract.symbol}: {position} @ {avgCost}")
        if account not in self.positions:
            self.positions[account] = {}
        self.positions[account][contract.symbol] = {
            'contract': contract,
            'position': position,
            'avgCost': avgCost
        }

    def positionEnd(self):
        super().positionEnd()
        print("Position data received.")
    

def run_loop():
    app.run()

app = IBapi()
app.connect('127.0.0.1', 4001, 1)

# Start the socket in a thread
api_thread = Thread(target=run_loop, daemon=True)
api_thread.start()

time.sleep(1)  # Sleep interval to allow time for connection to server

# Requesting account summary
print("Requesting Account Summary")
app.reqAccountSummary(9001, "All", AccountSummaryTags.AllTags)

# Also request positions
print("Requesting Current Positions")
app.reqPositions()  # Request to receive all positions

# Requesting execution data
app.reqExecutions(9002, ExecutionFilter())
# Sleep interval to allow time for account summary and positions to be returned; may need adjustment
time.sleep(3)

print(acc_ID_mapping)
# Access and display the stored account summary data
print("Account Summary Data:")
for account, summary in app.accountSummaryData.items():
    print(f"Account {account}:")
    for tag, value in summary.items():
        print(f"  {tag}: {value}")

# Access and display the stored positions
print("\nCurrent Positions Data:")
for account, positions in app.positions.items():
    print(f"Positions for Account {account}:")
    for symbol, details in positions.items():
        print(f"  Symbol: {symbol}, Position: {details['position']}, Average Cost: {details['avgCost']}")

# Get current price data from Yahoo Finance
symbols = [symbol for account in app.positions.values() for symbol in account.keys()]
current_prices = {}
historical_data = dict()
for symbol in symbols:
    if symbol == "VDST":
        ticker = yf.Ticker(symbol+".L")
    else:
        ticker = yf.Ticker(symbol)
    
    try:
        hist_data = ticker.history(start="2020-01-01")['Close']
        current_price = hist_data[-1]
        current_prices[symbol] = current_price

        historical_data[symbol] = hist_data

    except IndexError:
        print(f"Error: Could not find data for symbol {symbol}")

# Add current prices to the positions
for account, positions in app.positions.items():
    for symbol, details in positions.items():
        details['currentPrice'] = current_prices.get(symbol, None)

acc_summery_data = dict()
for account, KPIS in app.accountSummaryData.items():
        acc_summery_data[account] = KPIS

acc_summery_data = pd.DataFrame.from_dict(acc_summery_data, orient='index')

# Create DataFrame for positions with current prices
records = []
inv_acc_ID_mapping = {v: k for k, v in acc_ID_mapping.items()}
for acc_id, symbols in app.positions.items():
    for symbol, details in symbols.items():
        record = {
            'ACC_ID': acc_id,
            'Account_Name': inv_acc_ID_mapping.get(acc_id, acc_id),
            'Symbol': symbol,
            'Position': details['position'],
            'AverageCost': details['avgCost'],
            'CurrentPrice': details['currentPrice']
        }
        records.append(record)

df_positions = pd.DataFrame(records).set_index(['ACC_ID', 'Symbol'])
print(df_positions)



# Create a DataFrame with historical data
df_historical = pd.DataFrame(historical_data)

execution_records = []
for execution in app.executions:
    execution_records.append(execution)

df_executions = pd.DataFrame(execution_records)

# Save DataFrames as CSV files
data_for_reporting = os.path.join(current_directory, "data_for_reporting")

df_executions.to_csv(os.path.join(data_for_reporting, "executions.csv"))
daw_acc_sums.to_csv(os.path.join(data_for_reporting, "daw_acc_sums.csv"))
daw.to_csv(os.path.join(data_for_reporting, "daw.csv"))
df_historical.to_csv(os.path.join(data_for_reporting, "historical_data.csv"))
df_positions.to_csv(os.path.join(data_for_reporting, "positions.csv"))
acc_summery_data.to_csv(os.path.join(data_for_reporting, "master_overview.csv"))
app.disconnect()
