from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.common import *
from ibapi.contract import Contract
from ibapi.account_summary_tags import AccountSummaryTags
from threading import Thread
import time

class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.accountSummaryData = {}  # Dictionary to store account summary data
        self.positions = {}  # Dictionary to store position details

    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        super().accountSummary(reqId, account, tag, value, currency)
        # Store results in the account summary dictionary
        if account not in self.accountSummaryData:
            self.accountSummaryData[account] = {}
        self.accountSummaryData[account][tag] = value
        print(f"Received account summary for {account}: {tag} = {value}")

    def accountSummaryEnd(self, reqId: int):
        super().accountSummaryEnd(reqId)
        print("Account Summary End")
        # Do not disconnect here if expecting position data

    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        print(f"Received position for account {account}, {contract.symbol}: {position} @ {avgCost}")
        if account not in self.positions:
            self.positions[account] = {}
        self.positions[account][contract.symbol] = {
            'position': position,
            'avgCost': avgCost
        }

    def positionEnd(self):
        super().positionEnd()
        print("Position data received.")
        self.disconnect()  # Disconnect after receiving all position data

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

# Sleep interval to allow time for account summary and positions to be returned; may need adjustment
time.sleep(10)

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

app.disconnect()
