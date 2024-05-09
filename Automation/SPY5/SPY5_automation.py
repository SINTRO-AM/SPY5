from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import *
from ibapi.account_summary_tags import AccountSummaryTags
from ibapi.order import Order
import threading
import yfinance as yf
import time
from SPY5_logic import preparing_df, get_date_today, SMA_signal, VaR_signal, Rebound_signal, VaR_threshold
import math
import yaml
import os

# Load the YAML config file
current_directory = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_directory, "config", "config.yml")
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

ACTIVE_SYMBOL = config['trading_instruments']['active']['symbol']
PASSIVE_SYMBOL = config['trading_instruments']['passive']['symbol']
acc_ID = config['ACC_IDs']['ID_1']


class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.next_order_id = None
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
            'avgCost': avgCost,
            'total':position*avgCost
        }

    def positionEnd(self):
        super().positionEnd()
        print("Position data received.")
        # self.disconnect()  # Disconnect after receiving all position data


    def nextValidId(self, orderId: int):
        '''Called when the next valid order ID is received from TWS '''
        super().nextValidId(orderId)
        self.next_order_id = orderId
        print('The next valid order id is: ', self.next_order_id)

    def place_trade(self, symbol, action, quantity, type_of_order="MOC"):
        '''Method to place a trade, splitting into multiple orders if above size limit.'''
        max_order_size = 12000  # Maximum size for non-algorithmic orders
        contract = Contract()
        contract.symbol = symbol
        contract.secType = config['trading_instruments']['active']['type'] if symbol == ACTIVE_SYMBOL else config['trading_instruments']['passive']['type']
        contract.exchange = 'SMART'
        contract.currency = 'USD'
        
        while quantity > 0:
            order_size = min(quantity, max_order_size)
            order = Order()
            order.action = action
            order.totalQuantity = order_size
            order.orderType = type_of_order
            order.eTradeOnly = ""
            order.firmQuoteOnly = ""
            self.placeOrder(self.next_order_id, contract, order)
            print(f"Trade placed: {action} {order_size} {symbol} at {type_of_order} order type.")
            self.next_order_id += 1
            quantity -= order_size
            time.sleep(1)  # Delay between orders to prevent rate limiting and to manage order placement timing



    def disconnect_after_trading(self):
        time.sleep(3)  # Give some time for all trades to be processed
        self.disconnect()

def get_price(symbol):
    ticker = yf.Ticker(symbol)
    todays_data = ticker.history(period='max')
    return todays_data['Close'].iloc[-1]

def run_loop(app):
    app.run()


def main():
    app = IBapi()
    app.connect('127.0.0.1', 4002, 2)

    api_thread = threading.Thread(target=lambda: app.run(), daemon=True)
    api_thread.start()

    # Hier erfolgt die Logik, um auf eine gültige Order-ID zu warten und die Positionen abzufragen, wie zuvor definiert

    df = preparing_df()
    date_today = get_date_today(df)

    signal_SMA = SMA_signal(df,date_today)
    signal_VaR = VaR_signal(df,date_today)
    signal_rebound = Rebound_signal(df,date_today)
    threshold_VaR = VaR_threshold(df,date_today)

    signal = (threshold_VaR and (signal_SMA or signal_VaR or signal_rebound))
    signal = False
    # Requesting account summary
    print("Requesting Account Summary")
    app.reqAccountSummary(9001, "All", AccountSummaryTags.AllTags)
    time.sleep(3)
    # Also request positions
    print("Requesting Current Positions")
    app.reqPositions() 
    time.sleep(3)

    print("signal_SMA:", signal_SMA)
    print("signal_VaR:", signal_VaR)
    print("signal_rebound:", signal_rebound)
    print("Below our threshold_VaR:", threshold_VaR)
    print(f"Signal: {signal}")
    available_funds = float(app.accountSummaryData.get(acc_ID).get("TotalCashValue",0))
    print(available_funds)
    time.sleep(5)
    price_spy = get_price(ACTIVE_SYMBOL)  # Preisabfrage für SPY
    price_bond = get_price("VDST.L")  # Angenommen, du hast eine Methode, um den Preis für PASSIVE_SYMBOL zu bekommen

    app_positions = app.positions.get(acc_ID)
    print(app_positions)
    if signal: #if signal == True then buy future Risk On!!
        if ACTIVE_SYMBOL in app_positions:
            print("IN 1")

            # Kaufe active instrument basierend auf verfügbaren Mitteln
            quantity = math.floor(available_funds / price_spy)
            if quantity > 0:
                app.place_trade(ACTIVE_SYMBOL, 'BUY', quantity)
        elif not app_positions:
            print("IN 2")

            # Kaufe SPY mit allem Kapital
            quantity = math.floor(available_funds /math.ceil(price_spy))
            if quantity > 0:
                app.place_trade(ACTIVE_SYMBOL, 'BUY', quantity)
        elif PASSIVE_SYMBOL in app_positions:
            print("IN 3")

            # Verkaufe BOND und kaufe SPY mit allem Kapital
            bond_quantity = app_positions[PASSIVE_SYMBOL]['position']
            app.place_trade(PASSIVE_SYMBOL, 'SELL', bond_quantity, "MKT") ## MKT
            available_funds += bond_quantity * price_bond  # Angenommen, dies ist der Erlös aus dem Verkauf
            quantity = math.floor(available_funds / price_spy)
            if quantity > 0:
                app.place_trade(ACTIVE_SYMBOL, 'BUY', quantity) ## MOC - muss noch angepasst werden.

       # Continuing with the trading logic for the PASSIVE_SYMBOL signal
    elif not signal: # if signal is false then buy..
        if ACTIVE_SYMBOL in app_positions:
            spy_quantity = app_positions[ACTIVE_SYMBOL]['position']
            app.place_trade(ACTIVE_SYMBOL, 'SELL', spy_quantity, "MKT")
            time.sleep(1)
            available_funds += (spy_quantity * price_spy ) - 1 ### and check if other cash is in account and add it to funds!

            quantity = math.floor(available_funds / price_bond)
            if quantity > 0:
                app.place_trade(PASSIVE_SYMBOL, 'BUY', quantity, "MKT")
        elif not app_positions:
            # Buy BOND with all available capital
            quantity = math.floor(available_funds / price_bond)
            if quantity > 0:
                app.place_trade(PASSIVE_SYMBOL, 'BUY', quantity)
        elif PASSIVE_SYMBOL in app_positions:
            # Buy additional BONDs based on available funds
            quantity = math.floor(available_funds / price_bond)
            if quantity > 0:
                app.place_trade(PASSIVE_SYMBOL, 'BUY', quantity)


    app.disconnect_after_trading()

if __name__ == "__main__":
    main()