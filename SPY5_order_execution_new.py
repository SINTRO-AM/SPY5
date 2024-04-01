from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
import threading
import yfinance as yf
import time
from SPY5_logic import preparing_df, get_date_today, SMA_signal, VaR_signal, Rebound_signal, VaR_threshold
import math
import yaml

# Load the YAML config file
with open('config/config.yml', 'r') as file:
    config = yaml.safe_load(file)

ACTIVE_SYMBOL = config['trading_instruments']['active']['symbol']
PASSIVE_SYMBOL = config['trading_instruments']['passive']['symbol']

class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.account_summary = {}
        self.next_order_id = None
        self.positions = {}

    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        ''' Called when the account information is returned from IB server '''
        print("AccountSummary. ReqId:", reqId, "Account:", account, 
              "Tag: ", tag, "Value:", value, "Currency:", currency)
        self.account_summary[tag] = value

    def accountSummaryEnd(self, reqId: int):
        ''' Called after all account summary data has been received '''
        #print("AccountSummaryEnd. Req Id: ", reqId)
        # self.disconnect()

    def position(self, account: str, contract: Contract, holdings: float,
             avgCost: float):
        ''' Called whenever a position changes '''
        print(f'Received position for {contract.symbol}: {holdings}')
        key = contract.symbol  # you can extend this key if you want more details like currency, etc.
        self.positions[key] = {
            "holdings": holdings,
            "avgCost": avgCost
    }

    def nextValidId(self, orderId: int):
        '''Called when the next valid order ID is received from TWS '''
        super().nextValidId(orderId)
        self.next_order_id = orderId
        print('The next valid order id is: ', self.next_order_id)

    def place_trade(self, symbol, action, quantity):
        '''Simple Method, to place a trade.'''
        contract = Contract()
        contract.symbol = symbol
        contract.secType = config['trading_instruments']['active']['type'] if symbol == ACTIVE_SYMBOL else config['trading_instruments']['active']['type']
        contract.exchange = 'SMART'
        contract.currency = 'USD'
        
        order = Order()
        order.action = action
        order.totalQuantity = quantity
        order.orderType = 'MKT'
        
        self.placeOrder(self.next_order_id, contract, order)
        self.next_order_id += 1

def get_price(symbol):
    ticker = yf.Ticker(symbol)
    todays_data = ticker.history(period='max')
    return todays_data['Close'][0]

def run_loop(app):
    app.run()


def main():
    app = IBapi()
    app.connect('127.0.0.1', 4001, 99)

    api_thread = threading.Thread(target=lambda: app.run(), daemon=True)
    api_thread.start()

    # Hier erfolgt die Logik, um auf eine gültige Order-ID zu warten und die Positionen abzufragen, wie zuvor definiert

    df = preparing_df()
    date_today = get_date_today(df)

    signal_SMA = SMA_signal(df,date_today)
    signal_VaR = VaR_signal(df,date_today)
    signal_rebound = Rebound_signal(df,date_today)
    threshold_VaR = VaR_threshold(df,date_today)
    # Hier würdest du deine Signallogik implementieren
    signal = (threshold_VaR and (signal_SMA or signal_VaR or signal_rebound))

    print("signal_SMA:", signal_SMA)
    print("signal_VaR:", signal_VaR)
    print("signal_rebound:", signal_rebound)
    print("Below our threshold_VaR:", threshold_VaR)
    print(f"Signal: {signal}")
    available_funds = float(app.account_summary.get("TotalCashBalance", 0))
    price_spy = get_price(ACTIVE_SYMBOL)  # Preisabfrage für SPY
    price_bond = get_price(PASSIVE_SYMBOL)  # Angenommen, du hast eine Methode, um den Preis für PASSIVE_SYMBOL zu bekommen

    # Hier folgt die vereinfachte Logik basierend auf deinem Signal
    if signal: #if signal == True then buy future
        if ACTIVE_SYMBOL in app.positions:
            # Kaufe active instrument basierend auf verfügbaren Mitteln
            quantity = math.floor(available_funds / price_spy)
            if quantity > 0:
                app.place_trade(ACTIVE_SYMBOL, 'BUY', quantity)
        elif not app.positions:
            # Kaufe SPY mit allem Kapital
            quantity = math.floor(available_funds / price_spy)
            if quantity > 0:
                app.place_trade(ACTIVE_SYMBOL, 'BUY', quantity)
        elif PASSIVE_SYMBOL in app.positions:
            # Verkaufe BOND und kaufe SPY mit allem Kapital
            bond_quantity = app.positions[PASSIVE_SYMBOL]['holdings']
            app.place_trade(PASSIVE_SYMBOL, 'SELL', bond_quantity)
            available_funds += bond_quantity * price_bond  # Angenommen, dies ist der Erlös aus dem Verkauf
            quantity = math.floor(available_funds / price_spy)
            if quantity > 0:
                app.place_trade(ACTIVE_SYMBOL, 'BUY', quantity)

       # Continuing with the trading logic for the PASSIVE_SYMBOL signal
    elif not signal: # if signal is false then buy..
        if ACTIVE_SYMBOL in app.positions:
            # Sell SPY and use the proceeds plus any available funds to buy BOND
            spy_quantity = app.positions[ACTIVE_SYMBOL]['holdings']
            app.place_trade(ACTIVE_SYMBOL, 'SELL', spy_quantity)
            # Pause briefly to allow for the sale to be processed; in a real application, this would be handled differently
            time.sleep(1)
            available_funds += spy_quantity * price_spy  # Assuming this is the proceeds from the sale
            quantity = math.floor(available_funds / price_bond)
            if quantity > 0:
                app.place_trade(PASSIVE_SYMBOL, 'BUY', quantity)
        elif not app.positions:
            # Buy BOND with all available capital
            quantity = math.floor(available_funds / price_bond)
            if quantity > 0:
                app.place_trade(PASSIVE_SYMBOL, 'BUY', quantity)
        elif PASSIVE_SYMBOL in app.positions:
            # Buy additional BONDs based on available funds
            quantity = math.floor(available_funds / price_bond)
            if quantity > 0:
                app.place_trade(PASSIVE_SYMBOL, 'BUY', quantity)


    time.sleep(3)  # Warte, bis Trades verarbeitet sind
    app.disconnect()

if __name__ == "__main__":
    main()