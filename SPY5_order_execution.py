from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
import threading
import yfinance as yf
import time
from SPY5_logic import preparing_df, get_date_today, SMA_signal, VaR_signal, Rebound_signal, VaR_threshold
import math

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
        contract.secType = 'STK'
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
    # Instanz der API-Wrapper-Klasse erstellen
    app = IBapi()

    # Verbindung zum IB-Server herstellen
    app.connect('127.0.0.1', 4001, 99)

    # Start the socket in a thread
    api_thread = threading.Thread(target=run_loop, args=(app,), daemon=True)
    api_thread.start()

    time.sleep(3)  # sleep for a few seconds to let the server send the next valid order ID

    # Auf die Order ID warten
    while app.next_order_id is None:
        print('Waiting for valid order ID...')
        time.sleep(1)

    app.reqPositions()

    # Account Summary abfragen
    app.reqAccountSummary(1, 'All', '$LEDGER:ALL')
    
    time.sleep(3)

    app.positions
    time.sleep(3)

    print(app.positions)  # print the positions to the console

    # requests current SPY5 price
    buy_price = get_price("SPY")
    print(buy_price)
    available_funds = float(app.account_summary["TotalCashBalance"])
    df = preparing_df()
    date_today = get_date_today(df)

    signal_SMA = SMA_signal(df,date_today)
    signal_VaR = VaR_signal(df,date_today)
    signal_rebound = Rebound_signal(df,date_today)
    threshold_VaR = VaR_threshold(df,date_today)

    print("signal_SMA:", signal_SMA)
    print("signal_VaR:", signal_VaR)
    print("signal_rebound:", signal_rebound)
    print("Below our threshold_VaR:", threshold_VaR)
    if (threshold_VaR and (signal_SMA or signal_VaR or signal_rebound)): 

        
        quantity = math.floor(available_funds / buy_price) # diese Funktion rundet eine Dezimalzahl ab
        print("I would by now!"+ " Quanity:", quantity)
        if quantity >= 1:
            # ontract-Objekt für den Micro E-Mini S&P 500 Stock Price Index Future
            contract = Contract()
            contract.symbol = 'SPY'
            contract.secType = 'STK'
            # contract.LastTradeDateOrContractMonth ='202406' #this needs to be adjusted quarterly YYYYMM -> Mar, Jun, Sep, Dec
            contract.exchange = 'SMART'
            contract.currency = 'USD'
            # Erstellen Sie das Order-Objekt
            order = Order()
            order.action = 'BUY'
            order.totalQuantity = quantity
            order.orderType = 'MOC' ## "MKT" --> if error in this line
            order.eTradeOnly = False
            order.firmQuoteOnly = False

            # Platzieren Sie die Order
            app.placeOrder(app.next_order_id, contract, order)
        else:
            print("No remaining available funds to buy additional assets!")
    else:
        variables = {'signal_SMA': signal_SMA, 'signal_VaR': signal_VaR, 'signal_rebound': signal_rebound}
        for k,v in variables.items():
            if not v:
                print(f"The Signal of {k}, was False, therefore we hedge our exposure and short the market!")

                print("I sell now!"+ " Quanity:", quantity)
        amount_after_selling = sum([ data['holdings'] * data['avgCost'] for symbol, data in app.positions.items()])

        if amount_after_selling >= 0:
            for symbol, data in app.positions.items():
                contract = Contract()
                contract.symbol = symbol
                contract.secType = 'STK'  
                contract.exchange = 'SMART'  # SMART für automatische Routenwahl 
                contract.currency = 'USD'  

                order = Order()
                order.action = 'SELL'
                order.totalQuantity = data['holdings']  # Verkauf der gesamten gehaltene Menge dieser posi
                order.orderType = 'MKT'  # Sofort
                order.eTradeOnly = False
                order.firmQuoteOnly = False

                # Platzieren der Order
                app.placeOrder(app.next_order_id, contract, order)

                # Aktualisieren der next_order_id, um sicherzustellen, dass jede Order eine eindeutige ID hat
                app.next_order_id += 1

                time.sleep(1)
            
    time.sleep(3)  # sleep to allow order to be processed before disconnecting

    # disconnect after the order is placed
    app.disconnect()



if __name__ == "__main__":
    main()