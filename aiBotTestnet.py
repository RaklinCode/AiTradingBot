import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import ccxt
import csv
import os
import time
from typing import Literal

class MovingAverageCrossoverML:
    def __init__(self, config):
        # Configuration parameters
        self.symbol = config.get('symbol', 'BTC/USDT')
        self.short_window = config.get('short_window', 10)
        self.long_window = config.get('long_window', 50)
        self.take_profit_pct = config.get('take_profit', 0.05)  # 5% take profit
        self.stop_loss_pct = config.get('stop_loss', 0.03)  # 3% stop loss
        self.leverage = config.get('leverage', 1)
        self.investment = config.get('investment', 100)
        self.binance_api_key = config.get('binance_api_key')
        self.binance_api_secret = config.get('binance_api_secret')
        self.log_file = config.get('log_file', 'trades_log.csv')

        # Initialize Binance Futures testnet exchange connection via ccxt.
        self.exchange = ccxt.binance({
            'apiKey': self.binance_api_key,
            'secret': self.binance_api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # Futures trading
                'adjustTime': True,       # Adjust timestamps automatically
            },
            'urls': {
                'api': {
                    'public': 'https://testnet.binancefuture.com/fapi/v1',
                    'private': 'https://testnet.binancefuture.com/fapi/v1',
                },
            },
            'test': True,  # Use testnet flag
        })
        self.exchange.set_sandbox_mode(True)  # Ensure sandbox mode is enabled

        # Load markets and synchronize time difference
        self.exchange.load_markets()
        self.exchange.load_time_difference()  # Sync server time difference

        # Set leverage for the trading pair
        self.set_leverage()

        # Set up the ML model (a simple logistic regression in this example)
        self.model = LogisticRegression()

        # Initialize CSV log file with headers
        self.initialize_log_file()

        # A list to store open orders/trades
        self.open_orders = []

    def set_leverage(self):
        """
        Sets leverage for the symbol on Binance Futures.
        Binance expects the symbol id without '/' (e.g., 'BTCUSDT').
        """
        try:
            market = self.exchange.market(self.symbol)
            response = self.exchange.fapiPrivate_post_leverage({
                'symbol': market['id'],
                'leverage': self.leverage,
            })
            print("Leverage set successfully:", response)
        except Exception as e:
            print("Error setting leverage:", e)

    def initialize_log_file(self):
        """
        Initializes a CSV log file with headers:
        timestamp, symbol, direction, order_size, entry_price, exit_price, stop_loss, take_profit.
        """
        headers = ['timestamp', 'symbol', 'direction', 'order_size', 'entry_price', 'exit_price', 'stop_loss',
                   'take_profit']
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
        else:
            print(f"Log file '{self.log_file}' already exists.")

    def log_trade(self, direction, order_size, entry_price, exit_price, stop_loss, take_profit):
        """
        Appends trade details to the CSV log file.
        """
        with open(self.log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([pd.Timestamp.now(), self.symbol, direction, order_size, entry_price, exit_price, stop_loss,
                             take_profit])
        print("Trade logged to CSV.")

    def fetch_data(self, period="1y", interval="1d"):
        """
        Downloads historical price data for the symbol using yfinance.
        Adjusts the symbol for yfinance: if the symbol ends with 'USDT', it changes it to 'USD'.
        """
        # Convert symbol for yfinance: BTC/USDT -> BTC-USD if necessary
        symbol_yf = self.symbol.replace('/', '-')
        if symbol_yf.endswith("USDT"):
            symbol_yf = symbol_yf.replace("USDT", "USD")
        data = yf.download(symbol_yf, period=period, interval=interval)
        data.dropna(inplace=True)
        return data

    def preprocess_data(self, data):
        """
        Preprocesses historical data by calculating short and long-term moving averages,
        generating trading signals, and returning the processed data.
        """
        # Calculate moving averages
        data['SMA_short'] = data['Close'].rolling(window=self.short_window).mean()
        data['SMA_long'] = data['Close'].rolling(window=self.long_window).mean()

        # Generate trading signals: 1 if SMA_short > SMA_long, -1 if less
        data['signal'] = 0
        data.loc[data['SMA_short'] > data['SMA_long'], 'signal'] = 1
        data.loc[data['SMA_short'] < data['SMA_long'], 'signal'] = -1

        data.dropna(inplace=True)
        return data

    def train_model(self, data):
        """
        Trains a machine learning model using the processed data.
        - Creates a feature as the difference between the short and long moving averages.
        - Shifts the signal column to use the next period's signal as the target.
        - Splits the data into training and testing sets.
        - Trains a logistic regression model and prints its accuracy.
        """
        data['feature'] = data['SMA_short'] - data['SMA_long']
        data['target'] = data['signal'].shift(-1)
        data.dropna(inplace=True)

        X = data[['feature']]
        y = data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)
        accuracy = self.model.score(X_test, y_test)
        print(f"Model Accuracy: {accuracy:.2f}")
        return accuracy

    def predict_signal(self, current_data):
        """
        Predicts the next trading signal using the trained machine learning model.
        Expects current_data to have scalar floats for 'SMA_short' and 'SMA_long'.
        """
        # Ensure the model is trained
        if not hasattr(self.model, 'coef_'):
            raise ValueError("Model has not been trained. Please train the model before predicting.")

        # Calculate the difference and ensure correct shape
        diff = current_data['SMA_short'] - current_data['SMA_long']
        feature_value = np.array([[diff]])  # Ensures shape is (1, 1)
        print("Feature value shape:", feature_value.shape)

        prediction = self.model.predict(feature_value)
        print(f"Predicted signal: {prediction[0]}")
        return prediction[0]

    def place_order(self, signal):
        """
        Places a futures order on Binance testnet based on the predicted signal.
        """
        # Explicitly annotate side as a literal type
        side: Literal['buy', 'sell'] = 'buy' if signal == 1 else 'sell'
        ticker = self.exchange.fetch_ticker(self.symbol)
        current_price = ticker['last']
        amount = (self.investment * self.leverage) / current_price

        # Calculate stop loss and take profit levels based on order side
        if side == 'buy':
            stop_loss = current_price * (1 - self.stop_loss_pct)
            take_profit = current_price * (1 + self.take_profit_pct)
        else:
            stop_loss = current_price * (1 + self.stop_loss_pct)
            take_profit = current_price * (1 - self.take_profit_pct)

        print(f"Placing {side} order for {amount:.4f} {self.symbol} at {current_price}")

        try:
            order = self.exchange.create_order(
                symbol=self.symbol,
                type='market',
                side=side,
                amount=amount,
                params={'leverage': self.leverage, 'recvWindow': 10000}
            )
            # Record order details
            order_record = {
                'order_id': order.get('id', 'demo_order'),
                'direction': side,
                'order_size': amount,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            self.open_orders.append(order_record)
            print("Order placed and recorded in open orders.")
        except Exception as e:
            print("Error placing order:", e)

    def monitor_trades(self):
        """
        Monitors open orders and closes trades if price hits stop loss or take profit.
        """
        ticker = self.exchange.fetch_ticker(self.symbol)
        current_price = ticker['last']
        print(f"Monitoring trades at current price: {current_price}")

        for order in self.open_orders.copy():
            direction = order['direction']
            entry_price = order['entry_price']
            stop_loss = order['stop_loss']
            take_profit = order['take_profit']

            if direction == 'buy':
                if current_price <= stop_loss or current_price >= take_profit:
                    print(f"Triggering close for buy order at {current_price}")
                    self.close_order(order, current_price)
            elif direction == 'sell':
                if current_price >= stop_loss or current_price <= take_profit:
                    print(f"Triggering close for sell order at {current_price}")
                    self.close_order(order, current_price)

    def close_order(self, order, exit_price):
        """
        Closes an open order by placing the opposite market order.
        """
        closing_side: Literal['buy', 'sell'] = 'sell' if order['direction'] == 'buy' else 'buy'
        print(f"Closing order {order['order_id']} with a {closing_side} at {exit_price}")

        try:
            close_order = self.exchange.create_order(
                symbol=self.symbol,
                type='market',
                side=closing_side,
                amount=order['order_size']
            )
            self.log_trade(
                direction=order['direction'],
                order_size=order['order_size'],
                entry_price=order['entry_price'],
                exit_price=exit_price,
                stop_loss=order['stop_loss'],
                take_profit=order['take_profit']
            )
            self.open_orders.remove(order)
            print(f"Order {order['order_id']} closed and logged.")
        except Exception as e:
            print("Error closing order:", e)

    def sleep_with_details(self, total_minutes):
        """
        Sleeps for a given number of minutes with minute-by-minute updates.
        """
        for minute in range(1, total_minutes + 1):
            print(f"Sleeping... {minute} minute(s) passed out of {total_minutes}")
            time.sleep(60)

    def run_strategy(self):
        """
        Orchestrates the trading process continuously (24/7).
        """
        print("Starting Moving Average Crossover ML Strategy on Binance Futures Testnet.")

        while True:
            try:
                # Retrieve and preprocess market data
                data = self.fetch_data()
                processed_data = self.preprocess_data(data)

                # Update the ML model with the latest market trends
                self.train_model(processed_data)

                # Extract scalar values from the latest row to avoid FutureWarnings
                latest_row = processed_data.iloc[-1]
                sma_short = float(latest_row['SMA_short'].item())
                sma_long = float(latest_row['SMA_long'].item())
                current_data = {
                    'SMA_short': sma_short,
                    'SMA_long': sma_long
                }

                # Predict the next trading signal (1 for BUY, -1 for SELL)
                predicted_signal = self.predict_signal(current_data)
                signal_text = "BUY" if predicted_signal == 1 else "SELL"
                print(f"Predicted signal: {signal_text}")

                # Place a trade based on the predicted signal
                self.place_order(predicted_signal)

                # Monitor open trades for stop loss or take profit conditions
                self.monitor_trades()

            except Exception as e:
                print("Error encountered during strategy execution:", e)

            # Sleep between iterations; adjust the sleep duration as needed
            self.sleep_with_details(1)

if __name__ == "__main__":
    # Replace with your Binance Futures testnet API credentials.
    config = {
        'symbol': 'BTC/USDT',
        'short_window': 10,
        'long_window': 50,
        'take_profit': 0.05,
        'stop_loss': 0.03,
        'leverage': 2,
        'investment': 100,
        'binance_api_key': '2b61bb8697a5967f5d088d7fd30fc38b59d6ef14cc138e93a96d239c8b694bcd',
        'binance_api_secret': '7380053c52328a4250b2c5c7f7e83d7d7f2181e3276829c80ff73e49bf048151',
        'log_file': 'trades_log.csv'
    }

    # Initialize and run the strategy continuously (24/7)
    strategy = MovingAverageCrossoverML(config)
    strategy.run_strategy()
