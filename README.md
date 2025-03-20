# Moving Average Crossover ML Trading Bot

## Overview
This repository contains a Python-based trading bot that utilizes a Moving Average Crossover strategy enhanced with Machine Learning to execute trades on Binance Futures Testnet. The bot fetches historical data, calculates moving averages, predicts trading signals using a Logistic Regression model, and places automated trades with risk management strategies.

## Features
- **Historical Data Fetching**: Uses `yfinance` to retrieve past trading data.
- **Technical Analysis**: Implements Short-Term (10-day) and Long-Term (50-day) Moving Averages.
- **Machine Learning Model**: A logistic regression model trained on moving average differentials to predict trade signals.
- **Automated Trading**: Places buy/sell orders on Binance Futures Testnet using `ccxt`.
- **Risk Management**: Implements Take-Profit (5%) and Stop-Loss (3%) strategies.
- **Trade Logging**: Logs all executed trades in a CSV file for analysis.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.7+
- Required Python Libraries:

```bash
pip install yfinance pandas numpy scikit-learn ccxt
```

### Binance Testnet API Setup
1. Create a Binance Testnet account.
2. Generate API keys from [Binance Futures Testnet](https://testnet.binancefuture.com/).
3. Add your API credentials in the `config` dictionary in `aiBotTestnet.py`.

## Usage
### Running the Bot
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```
2. Run the script:
   ```bash
   python aiBotTestnet.py
   ```

### Configuration
Modify the following parameters in `aiBotTestnet.py` under the `config` dictionary:
```python
config = {
    'symbol': 'BTC/USDT',  # Trading pair
    'short_window': 10,  # Short moving average window
    'long_window': 50,  # Long moving average window
    'take_profit': 0.05,  # 5% Take Profit
    'stop_loss': 0.03,  # 3% Stop Loss
    'leverage': 2,  # Leverage amount
    'investment': 100,  # Initial investment
    'binance_api_key': 'your_api_key',
    'binance_api_secret': 'your_api_secret',
    'log_file': 'trades_log.csv'
}
```

## How It Works
1. **Fetch Market Data**: Retrieves historical price data from Yahoo Finance.
2. **Calculate Indicators**: Computes short and long-term moving averages.
3. **Train ML Model**: Uses past data to train a logistic regression model.
4. **Predict Trading Signal**: Determines whether to buy or sell based on trends.
5. **Execute Trades**: Places orders on Binance Futures Testnet.
6. **Monitor Open Trades**: Checks stop-loss and take-profit conditions.

## Logging and Monitoring
The bot logs trade data in `trades_log.csv` with the following columns:
- `timestamp`: Trade execution time
- `symbol`: Trading pair
- `direction`: Buy/Sell
- `order_size`: Order quantity
- `entry_price`: Trade entry price
- `exit_price`: Trade exit price
- `stop_loss`: Stop loss threshold
- `take_profit`: Take profit threshold

## Important Notes
- **Sandbox Mode**: This bot operates on Binance Testnet and will not execute real trades.
- **Use With Caution**: This bot is for educational purposes. Use real funds at your own risk.
- **API Key Security**: Never share your API keys. Store them securely.

## License
This project is licensed under the MIT License.

## Contributions
Feel free to fork, modify, and improve the bot. Contributions are welcome!

## Author
Developed by Your Name - [GitHub Profile](https://github.com/your-profile).

