import ccxt

# Replace these with your API key and secret
api_key = '8fNnp25yUwSEBt5FRDJuTK5wwUU342TjX63TJSmORA3Fo6g35qY4ppB3uss0YTGH'
api_secret = 'khAEW0Y18b6ByjSReobkquTj5MZuPZfFVnEOUf7MEWuxJx01HhK8GcYpKFFXLzVb'

# Initialize the Binance exchange
binance = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
})

# Fetch account balance
balance = binance.fetch_balance()

# Print available balances (non-zero)
for currency, details in balance['total'].items():
    if details > 0:
        available = balance['free'][currency]
        print(f'{currency}: {available}')