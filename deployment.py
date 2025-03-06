import os
import sys
import importlib
import pandas as pd
import time

# Define Virtual Environment Paths
VENV_5PAISA = "/home/ubuntu/env_5paisa"
VENV_KOTAKNEO = "/home/ubuntu/env_kotakneo"

# Function to manually activate virtual environment
def activate_venv(venv_path):
    site_packages = os.path.join(venv_path, "lib", "python3.10", "site-packages")  # Adjust for Python version
    if os.path.exists(site_packages):
        sys.path.insert(0, site_packages)  # Add site-packages to sys.path
    else:
        print(f"[ERROR] Virtual environment not found: {venv_path}")
        sys.exit(1)

# Activate 5Paisa Virtual Environment and Import
activate_venv(VENV_5PAISA)
try:
    py5paisa = importlib.import_module("py5paisa")
    FivePaisaClient = py5paisa.FivePaisaClient
except ModuleNotFoundError:
    print("[ERROR] py5paisa module not found. Install it inside env_5paisa.")
    sys.exit(1)

# Activate Kotak Neo Virtual Environment and Import
activate_venv(VENV_KOTAKNEO)
try:
    neo_api_client = importlib.import_module("neo_api_client")
    NeoAPI = neo_api_client.NeoAPI
except ModuleNotFoundError:
    print("[ERROR] neo_api_client module not found. Install it inside env_kotakneo.")
    sys.exit(1)

# 5Paisa API Credentials
cred = {
    "APP_NAME": "5P50289032",
    "APP_SOURCE": "22145",
    "USER_ID": "jv0zaXaW7lD",
    "PASSWORD": "ZusnUUqsJoh",
    "USER_KEY": "24BLhwIxzMHo31rotJYypWuvYUU4mCHZ",
    "ENCRYPTION_KEY": "FanCs8NKjzunmTmGXgxkOPYS5QUwsXvU"
}
client_5paisa = FivePaisaClient(cred=cred)

# Kotak Neo API Credentials
client_neo = NeoAPI(
    consumer_key="fmHOCOoINQuyTfdB8S_aiiWMdlQa",
    consumer_secret="xjI_osC4q4r4zkWbFpq_Vgw4LTga",
    environment='prod',
    access_token=None,
    neo_fin_key=None
)

# Kotak Neo Login
client_neo.login(mobilenumber="+916303008951", password="Avks@1234")
client_neo.session_2fa(OTP="271707")

# Function to Fetch Live Prices
def get_price(scrip):
    req_data = [{"Exch": "N", "ExchType": "C", "ScripData": scrip}]
    try:
        response = client_5paisa.fetch_market_feed_scrip(req_data)
        return response['Data'][0]['LastRate']
    except Exception as e:
        print(f"[ERROR] Failed to fetch price: {e}")
        return None

# Mean Reversion Strategy Using Bollinger Bands
def mean_reversion_strategy(scrip, period=20, deviation=2):
    prices = []
    while True:
        price = get_price(scrip)
        if price:
            prices.append(price)
            if len(prices) > period:
                prices.pop(0)
            
            if len(prices) >= period:
                df = pd.DataFrame(prices, columns=['Close'])
                df['SMA'] = df['Close'].rolling(period).mean()
                df['Upper'] = df['SMA'] + (df['Close'].rolling(period).std() * deviation)
                df['Lower'] = df['SMA'] - (df['Close'].rolling(period).std() * deviation)
                
                last_price = df['Close'].iloc[-1]
                upper_band = df['Upper'].iloc[-1]
                lower_band = df['Lower'].iloc[-1]
                
                if last_price <= lower_band:
                    place_order(scrip, 'BUY')
                elif last_price >= upper_band:
                    place_order(scrip, 'SELL')
        
        time.sleep(60)  # Fetch price every minute

# Trade Execution using Kotak Neo API
def place_order(scrip, transaction_type):
    client_neo.place_order(
        exchange_segment='nse_cm',
        product='MIS',
        price='0',
        order_type='MKT',
        quantity=5,
        validity='DAY',
        trading_symbol=scrip,
        transaction_type=transaction_type
    )
    print(f"{transaction_type} Order placed for {scrip}")

# Run the Strategy
scrip = "ITC"
mean_reversion_strategy(scrip)
