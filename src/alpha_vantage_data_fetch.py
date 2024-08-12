import requests
import os
import json
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(filename='alpha_vantage_data_fetch.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Alpha Vantage API configuration
API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')  # Store your API key as an environment variable
BASE_URL = 'https://www.alphavantage.co/query'

def fetch_daily_stock_data(symbol):
    """
    Fetch daily stock data for a given symbol from Alpha Vantage API.
    """
    function = 'TIME_SERIES_DAILY'
    params = {
        'function': function,
        'symbol': symbol,
        'apikey': API_KEY
    }

    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()

        if 'Error Message' in data:
            logging.error(f"Error fetching data for {symbol}: {data['Error Message']}")
            return None

        return data
    except requests.RequestException as e:
        logging.error(f"Request failed for {symbol}: {str(e)}")
        return None

def save_data_to_file(data, symbol):
    """
    Save the fetched data to a JSON file.
    """
    filename = f"{symbol}_daily_data_{datetime.now().strftime('%Y%m%d')}.json"
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        logging.info(f"Data for {symbol} saved to {filename}")
    except IOError as e:
        logging.error(f"Error saving data for {symbol}: {str(e)}")

def main():
    symbols = ['AAPL', 'GOOGL', 'MSFT']  # Example stock symbols

    for symbol in symbols:
        logging.info(f"Fetching data for {symbol}")
        data = fetch_daily_stock_data(symbol)
        if data:
            save_data_to_file(data, symbol)
        else:
            logging.warning(f"No data fetched for {symbol}")


if __name__ == "__main__":
    if not API_KEY:
        logging.error("Alpha Vantage API key not found. Please set the ALPHA_VANTAGE_API_KEY environment variable.")
    else:
        main()
