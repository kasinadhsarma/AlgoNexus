import yfinance as yf
import pandas as pd
import logging
from datetime import datetime
import os

# Set up logging
logging.basicConfig(filename='yahoo_finance_data_fetch.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def fetch_stock_data(symbol, start_date, end_date):
    """
    Fetch stock data for a given symbol from Yahoo Finance API.
    """
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(start=start_date, end=end_date)
        return data
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def save_data_to_csv(data, symbol, output_dir):
    """
    Save the fetched data to a CSV file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = os.path.join(output_dir, f"{symbol}_data_{datetime.now().strftime('%Y%m%d')}.csv")
    try:
        data.to_csv(filename)
        logging.info(f"Data for {symbol} saved to {filename}")
    except IOError as e:
        logging.error(f"Error saving data for {symbol}: {str(e)}")

def main():
    symbols = ['AAPL', 'GOOGL', 'MSFT']  # Example stock symbols
    start_date = '2022-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    output_dir = 'src/data'  # Directory to save the data

    for symbol in symbols:
        logging.info(f"Fetching data for {symbol}")
        data = fetch_stock_data(symbol, start_date, end_date)
        if data is not None and not data.empty:
            save_data_to_csv(data, symbol, output_dir)
        else:
            logging.warning(f"No data fetched for {symbol}")


if __name__ == "__main__":
    main()
