import time
import yfinance as yf
import requests

class StockData:
    def __init__(self, symbol):
        self.symbol = symbol

    def get_data(self):
        retry_count = 3
        for attempt in range(retry_count):
            try:
                data = yf.download(self.symbol)
                if data.empty:
                    raise ValueError("No data found for symbol: {self.symbol}")
                return data
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    if attempt < retry_count - 1:
                        print(f"404 not found for {self.symbol}. Retrying... (Attempt {attempt + 1})")
                        time.sleep(2)  # Sleep before retrying
                    else:
                        raise ValueError(f"Stock data retrieval failed: {e}")
                else:
                    raise
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                break
        return None

# Example usage
# stock_data = StockData('AAPL')
# print(stock_data.get_data())