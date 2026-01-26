import time
import yfinance as yf
import requests
import pandas as pd
from typing import Optional


class StockData:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.tickers_to_try = [symbol]

        # Auto-add BSE fallback for NSE tickers
        if symbol.endswith(".NS"):
            self.tickers_to_try.append(symbol.replace(".NS", ".BO"))

    def get_data(self, period: str = "1y") -> Optional[pd.DataFrame]:
        """Get stock data with NSE/BSE/US fallback + retries"""
        retry_count = 3

        for attempt in range(retry_count):
            for ticker in self.tickers_to_try:
                try:
                    print(f"Trying {ticker} (attempt {attempt + 1})")

                    stock = yf.Ticker(ticker)
                    df = stock.history(period=period, auto_adjust=True)

                    if not df.empty and len(df) > 5:
                        df["ticker"] = ticker
                        return df

                except requests.exceptions.HTTPError as e:
                    if e.response and e.response.status_code == 404:
                        continue
                    break

                except Exception:
                    time.sleep(1)
                    continue

            if attempt < retry_count - 1:
                time.sleep(2)

        return pd.DataFrame()


# ✅ THIS MUST BE OUTSIDE THE CLASS
def get_stock_data(symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
    stock = StockData(symbol)
    return stock.get_data(period)
