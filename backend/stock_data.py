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
        if symbol.endswith('.NS'):
            self.tickers_to_try.append(symbol.replace('.NS', '.BO'))
    
    def get_data(self, period: str = "1y") -> Optional[pd.DataFrame]:
        """Get stock data with NSE/BSE/US fallback + retries"""
        retry_count = 3
        
        for attempt in range(retry_count):
            for ticker in self.tickers_to_try:
                try:
                    print(f"🔄 Trying {ticker} (attempt {attempt + 1})")
                    
                    # Use Ticker method (more reliable than download)
                    stock = yf.Ticker(ticker)
                    df = stock.history(period=period, auto_adjust=True)
                    
                    if not df.empty and len(df) > 5:
                        print(f"✅ SUCCESS: {ticker} ({len(df)} days)")
                        df['ticker'] = ticker  # Add ticker column
                        return df
                        
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 404:
                        print(f"❌ 404: {ticker}")
                        continue
                    else:
                        print(f"❌ HTTP {e.response.status_code}: {ticker}")
                        break
                        
                except Exception as e:
                    print(f"❌ Error {ticker}: {str(e)[:50]}")
                    time.sleep(1)
                    continue
            
            if attempt < retry_count - 1:
                print(f"Retrying all tickers... (Attempt {attempt + 2}/{retry_count})")
                time.sleep(2)
        
        print(f"❌ ALL FAILED: {self.symbol}")
        return pd.DataFrame()  # Return empty DataFrame (no crash)
