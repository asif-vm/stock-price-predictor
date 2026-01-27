# backend/stock_data.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def get_yf_session():
    """Create a session with retry logic and proper headers"""
    session = requests.Session()
    
    # Add headers to avoid being blocked
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    # Add retry logic
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    return session

def get_stock_data(ticker: str, period: str = "1y"):
    """
    Fetch stock data from Yahoo Finance
    """
    try:
        session = get_yf_session()
        stock = yf.Ticker(ticker, session=session)
        time.sleep(0.5)
        df = stock.history(period=period)
        
        if df.empty:
            print(f"No data returned for {ticker}")
            return None
        
        df.reset_index(inplace=True)
        return df
    
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def get_current_price(ticker: str):
    """Get current stock price"""
    try:
        session = get_yf_session()
        stock = yf.Ticker(ticker, session=session)
        time.sleep(0.5)
        data = stock.history(period='1d')
        if not data.empty:
            return data['Close'].iloc[-1]
        return None
    except Exception as e:
        print(f"Error getting current price for {ticker}: {e}")
        return None

def get_stock_info(ticker: str):
    """Get stock information"""
    try:
        session = get_yf_session()
        stock = yf.Ticker(ticker, session=session)
        time.sleep(0.5)
        info = stock.info
        
        return {
            "name": info.get('longName', ticker),
            "symbol": ticker,
            "sector": info.get('sector', 'N/A'),
            "industry": info.get('industry', 'N/A'),
            "market_cap": info.get('marketCap', 0),
            "pe_ratio": info.get('trailingPE', 0),
            "52_week_high": info.get('fiftyTwoWeekHigh', 0),
            "52_week_low": info.get('fiftyTwoWeekLow', 0),
        }
    except Exception as e:
        print(f"Error getting info for {ticker}: {e}")
        return {
            "name": ticker,
            "symbol": ticker,
            "sector": "N/A",
            "industry": "N/A",
            "market_cap": 0,
            "pe_ratio": 0,
            "52_week_high": 0,
            "52_week_low": 0,
        }

def calculate_technical_indicators(df: pd.DataFrame):
    """Calculate technical indicators"""
    
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    return df

def generate_signals(df: pd.DataFrame):
    """Generate buy/sell signals based on technical indicators"""
    
    signals = []
    latest = df.iloc[-1]
    
    if latest['SMA_20'] > latest['SMA_50']:
        signals.append({
            "type": "BUY",
            "indicator": "SMA Crossover",
            "reason": "20-day SMA crossed above 50-day SMA",
            "strength": "Medium"
        })
    elif latest['SMA_20'] < latest['SMA_50']:
        signals.append({
            "type": "SELL",
            "indicator": "SMA Crossover",
            "reason": "20-day SMA crossed below 50-day SMA",
            "strength": "Medium"
        })
    
    if latest['RSI'] < 30:
        signals.append({
            "type": "BUY",
            "indicator": "RSI",
            "reason": f"RSI is oversold at {latest['RSI']:.2f}",
            "strength": "Strong"
        })
    elif latest['RSI'] > 70:
        signals.append({
            "type": "SELL",
            "indicator": "RSI",
            "reason": f"RSI is overbought at {latest['RSI']:.2f}",
            "strength": "Strong"
        })
    
    if latest['MACD'] > latest['Signal_Line']:
        signals.append({
            "type": "BUY",
            "indicator": "MACD",
            "reason": "MACD crossed above signal line",
            "strength": "Medium"
        })
    elif latest['MACD'] < latest['Signal_Line']:
        signals.append({
            "type": "SELL",
            "indicator": "MACD",
            "reason": "MACD crossed below signal line",
            "strength": "Medium"
        })
    
    if latest['Close'] < latest['BB_Lower']:
        signals.append({
            "type": "BUY",
            "indicator": "Bollinger Bands",
            "reason": "Price below lower band (oversold)",
            "strength": "Medium"
        })
    elif latest['Close'] > latest['BB_Upper']:
        signals.append({
            "type": "SELL",
            "indicator": "Bollinger Bands",
            "reason": "Price above upper band (overbought)",
            "strength": "Medium"
        })
    
    return signals

INDIAN_STOCKS = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank",
    "INFY.NS": "Infosys",
    "ICICIBANK.NS": "ICICI Bank",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ITC.NS": "ITC Limited",
    "SBIN.NS": "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "BAJFINANCE.NS": "Bajaj Finance",
    "LT.NS": "Larsen & Toubro",
    "ASIANPAINT.NS": "Asian Paints",
    "MARUTI.NS": "Maruti Suzuki",
    "TITAN.NS": "Titan Company"
}

US_STOCKS = {
    "AAPL": "Apple",
    "GOOGL": "Google",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "TSLA": "Tesla",
    "META": "Meta",
    "NVDA": "NVIDIA"
}
