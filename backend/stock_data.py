# backend/stock_data.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np

def get_yf_session():
    """Create a session with retry logic and proper headers"""
    session = requests.Session()
    
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    return session

def get_sample_stock_data(ticker: str):
    """Generate realistic sample data for demo purposes"""
    np.random.seed(hash(ticker) % (2**32))
    
    end_date = datetime.now()
    dates = pd.date_range(end=end_date, periods=365, freq='D')
    
    base_prices = {
        'RELIANCE.NS': 2500,
        'TCS.NS': 3800,
        'INFY.NS': 1500,
        'HDFCBANK.NS': 1600,
        'ICICIBANK.NS': 1000,
        'HINDUNILVR.NS': 2400,
        'ITC.NS': 450,
        'SBIN.NS': 600,
        'BHARTIARTL.NS': 1200,
        'KOTAKBANK.NS': 1800,
        'BAJFINANCE.NS': 7000,
        'LT.NS': 3500,
        'ASIANPAINT.NS': 3200,
        'MARUTI.NS': 11000,
        'TITAN.NS': 3400,
        'AAPL': 180,
        'GOOGL': 140,
        'MSFT': 380,
        'AMZN': 150,
        'TSLA': 250,
        'META': 350,
        'NVDA': 500,
    }
    base_price = base_prices.get(ticker, 2000)
    
    returns = np.random.normal(0.0005, 0.02, 365)
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices * np.random.uniform(0.98, 1.00, 365),
        'High': prices * np.random.uniform(1.00, 1.03, 365),
        'Low': prices * np.random.uniform(0.97, 1.00, 365),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 365)
    })
    
    return df

def get_stock_data(ticker: str, period: str = "1y"):
    """Fetch stock data from Yahoo Finance with fallback to sample data"""
    try:
        print(f"Attempting to fetch {ticker} from Yahoo Finance...")
        session = get_yf_session()
        stock = yf.Ticker(ticker, session=session)
        time.sleep(0.5)
        df = stock.history(period=period)
        
        if df.empty:
            print(f"Yahoo Finance returned empty data for {ticker}, using sample data")
            return get_sample_stock_data(ticker)
        
        df.reset_index(inplace=True)
        print(f"Successfully fetched {ticker} from Yahoo Finance")
        return df
    
    except Exception as e:
        print(f"Error fetching {ticker}: {e}. Using sample data for demo.")
        return get_sample_stock_data(ticker)

def get_current_price(ticker: str):
    """Get current stock price"""
    try:
        session = get_yf_session()
        stock = yf.Ticker(ticker, session=session)
        time.sleep(0.5)
        data = stock.history(period='1d')
        if not data.empty:
            return data['Close'].iloc[-1]
        
        df = get_sample_stock_data(ticker)
        return df['Close'].iloc[-1]
    except Exception as e:
        print(f"Error getting current price for {ticker}: {e}")
        df = get_sample_stock_data(ticker)
        return df['Close'].iloc[-1]

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
            "name": INDIAN_STOCKS.get(ticker, US_STOCKS.get(ticker, ticker)),
            "symbol": ticker,
            "sector": "Technology" if ticker in US_STOCKS else "Financial Services",
            "industry": "Software" if ticker in US_STOCKS else "Banking",
            "market_cap": 1000000000,
            "pe_ratio": 25.5,
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
    
    if pd.notna(latest['SMA_20']) and pd.notna(latest['SMA_50']):
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
    
    if pd.notna(latest['RSI']):
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
    
    if pd.notna(latest['MACD']) and pd.notna(latest['Signal_Line']):
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
    
    if pd.notna(latest['BB_Lower']) and pd.notna(latest['BB_Upper']):
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
