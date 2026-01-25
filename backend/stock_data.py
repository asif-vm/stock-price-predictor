# backend/stock_data.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_stock_data(ticker: str, period: str = "1y"):
    """
    Fetch stock data from Yahoo Finance
    
    Args:
        ticker: Stock symbol (e.g., 'RELIANCE.NS', 'TCS.NS')
        period: Data period ('1mo', '3mo', '6mo', '1y', '2y', '5y')
    
    Returns:
        DataFrame with stock data
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            return None
        
        # Reset index to make Date a column
        df.reset_index(inplace=True)
        return df
    
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def get_current_price(ticker: str):
    """Get current stock price"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1d')
        if not data.empty:
            return data['Close'].iloc[-1]
        return None
    except:
        return None

def get_stock_info(ticker: str):
    """Get stock information"""
    try:
        stock = yf.Ticker(ticker)
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
    except:
        return {
            "name": ticker,
            "symbol": ticker,
            "sector": "N/A",
            "industry": "N/A"
        }

def calculate_technical_indicators(df: pd.DataFrame):
    """Calculate technical indicators"""
    
    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Exponential Moving Average
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    return df

def generate_signals(df: pd.DataFrame):
    """Generate buy/sell signals based on technical indicators"""
    
    signals = []
    latest = df.iloc[-1]
    
    # SMA Crossover Signal
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
    
    # RSI Signal
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
    
    # MACD Signal
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
    
    # Bollinger Bands Signal
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

# Popular Indian stocks
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

# US Stocks (optional)
US_STOCKS = {
    "AAPL": "Apple",
    "GOOGL": "Google",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "TSLA": "Tesla",
    "META": "Meta",
    "NVDA": "NVIDIA"
}