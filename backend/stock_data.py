# backend/stock_data.py

from __future__ import annotations

import yfinance as yf
import pandas as pd
from typing import List, Dict


# ==================================================
# Core stock data
# ==================================================

def get_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance.
    Always returns a DataFrame (empty if failure).
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if df is None or df.empty:
            return pd.DataFrame()

        return df

    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return pd.DataFrame()


def get_current_price(ticker: str) -> float | None:
    """
    Get latest closing price.
    """
    try:
        df = get_stock_data(ticker, period="5d")
        if df.empty:
            return None
        return float(df["Close"].iloc[-1])
    except Exception:
        return None


# ==================================================
# Stock metadata
# ==================================================

def get_stock_info(ticker: str) -> Dict:
    """
    Fetch stock metadata safely.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info if hasattr(stock, "info") else {}

        return {
            "name": info.get("longName", ticker),
            "symbol": ticker,
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            "currency": info.get("currency"),
            "exchange": info.get("exchange"),
        }

    except Exception:
        return {
            "name": ticker,
            "symbol": ticker,
            "sector": "N/A",
            "industry": "N/A",
        }


# ==================================================
# Technical indicators
# ==================================================

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate SMA, EMA, MACD, RSI, Bollinger Bands.
    """
    if df.empty:
        return df

    df = df.copy()

    # SMA
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()

    # EMA
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_Middle"] = mid
    df["BB_Upper"] = mid + (2 * std)
    df["BB_Lower"] = mid - (2 * std)

    return df


# ==================================================
# Trading signals
# ==================================================

def generate_signals(df: pd.DataFrame) -> List[Dict]:
    """
    Generate BUY / SELL signals.
    """
    signals: List[Dict] = []

    if df.empty or len(df) < 2:
        return signals

    latest = df.iloc[-1]

    if latest["SMA_20"] > latest["SMA_50"]:
        signals.append({"type": "BUY", "indicator": "SMA"})
    elif latest["SMA_20"] < latest["SMA_50"]:
        signals.append({"type": "SELL", "indicator": "SMA"})

    if latest["RSI"] < 30:
        signals.append({"type": "BUY", "indicator": "RSI"})
    elif latest["RSI"] > 70:
        signals.append({"type": "SELL", "indicator": "RSI"})

    if latest["MACD"] > latest["Signal_Line"]:
        signals.append({"type": "BUY", "indicator": "MACD"})
    elif latest["MACD"] < latest["Signal_Line"]:
        signals.append({"type": "SELL", "indicator": "MACD"})

    if latest["Close"] < latest["BB_Lower"]:
        signals.append({"type": "BUY", "indicator": "BB"})
    elif latest["Close"] > latest["BB_Upper"]:
        signals.append({"type": "SELL", "indicator": "BB"})

    return signals


# ==================================================
# Stock lists
# ==================================================

INDIAN_STOCKS = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank",
    "INFY.NS": "Infosys",
    "ICICIBANK.NS": "ICICI Bank",
    "ITC.NS": "ITC Limited",
    "SBIN.NS": "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel",
}

US_STOCKS = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "AMZN": "Amazon",
    "NVDA": "NVIDIA",
    "META": "Meta",
    "TSLA": "Tesla",
}
