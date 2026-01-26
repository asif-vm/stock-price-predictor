# =========================
# Stock metadata
# =========================

def get_stock_info(symbol: str) -> dict:
    try:
        stock = yf.Ticker(symbol)
        info = stock.info or {}

        return {
            "symbol": symbol,
            "name": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "currency": info.get("currency"),
            "exchange": info.get("exchange"),
        }
    except Exception:
        return {"symbol": symbol}


# =========================
# Technical indicators
# =========================

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2

    return df


# =========================
# Trading signals
# =========================

def generate_signals(df: pd.DataFrame) -> list[dict]:
    signals = []

    if len(df) < 2:
        return signals

    latest = df.iloc[-1]

    if latest["RSI"] < 30:
        signals.append({"type": "BUY", "reason": "RSI oversold"})

    if latest["RSI"] > 70:
        signals.append({"type": "SELL", "reason": "RSI overbought"})

    if latest["Close"] > latest["SMA_50"]:
        signals.append({"type": "BUY", "reason": "Price above SMA 50"})

    if latest["Close"] < latest["SMA_50"]:
        signals.append({"type": "SELL", "reason": "Price below SMA 50"})

    return signals


# =========================
# Stock lists
# =========================

INDIAN_STOCKS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
]

US_STOCKS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
]
