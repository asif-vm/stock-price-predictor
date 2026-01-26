# backend/main.py

import os
from typing import List
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .stock_data import (
    get_stock_data,
    get_current_price,
    get_stock_info,
    calculate_technical_indicators,
    generate_signals,
    INDIAN_STOCKS,
    US_STOCKS,
)

from .predictor import (
    linear_regression_prediction,
    advanced_prediction_with_features,
    calculate_price_targets,
)

# --------------------------------------------------
# FastAPI initialization
# --------------------------------------------------

app = FastAPI(
    title="Stock Price Prediction API",
    description="Real-time stock analysis and ML-based price prediction",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Pydantic models
# --------------------------------------------------

class PredictionRequest(BaseModel):
    ticker: str
    days: int = 7
    method: str = "advanced"  # simple | advanced


# --------------------------------------------------
# Root & health
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Stock Price Prediction API",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


# --------------------------------------------------
# Stock list
# --------------------------------------------------

@app.get("/stocks")
def list_stocks():
    return {
        "indian_stocks": INDIAN_STOCKS,
        "us_stocks": US_STOCKS,
        "total": len(INDIAN_STOCKS) + len(US_STOCKS),
    }


# --------------------------------------------------
# Historical stock data
# --------------------------------------------------

@app.get("/stock/{ticker}")
def get_stock(ticker: str, period: str = "1y"):
    try:
        df = get_stock_data(ticker, period)

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No data found")

        df = df.reset_index()

        info = get_stock_info(ticker)

        return {
            "ticker": ticker,
            "info": info,
            "period": period,
            "total_records": len(df),
            "latest_price": float(df["Close"].iloc[-1]),
            "date_range": {
                "start": str(df["Date"].iloc[0]),
                "end": str(df["Date"].iloc[-1]),
            },
            "data": df.tail(100).to_dict(orient="records"),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------
# Price prediction
# --------------------------------------------------

@app.post("/predict")
def predict_price(request: PredictionRequest):
    if request.days < 1 or request.days > 30:
        raise HTTPException(status_code=400, detail="Days must be between 1 and 30")

    try:
        df = get_stock_data(request.ticker, "1y")

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No data found")

        df = df.reset_index()

        if request.method == "advanced":
            df = calculate_technical_indicators(df)
            result = advanced_prediction_with_features(df, request.days)
        else:
            result = linear_regression_prediction(df, request.days)

        current_price = get_current_price(request.ticker)

        return {
            "ticker": request.ticker,
            "method": request.method,
            "current_price": current_price,
            "predicted_price": result["predicted_price"],
            "predictions": result["predictions"],
            "dates": result["dates"],
            "trend": result["trend"],
            "change_percentage": result["change_percentage"],
            "confidence": result.get("confidence", 75.0),
            "days_predicted": request.days,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------
# Technical analysis
# --------------------------------------------------

@app.get("/analysis/{ticker}")
def technical_analysis(ticker: str):
    try:
        df = get_stock_data(ticker, "1y")

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No data found")

        df = df.reset_index()
        df = calculate_technical_indicators(df)

        latest = df.iloc[-1]
        targets = calculate_price_targets(latest["Close"], df)

        return {
            "ticker": ticker,
            "current_price": float(latest["Close"]),
            "date": str(latest["Date"]),
            "indicators": {
                "SMA_20": latest.get("SMA_20"),
                "SMA_50": latest.get("SMA_50"),
                "SMA_200": latest.get("SMA_200"),
                "RSI": latest.get("RSI"),
                "MACD": latest.get("MACD"),
            },
            "price_targets": targets,
            "volume": float(latest["Volume"]),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------
# Trading signals
# --------------------------------------------------

@app.get("/signals/{ticker}")
def trading_signals(ticker: str):
    try:
        df = get_stock_data(ticker, "6mo")

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No data found")

        df = df.reset_index()
        df = calculate_technical_indicators(df)
        signals = generate_signals(df)

        buy = len([s for s in signals if s["type"] == "BUY"])
        sell = len([s for s in signals if s["type"] == "SELL"])

        if buy > sell:
            recommendation = "BUY"
        elif sell > buy:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"

        return {
            "ticker": ticker,
            "recommendation": recommendation,
            "signals": signals,
            "current_price": float(df["Close"].iloc[-1]),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------
# Compare stocks
# --------------------------------------------------

@app.post("/compare")
def compare_stocks(tickers: List[str]):
    if len(tickers) > 5:
        raise HTTPException(status_code=400, detail="Max 5 tickers allowed")

    results = []

    for ticker in tickers:
        try:
            df = get_stock_data(ticker, "1mo")
            if df is None or df.empty:
                continue

            df = df.reset_index()
            start = df["Close"].iloc[0]
            end = df["Close"].iloc[-1]
            change = ((end - start) / start) * 100

            results.append(
                {
                    "ticker": ticker,
                    "current_price": float(end),
                    "month_change_pct": round(change, 2),
                    "trend": "UP" if change > 0 else "DOWN",
                }
            )
        except Exception:
            continue

    return {"count": len(results), "comparison": results}


# --------------------------------------------------
# Entry point (Render compatible)
# --------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False,
    )
