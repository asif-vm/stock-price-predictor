# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd

from .stock_data import (
    get_stock_data, 
    get_current_price, 
    get_stock_info,
    calculate_technical_indicators,
    generate_signals,
    INDIAN_STOCKS,
    US_STOCKS
)
from .predictor import (
    linear_regression_prediction,
    advanced_prediction_with_features,
    calculate_price_targets
)

# Initialize FastAPI
app = FastAPI(
    title="Stock Price Prediction API",
    description="Real-time stock analysis and price prediction using ML",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class StockRequest(BaseModel):
    ticker: str
    period: str = "1y"

class PredictionRequest(BaseModel):
    ticker: str
    days: int = 7
    method: str = "advanced"  # "simple" or "advanced"

# Root endpoint
@app.get("/")
def read_root():
    return {
        "message": "Stock Price Prediction API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "stocks": "/stocks",
            "stock_data": "/stock/{ticker}",
            "predict": "/predict",
            "analysis": "/analysis/{ticker}",
            "signals": "/signals/{ticker}"
        }
    }

# Health check
@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "stock-prediction-api"}

# Get list of available stocks
@app.get("/stocks")
def get_stocks():
    """Get list of available stocks"""
    return {
        "indian_stocks": INDIAN_STOCKS,
        "us_stocks": US_STOCKS,
        "total": len(INDIAN_STOCKS) + len(US_STOCKS)
    }

# Get stock data
@app.get("/stock/{ticker}")
def get_stock(ticker: str, period: str = "1y"):
    """
    Get historical stock data
    
    Args:
        ticker: Stock symbol (e.g., 'RELIANCE.NS', 'AAPL')
        period: Time period ('1mo', '3mo', '6mo', '1y', '2y', '5y')
    """
    try:
        df = get_stock_data(ticker, period)
        
        if df is None or df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for ticker: {ticker}"
            )
        
        # Get stock info
        info = get_stock_info(ticker)
        
        # Convert DataFrame to dict
        data_dict = df.to_dict(orient='records')
        
        return {
            "ticker": ticker,
            "info": info,
            "data": data_dict[:100],  # Limit to 100 recent records for API
            "total_records": len(df),
            "period": period,
            "latest_price": float(df['Close'].iloc[-1]),
            "date_range": {
                "start": str(df['Date'].iloc[0]),
                "end": str(df['Date'].iloc[-1])
            }
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching stock data: {str(e)}"
        )

# Predict stock price
@app.post("/predict")
def predict_stock_price(request: PredictionRequest):
    """
    Predict future stock prices
    
    Args:
        ticker: Stock symbol
        days: Number of days to predict (1-30)
        method: Prediction method ('simple' or 'advanced')
    """
    try:
        if request.days < 1 or request.days > 30:
            raise HTTPException(
                status_code=400,
                detail="Days must be between 1 and 30"
            )
        
        # Get stock data
        df = get_stock_data(request.ticker, period="1y")
        
        if df is None or df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for ticker: {request.ticker}"
            )
        
        # Calculate technical indicators for advanced prediction
        if request.method == "advanced":
            df = calculate_technical_indicators(df)
            prediction_result = advanced_prediction_with_features(df, request.days)
        else:
            prediction_result = linear_regression_prediction(df, request.days)
        
        # Get current price
        current_price = get_current_price(request.ticker)
        
        return {
            "ticker": request.ticker,
            "method": request.method,
            "current_price": current_price or prediction_result['current_price'],
            "predictions": prediction_result['predictions'],
            "prediction_dates": prediction_result['dates'],
            "predicted_price": prediction_result['predicted_price'],
            "trend": prediction_result['trend'],
            "change_percentage": prediction_result['change_percentage'],
            "confidence": prediction_result.get('confidence', 75.0),
            "days_predicted": request.days
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

# Get technical analysis
@app.get("/analysis/{ticker}")
def get_technical_analysis(ticker: str):
    """
    Get technical analysis with indicators
    
    Args:
        ticker: Stock symbol
    """
    try:
        # Get stock data
        df = get_stock_data(ticker, period="1y")
        
        if df is None or df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for ticker: {ticker}"
            )
        
        # Calculate technical indicators
        df = calculate_technical_indicators(df)
        
        # Get latest values
        latest = df.iloc[-1]
        
        # Calculate price targets
        price_targets = calculate_price_targets(latest['Close'], df)
        
        return {
            "ticker": ticker,
            "current_price": float(latest['Close']),
            "date": str(latest['Date']),
            "indicators": {
                "SMA_20": float(latest['SMA_20']) if not pd.isna(latest['SMA_20']) else None,
                "SMA_50": float(latest['SMA_50']) if not pd.isna(latest['SMA_50']) else None,
                "SMA_200": float(latest['SMA_200']) if not pd.isna(latest['SMA_200']) else None,
                "RSI": float(latest['RSI']) if not pd.isna(latest['RSI']) else None,
                "MACD": float(latest['MACD']) if not pd.isna(latest['MACD']) else None,
                "Signal_Line": float(latest['Signal_Line']) if not pd.isna(latest['Signal_Line']) else None,
                "BB_Upper": float(latest['BB_Upper']) if not pd.isna(latest['BB_Upper']) else None,
                "BB_Lower": float(latest['BB_Lower']) if not pd.isna(latest['BB_Lower']) else None,
            },
            "price_targets": price_targets,
            "volume": float(latest['Volume'])
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

# Get trading signals
@app.get("/signals/{ticker}")
def get_trading_signals(ticker: str):
    """
    Get buy/sell trading signals
    
    Args:
        ticker: Stock symbol
    """
    try:
        # Get stock data
        df = get_stock_data(ticker, period="6mo")
        
        if df is None or df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for ticker: {ticker}"
            )
        
        # Calculate technical indicators
        df = calculate_technical_indicators(df)
        
        # Generate signals
        signals = generate_signals(df)
        
        # Count buy vs sell signals
        buy_signals = len([s for s in signals if s['type'] == 'BUY'])
        sell_signals = len([s for s in signals if s['type'] == 'SELL'])
        
        # Overall recommendation
        if buy_signals > sell_signals:
            recommendation = "BUY"
            confidence = min(95, 60 + (buy_signals - sell_signals) * 10)
        elif sell_signals > buy_signals:
            recommendation = "SELL"
            confidence = min(95, 60 + (sell_signals - buy_signals) * 10)
        else:
            recommendation = "HOLD"
            confidence = 50
        
        return {
            "ticker": ticker,
            "recommendation": recommendation,
            "confidence": confidence,
            "signals": signals,
            "signal_count": {
                "buy": buy_signals,
                "sell": sell_signals,
                "total": len(signals)
            },
            "current_price": float(df['Close'].iloc[-1])
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Signal generation failed: {str(e)}"
        )

# Compare multiple stocks
@app.post("/compare")
def compare_stocks(tickers: List[str]):
    """
    Compare multiple stocks
    
    Args:
        tickers: List of stock symbols
    """
    if len(tickers) > 5:
        raise HTTPException(
            status_code=400,
            detail="Maximum 5 stocks can be compared at once"
        )
    
    results = []
    
    for ticker in tickers:
        try:
            df = get_stock_data(ticker, period="1mo")
            if df is not None and not df.empty:
                current = df['Close'].iloc[-1]
                month_ago = df['Close'].iloc[0]
                change_pct = ((current - month_ago) / month_ago) * 100
                
                results.append({
                    "ticker": ticker,
                    "current_price": float(current),
                    "month_change_pct": float(change_pct),
                    "trend": "UP" if change_pct > 0 else "DOWN"
                })
        except:
            continue
    
    return {
        "comparison": results,
        "count": len(results)
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

