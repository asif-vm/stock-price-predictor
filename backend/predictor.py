# backend/predictor.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

def linear_regression_prediction(df: pd.DataFrame, days: int = 7):
    """Predict stock prices using Linear Regression"""
    df_copy = df.copy()
    df_copy['Days'] = range(len(df_copy))
    
    train_data = df_copy.tail(90)
    
    X = train_data[['Days']].values
    y = train_data['Close'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    last_day = df_copy['Days'].iloc[-1]
    future_days = np.array([[last_day + i] for i in range(1, days + 1)])
    predictions = model.predict(future_days)
    
    trend = "Upward" if predictions[-1] > y[-1] else "Downward"
    change_pct = ((predictions[-1] - y[-1]) / y[-1]) * 100
    
    return {
        "predictions": predictions.tolist(),
        "dates": [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') 
                  for i in range(1, days + 1)],
        "current_price": float(y[-1]),
        "predicted_price": float(predictions[-1]),
        "trend": trend,
        "change_percentage": float(change_pct),
        "model_score": float(model.score(X, y))
    }

def advanced_prediction_with_features(df: pd.DataFrame, days: int = 7):
    """Advanced prediction using multiple features"""
    features = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'MACD']
    
    available_features = [f for f in features if f in df.columns]
    
    if len(available_features) < 3:
        return linear_regression_prediction(df, days)
    
    train_data = df[available_features].dropna().tail(90)
    
    if len(train_data) < 30:
        return linear_regression_prediction(df, days)
    
    X = train_data.drop('Close', axis=1).values
    y = train_data['Close'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    last_features = train_data.drop('Close', axis=1).iloc[-1].values.reshape(1, -1)
    
    predictions = []
    current_features = last_features.copy()
    
    for i in range(days):
        pred = model.predict(current_features)[0]
        predictions.append(pred)
        
        if len(current_features[0]) > 0:
            current_features[0][0] = pred
    
    recent_volatility = df['Close'].tail(30).std()
    confidence = max(50, min(95, 85 - (recent_volatility / df['Close'].iloc[-1] * 100)))
    
    return {
        "predictions": predictions,
        "dates": [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') 
                  for i in range(1, days + 1)],
        "current_price": float(df['Close'].iloc[-1]),
        "predicted_price": float(predictions[-1]),
        "trend": "Upward" if predictions[-1] > df['Close'].iloc[-1] else "Downward",
        "change_percentage": float(((predictions[-1] - df['Close'].iloc[-1]) / df['Close'].iloc[-1]) * 100),
        "confidence": float(confidence),
        "model_type": "Multi-Feature Linear Regression"
    }

def calculate_price_targets(current_price: float, df: pd.DataFrame):
    """Calculate support and resistance levels"""
    
    recent_high = df['High'].tail(30).max()
    recent_low = df['Low'].tail(30).min()
    
    volatility = df['Close'].tail(30).std()
    
    targets = {
        "current_price": float(current_price),
        "support_levels": [
            float(current_price - volatility),
            float(current_price - (volatility * 2)),
            float(recent_low)
        ],
        "resistance_levels": [
            float(current_price + volatility),
            float(current_price + (volatility * 2)),
            float(recent_high)
        ],
        "stop_loss": float(current_price - (volatility * 1.5)),
        "take_profit": float(current_price + (volatility * 2))
    }
    
    return targets
