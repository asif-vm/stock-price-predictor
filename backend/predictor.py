# backend/predictor.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

def prepare_data_for_prediction(df: pd.DataFrame, lookback: int = 60):
    """
    Prepare data for ML prediction using sliding window
    
    Args:
        df: DataFrame with stock data
        lookback: Number of days to look back
    
    Returns:
        X, y, scaler for training
    """
    # Use Close price
    data = df['Close'].values.reshape(-1, 1)
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    return X, y, scaler

def simple_moving_average_prediction(df: pd.DataFrame, days: int = 7):
    """
    Simple prediction using moving average
    
    Args:
        df: DataFrame with stock data
        days: Number of days to predict
    
    Returns:
        List of predicted prices
    """
    # Calculate moving average of last 30 days
    ma = df['Close'].tail(30).mean()
    
    # Simple trend calculation
    recent_prices = df['Close'].tail(10).values
    trend = (recent_prices[-1] - recent_prices[0]) / len(recent_prices)
    
    predictions = []
    last_price = df['Close'].iloc[-1]
    
    for i in range(1, days + 1):
        # Predict with trend and some random variation
        predicted_price = last_price + (trend * i) + np.random.normal(0, trend * 0.1)
        predictions.append(predicted_price)
    
    return predictions

def linear_regression_prediction(df: pd.DataFrame, days: int = 7):
    """
    Predict stock prices using Linear Regression
    
    Args:
        df: DataFrame with stock data
        days: Number of days to predict
    
    Returns:
        Dictionary with predictions and metrics
    """
    # Prepare data
    df_copy = df.copy()
    df_copy['Days'] = range(len(df_copy))
    
    # Use last 90 days for training
    train_data = df_copy.tail(90)
    
    X = train_data[['Days']].values
    y = train_data['Close'].values
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict future
    last_day = df_copy['Days'].iloc[-1]
    future_days = np.array([[last_day + i] for i in range(1, days + 1)])
    predictions = model.predict(future_days)
    
    # Calculate trend
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
    """
    Advanced prediction using multiple features
    
    Args:
        df: DataFrame with stock data (must have technical indicators)
        days: Number of days to predict
    
    Returns:
        Dictionary with predictions and confidence
    """
    # Prepare features
    features = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'MACD']
    
    # Check if features exist
    available_features = [f for f in features if f in df.columns]
    
    if len(available_features) < 3:
        # Fall back to simple prediction
        return linear_regression_prediction(df, days)
    
    # Drop NaN values
    train_data = df[available_features].dropna().tail(90)
    
    if len(train_data) < 30:
        return linear_regression_prediction(df, days)
    
    # Prepare X and y
    X = train_data.drop('Close', axis=1).values
    y = train_data['Close'].values
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # For future prediction, use last known values
    last_features = train_data.drop('Close', axis=1).iloc[-1].values.reshape(1, -1)
    
    predictions = []
    current_features = last_features.copy()
    
    for i in range(days):
        pred = model.predict(current_features)[0]
        predictions.append(pred)
        
        # Update features (simplified - in reality would recalculate indicators)
        current_features[0][0] = pred  # Update volume (simplified)
    
    # Calculate confidence based on recent volatility
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
    
    # Calculate recent high and low
    recent_high = df['High'].tail(30).max()
    recent_low = df['Low'].tail(30).min()
    
    # Calculate volatility
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