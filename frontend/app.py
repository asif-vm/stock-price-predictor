# frontend/app.py
import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

# Configuration
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
        color: white;
    }
    .buy-signal {
        background-color: #1a3f1a;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        color: white;
    }
    .sell-signal {
        background-color: #3f1a1a;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)


# Title
st.title("📈 Stock Price Predictor & Analysis")
st.markdown("### Real-time stock analysis and ML-powered price predictions")

# Check API connection
@st.cache_data(ttl=60)
def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

if not check_api_health():
    st.error("❌ Cannot connect to API. Make sure FastAPI is running on port 8000")
    st.info("Run: `cd backend && uvicorn main:app --reload --port 8000`")
    st.stop()
else:
    st.success("✅ Connected to API")

# Fetch available stocks
@st.cache_data(ttl=3600)
def get_available_stocks():
    try:
        response = requests.get(f"{API_URL}/stocks")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

stocks_data = get_available_stocks()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Stock selection
    if stocks_data:
        market = st.selectbox(
            "Select Market",
            ["Indian Stocks", "US Stocks"]
        )
        
        if market == "Indian Stocks":
            stock_dict = stocks_data['indian_stocks']
        else:
            stock_dict = stocks_data['us_stocks']
        
        # Create display format: "RELIANCE.NS - Reliance Industries"
        stock_options = [f"{ticker} - {name}" for ticker, name in stock_dict.items()]
        
        selected_stock = st.selectbox(
            "Select Stock",
            stock_options
        )
        
        # Extract ticker from selection
        ticker = selected_stock.split(" - ")[0]
    else:
        ticker = st.text_input("Enter Stock Ticker", "RELIANCE.NS")
    
    # Time period
    period = st.selectbox(
        "Time Period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3
    )
    
    # Prediction settings
    st.markdown("---")
    st.subheader("🔮 Prediction Settings")
    
    pred_days = st.slider(
        "Days to Predict",
        min_value=1,
        max_value=30,
        value=7
    )
    
    pred_method = st.radio(
        "Prediction Method",
        ["Advanced (Multi-Feature)", "Simple (Linear)"],
        help="Advanced uses technical indicators for better accuracy"
    )
    
    method = "advanced" if "Advanced" in pred_method else "simple"
    
    # Refresh button
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔮 Predictions", "📈 Technical Analysis", "⚡ Trading Signals"])

# TAB 1: Overview
with tab1:
    st.subheader(f"Stock Overview: {ticker}")
    
    # Fetch stock data
    with st.spinner("Fetching stock data..."):
        try:
            response = requests.get(f"{API_URL}/stock/{ticker}", params={"period": period})
            
            if response.status_code == 200:
                stock_data = response.json()
                
                # Display stock info
                col1, col2, col3, col4 = st.columns(4)
                
                info = stock_data['info']
                col1.metric("Company", info['name'])
                col2.metric("Sector", info['sector'])
                col3.metric("Current Price", f"₹{stock_data['latest_price']:.2f}")
                col4.metric("Total Records", stock_data['total_records'])
                
                # Convert data to DataFrame
                df = pd.DataFrame(stock_data['data'])
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Create price chart
                fig = go.Figure()
                
                fig.add_trace(go.Candlestick(
                    x=df['Date'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name="Price"
                ))
                
                fig.update_layout(
                    title=f"{ticker} - Price Chart ({period})",
                    yaxis_title="Price (₹)",
                    xaxis_title="Date",
                    height=500,
                    template="plotly_dark"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Volume chart
                fig_volume = px.bar(
                    df,
                    x='Date',
                    y='Volume',
                    title=f"{ticker} - Trading Volume"
                )
                fig_volume.update_layout(height=300)
                
                st.plotly_chart(fig_volume, use_container_width=True)
                
                # Additional metrics
                st.markdown("---")
                st.subheader("📊 Additional Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("52 Week High", f"₹{info.get('52_week_high', 0):.2f}")
                col2.metric("52 Week Low", f"₹{info.get('52_week_low', 0):.2f}")
                col3.metric("Market Cap", f"{info.get('market_cap', 0) / 1e9:.2f}B")
                col4.metric("P/E Ratio", f"{info.get('pe_ratio', 0):.2f}")
                
            else:
                st.error(f"Error: {response.json()['detail']}")
        
        except Exception as e:
            st.error(f"Error fetching data: {e}")

# TAB 2: Predictions
with tab2:
    st.subheader(f"🔮 Price Predictions for {ticker}")
    
    if st.button("Generate Predictions", type="primary", use_container_width=True):
        with st.spinner("Generating predictions..."):
            try:
                payload = {
                    "ticker": ticker,
                    "days": pred_days,
                    "method": method
                }
                
                response = requests.post(f"{API_URL}/predict", json=payload)
                
                if response.status_code == 200:
                    pred_data = response.json()
                    
                    # Display prediction summary
                    col1, col2, col3, col4 = st.columns(4)
                    
                    col1.metric("Current Price", f"₹{pred_data['current_price']:.2f}")
                    col2.metric(
                        f"Predicted ({pred_days}d)",
                        f"₹{pred_data['predicted_price']:.2f}",
                        delta=f"{pred_data['change_percentage']:.2f}%"
                    )
                    col3.metric("Trend", pred_data['trend'])
                    col4.metric("Confidence", f"{pred_data['confidence']:.1f}%")
                    
                    # Create prediction chart
                    pred_df = pd.DataFrame({
                        'Date': pd.to_datetime(pred_data['prediction_dates']),
                        'Predicted Price': pred_data['predictions']
                    })
                    
                    fig = go.Figure()
                    
                    # Add current price as starting point
                    fig.add_trace(go.Scatter(
                        x=[datetime.now()],
                        y=[pred_data['current_price']],
                        mode='markers',
                        name='Current Price',
                        marker=dict(size=12, color='blue')
                    ))
                    
                    # Add predictions
                    fig.add_trace(go.Scatter(
                        x=pred_df['Date'],
                        y=pred_df['Predicted Price'],
                        mode='lines+markers',
                        name='Predicted Price',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title=f"Price Prediction for {ticker} (Next {pred_days} Days)",
                        yaxis_title="Price (₹)",
                        xaxis_title="Date",
                        height=500,
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Prediction table
                    st.markdown("### 📅 Daily Predictions")
                    pred_df['Predicted Price'] = pred_df['Predicted Price'].apply(lambda x: f"₹{x:.2f}")
                    st.dataframe(pred_df, use_container_width=True)
                    
                    # Investment recommendation
                    st.markdown("---")
                    st.markdown("### 💡 Investment Insight")
                    
                    if pred_data['trend'] == "Upward":
                        st.success(f"📈 **Bullish Signal**: Price expected to increase by {pred_data['change_percentage']:.2f}% in {pred_days} days")
                    else:
                        st.warning(f"📉 **Bearish Signal**: Price expected to decrease by {abs(pred_data['change_percentage']):.2f}% in {pred_days} days")
                
                else:
                    st.error(f"Prediction failed: {response.json()['detail']}")
            
            except Exception as e:
                st.error(f"Error: {e}")

# TAB 3: Technical Analysis
with tab3:
    st.subheader(f"📈 Technical Analysis for {ticker}")
    
    with st.spinner("Analyzing..."):
        try:
            response = requests.get(f"{API_URL}/analysis/{ticker}")
            
            if response.status_code == 200:
                analysis = response.json()
                
                # Current price and indicators
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Price", f"₹{analysis['current_price']:.2f}")
                col2.metric("Volume", f"{analysis['volume']:,.0f}")
                col3.metric("Date", analysis['date'])
                
                st.markdown("---")
                
                # Technical Indicators
                st.markdown("### 📊 Technical Indicators")
                
                indicators = analysis['indicators']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("**Moving Averages**")
                    if indicators['SMA_20']:
                        st.metric("SMA (20)", f"₹{indicators['SMA_20']:.2f}")
                    if indicators['SMA_50']:
                        st.metric("SMA (50)", f"₹{indicators['SMA_50']:.2f}")
                    if indicators['SMA_200']:
                        st.metric("SMA (200)", f"₹{indicators['SMA_200']:.2f}")
                
                with col2:
                    st.markdown("**Momentum**")
                    if indicators['RSI']:
                        rsi_value = indicators['RSI']
                        st.metric("RSI", f"{rsi_value:.2f}")
                        if rsi_value < 30:
                            st.success("Oversold ✅")
                        elif rsi_value > 70:
                            st.error("Overbought ⚠️")
                        else:
                            st.info("Neutral")
                
                with col3:
                    st.markdown("**MACD**")
                    if indicators['MACD']:
                        st.metric("MACD", f"{indicators['MACD']:.2f}")
                    if indicators['Signal_Line']:
                        st.metric("Signal", f"{indicators['Signal_Line']:.2f}")
                
                with col4:
                    st.markdown("**Bollinger Bands**")
                    if indicators['BB_Upper']:
                        st.metric("Upper", f"₹{indicators['BB_Upper']:.2f}")
                    if indicators['BB_Lower']:
                        st.metric("Lower", f"₹{indicators['BB_Lower']:.2f}")
                
                # Price Targets
                st.markdown("---")
                st.markdown("### 🎯 Price Targets")
                
                targets = analysis['price_targets']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Support Levels**")
                    for i, level in enumerate(targets['support_levels'], 1):
                        st.write(f"Support {i}: ₹{level:.2f}")
                
                with col2:
                    st.markdown("**Resistance Levels**")
                    for i, level in enumerate(targets['resistance_levels'], 1):
                        st.write(f"Resistance {i}: ₹{level:.2f}")
                
                col1, col2 = st.columns(2)
                col1.metric("🛑 Stop Loss", f"₹{targets['stop_loss']:.2f}")
                col2.metric("💰 Take Profit", f"₹{targets['take_profit']:.2f}")
            
            else:
                st.error(f"Analysis failed: {response.json()['detail']}")
        
        except Exception as e:
            st.error(f"Error: {e}")

# TAB 4: Trading Signals
with tab4:
    st.subheader(f"⚡ Trading Signals for {ticker}")
    
    with st.spinner("Generating signals..."):
        try:
            response = requests.get(f"{API_URL}/signals/{ticker}")
            
            if response.status_code == 200:
                signals_data = response.json()
                
                # Overall recommendation
                recommendation = signals_data['recommendation']
                confidence = signals_data['confidence']
                
                if recommendation == "BUY":
                    st.success(f"### 🟢 **{recommendation}** - Confidence: {confidence}%")
                elif recommendation == "SELL":
                    st.error(f"### 🔴 **{recommendation}** - Confidence: {confidence}%")
                else:
                    st.info(f"### 🟡 **{recommendation}** - Confidence: {confidence}%")
                
                # Signal summary
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current Price", f"₹{signals_data['current_price']:.2f}")
                col2.metric("Buy Signals", signals_data['signal_count']['buy'])
                col3.metric("Sell Signals", signals_data['signal_count']['sell'])
                col4.metric("Total Signals", signals_data['signal_count']['total'])
                
                # Individual signals
                st.markdown("---")
                st.markdown("### 📋 Signal Details")
                
                for signal in signals_data['signals']:
                    signal_class = "buy-signal" if signal['type'] == "BUY" else "sell-signal"
                    
                    st.markdown(f"""
                        <div class="{signal_class}">
                            <strong>{signal['type']}</strong> - {signal['indicator']} ({signal['strength']})<br>
                            {signal['reason']}
                        </div>
                    """, unsafe_allow_html=True)
                    st.markdown("")
            
            else:
                st.error(f"Signal generation failed: {response.json()['detail']}")
        
        except Exception as e:
            st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>📈 Stock Price Predictor | Built with FastAPI + Streamlit | Data from Yahoo Finance</p>
        <p>⚠️ Disclaimer: This is for educational purposes only. Not financial advice.</p>
    </div>
""", unsafe_allow_html=True)