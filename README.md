📈 Stock Price Prediction & Analysis System
A full-stack machine learning application for real-time stock analysis and price prediction using FastAPI backend and Streamlit frontend.
Show Image
Show Image
Show Image
🎯 Features
📊 Real-Time Stock Data

Fetch live stock prices from Yahoo Finance
Support for Indian (NSE/BSE) and US stocks
Historical data visualization
Candlestick charts and volume analysis

🔮 ML-Powered Predictions

Linear regression-based price predictions
Multi-feature advanced predictions using technical indicators
Configurable prediction horizon (1-30 days)
Confidence scores and trend analysis

📈 Technical Analysis

Moving Averages: SMA (20, 50, 200 days)
Momentum Indicators: RSI, MACD
Volatility: Bollinger Bands
Price Targets: Support/Resistance levels
Stop Loss & Take Profit suggestions

⚡ Trading Signals

Automated buy/sell signal generation
Multi-indicator analysis
Signal strength ratings
Overall recommendation with confidence score

🛠️ Tech Stack
Backend:

FastAPI (REST API)
scikit-learn (Machine Learning)
yfinance (Stock Data)
pandas & numpy (Data Processing)

Frontend:

Streamlit (Interactive Dashboard)
Plotly (Visualizations)

🚀 Installation
Prerequisites

Python 3.8 or higher
pip package manager

Setup

Clone the repository

bashgit clone https://github.com/asif-vm/stock-price-predictor.git
cd stock-price-predictor

Create virtual environment

bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies

bashpip install -r requirements.txt
🎮 Usage
Start Backend (FastAPI)
bashcd backend
uvicorn main:app --reload --port 8000
Backend will be available at:

API: http://localhost:8000
Interactive Docs: http://localhost:8000/docs

Start Frontend (Streamlit)
Open a new terminal:
bashcd frontend
streamlit run app.py
Frontend will open automatically at http://localhost:8501

“Live market data uses third-party providers that may rate-limit cloud IPs.
Fallback and caching implemented.”
📖 API Endpoints
Stock Data
GET  /stocks              - Get list of available stocks
GET  /stock/{ticker}      - Get historical stock data
POST /compare             - Compare multiple stocks
Predictions
POST /predict             - Predict future stock prices
Analysis
GET  /analysis/{ticker}   - Get technical analysis
GET  /signals/{ticker}    - Get trading signals
Health
GET  /                    - API info
GET  /health              - Health check
📊 Supported Stocks
Indian Stocks (NSE)

RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS
ICICIBANK.NS, HINDUNILVR.NS, ITC.NS, SBIN.NS
BHARTIARTL.NS, KOTAKBANK.NS, and more...

US Stocks

AAPL, GOOGL, MSFT, AMZN, TSLA, META, NVDA

🎯 How It Works
1. Data Collection

Fetches real-time and historical data from Yahoo Finance
Processes OHLCV (Open, High, Low, Close, Volume) data

2. Feature Engineering

Calculates technical indicators (SMA, RSI, MACD, Bollinger Bands)
Generates momentum and volatility features

3. ML Prediction

Simple Model: Linear regression on historical prices
Advanced Model: Multi-feature regression with technical indicators
Returns predictions with confidence scores

4. Signal Generation

Analyzes multiple technical indicators
Generates buy/sell signals with reasoning
Provides overall recommendation

📝 Example Usage
Via API (Python)
pythonimport requests

# Get stock data
response = requests.get("http://localhost:8000/stock/RELIANCE.NS")
data = response.json()

# Predict prices
response = requests.post(
    "http://localhost:8000/predict",
    json={"ticker": "RELIANCE.NS", "days": 7, "method": "advanced"}
)
prediction = response.json()
print(f"Predicted price: ₹{prediction['predicted_price']}")

# Get trading signals
response = requests.get("http://localhost:8000/signals/RELIANCE.NS")
signals = response.json()
print(f"Recommendation: {signals['recommendation']}")
Via Streamlit Dashboard

Select stock from dropdown
Choose time period and prediction days
View charts and predictions
Analyze technical indicators
Check trading signals

🎨 Screenshots
Add screenshots here after running the app:

Dashboard overview
Price predictions
Technical analysis
Trading signals

📈 Prediction Methods
Simple Method

Uses linear regression on historical close prices
Fast and straightforward
Good for stable stocks

Advanced Method

Incorporates technical indicators
Multi-feature regression
Higher accuracy for volatile stocks
Confidence scoring based on volatility

⚠️ Disclaimer
This application is for educational and research purposes only.

Not financial advice
Past performance doesn't guarantee future results
Always do your own research
Consult financial advisors before investing

🔮 Future Enhancements

 LSTM/GRU models for better predictions
 Sentiment analysis from news
 Portfolio optimization
 Real-time alerts and notifications
 Cryptocurrency support
 Backtesting framework
 User authentication and portfolios
 Database integration for historical predictions

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
📄 License
This project is licensed under the MIT License.
👤 Author
Asif V M

GitHub: @asif-vm
LinkedIn: asif-v-m
Email: asifvm15@gmail.com

🙏 Acknowledgments

Yahoo Finance for stock data (via yfinance)
FastAPI and Streamlit communities
scikit-learn for ML algorithms


⭐ If you find this project useful, please consider giving it a star!
📚 Resources

FastAPI Documentation
Streamlit Documentation
yfinance Documentation
Technical Indicators Guide

