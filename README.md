# 📈 AI-Powered Real-Time Stock Portfolio Optimization for NIFTY50

In the face of rapid market fluctuations and complex data environments, this project delivers an intelligent investment platform tailored for the Indian stock market. Leveraging real-time and historical data from Yahoo Finance, enriched with technical indicators such as SMA, EMA, RSI, MACD, and Bollinger Bands, the system uses an XGBoost model for short-term Nifty50 stock prediction. Portfolio optimization is performed using the Markowitz Efficient Frontier, considering user-specific risk tolerance, expected returns, and key ratios like Sharpe and Sortino.

A robust **PostgreSQL database** powers the backend data pipeline, ensuring reliable storage and retrieval of processed stock data, technical indicators, and prediction outputs.

The platform is implemented via a Flask web app with secure login (including Google OAuth), interactive dashboards, and real-time WhatsApp alerts using Twilio.

---

### 🧠 Project Overview
This system is designed to help users:
- Predict short-term NIFTY50 stock movements using ML
- Build optimized portfolios based on risk-return trade-offs
- Receive real-time investment alerts via WhatsApp

---

### 🚀 Features
- 📊 Real-time and historical stock data from Yahoo Finance
- 🗃️ PostgreSQL-based data pipeline for storing:
  - Raw and cleaned market data
  - Technical indicators
  - Predicted and actual stock prices
- 📈 Technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands, Volatility
- 🤖 XGBoost model for real-time stock price prediction
- 💹 Markowitz Efficient Frontier for portfolio optimization
- 🔐 Secure user login with manual and Google Sign-In
- 📲 Real-time WhatsApp alerts using Twilio API
- 📉 Visualization of actual vs. predicted prices and portfolio performance

---

### 📂 Backend File - `nifty50.py`
This script handles:
- Data ingestion (historical & real-time from Yahoo Finance)
- Feature engineering (daily returns, SMA_20, SMA_50, EMA_20, EMA_50, RSI_14, MACD, Bollinger Bands, volatility)
- Portfolio optimization using Markowitz Efficient Frontier
- Prediction modeling using XGBoost
- Data persistence using PostgreSQL
- Visualization of stock performance and predictions

---


