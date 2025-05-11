import os
import time
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, render_template, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_login import login_required

from dotenv import load_dotenv
from config import SQLALCHEMY_DATABASE_URI
from models import db, User
from auth import auth, init_oauth
from flask_apscheduler import APScheduler  
from sqlalchemy.orm import sessionmaker
from sqlalchemy import asc, desc, text  
from twilio.rest import Client

# Load environment variables
load_dotenv()

app = Flask(__name__)

# 🔹 App Configurations
app.config["SQLALCHEMY_DATABASE_URI"] = SQLALCHEMY_DATABASE_URI
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
app.config["GOOGLE_CLIENT_ID"] = os.getenv("GOOGLE_CLIENT_ID")
app.config["GOOGLE_CLIENT_SECRET"] = os.getenv("GOOGLE_CLIENT_SECRET")

# Twilio Config
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# 🔹 Initialize OAuth & Database
init_oauth(app)
db.init_app(app)

# 🔹 Initialize Login Manager
login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# 🔹 Register Blueprints
app.register_blueprint(auth, url_prefix="/auth")


# 🔹 Define ORM Model for Optimized Portfolio
class OptimizedPortfolio(db.Model):
    __tablename__ = "optimized_portfolio"

    symbol = db.Column(db.String, primary_key=True)
    allocation = db.Column(db.Float)
    return_pct = db.Column("return", db.Float)  # 'return' is a reserved keyword, so use alias
    volatility = db.Column(db.Float)
    sharpe_ratio = db.Column(db.Float)


# 🔹 Route to render frontend
@app.route("/optimized_portfolio")
def optimized_portfolio():
    return render_template("optimized_portfolio.html")


@app.route("/api/optimized_portfolio", methods=["GET"])
def get_optimized_portfolio():
    try:
        investment_amount = float(request.args.get("amount", 0))
        risk_level = request.args.get("risk_level", "medium").lower()
        whatsapp_number = request.args.get("whatsapp_number", "").strip()
        whatsapp_enabled = request.args.get("whatsapp_enabled", "false").lower() == "true"

        if investment_amount <= 0:
            return jsonify({"error": "Invalid investment amount"}), 400

        # Fetch portfolio allocations from the database
        portfolio = OptimizedPortfolio.query.all()
        portfolio_sorted = sorted(portfolio, key=lambda x: x.allocation, reverse=True)

        # Define risk levels
        risk_factors = {"low": 0.8, "medium": 1.0, "high": 1.2}
        allocation_factor = risk_factors.get(risk_level, 1.0)

        # Process portfolio data
        total_allocation = sum(stock.allocation for stock in portfolio)
        portfolio_data = []

        for i, stock in enumerate(portfolio_sorted):
            allocation_pct = (stock.allocation / total_allocation) * 100
            investment_allocation = (stock.allocation * investment_amount) * allocation_factor

            portfolio_data.append(
                {
                    "symbol": stock.symbol,
                    "allocation_pct": round(allocation_pct, 2),
                    "investment_allocation": round(investment_allocation, 2),
                    "highlight": i < 10,  # Highlight top 10
                }
            )

        # Prepare data for visualization
        pie_chart_data = {
            "labels": [stock["symbol"] for stock in portfolio_data[:10]] + ["Other Stocks"],
            "values": [stock["investment_allocation"] for stock in portfolio_data[:10]]
            + [sum(stock["investment_allocation"] for stock in portfolio_data[10:])],
        }

        # ✅ WhatsApp Message Formatting
        # ✅ If WhatsApp is enabled, send a message
        if whatsapp_enabled and whatsapp_number:
            send_whatsapp_notification(portfolio_data, whatsapp_number)

        return jsonify({"portfolio": portfolio_data, "chart_data": pie_chart_data})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def send_whatsapp_notification(portfolio_data, whatsapp_number):
    try:
        # Ensure the number starts with 'whatsapp:'
        if not whatsapp_number.startswith("whatsapp:"):
            whatsapp_number = f"whatsapp:{whatsapp_number}"

        # Prepare message body
        message_body = f"📊 Your Optimized Portfolio:\n"
        for stock in portfolio_data:
            message_body += f"{stock['symbol']}: {stock['allocation_pct']}% (₹{stock['investment_allocation']:.2f})\n"

        # Send WhatsApp message
        message = client.messages.create(
            body=message_body,
            from_=TWILIO_WHATSAPP_NUMBER,
            to=whatsapp_number
        )

        print(f"WhatsApp message sent successfully. SID: {message.sid}")

    except Exception as e:
        print(f"Error sending WhatsApp notification: {e}")

# 🔹 Define StockPrediction Model
class StockPrediction(db.Model):
    __tablename__ = 'stock_predictions'
    
    symbol = db.Column(db.String, primary_key=True)
    date = db.Column(db.DateTime, primary_key=True)
    close = db.Column(db.Float)
    predicted_close = db.Column(db.Float)

# 🔹 Route to Display Stock Predictions Page
@app.route("/stock_predictions")
def stock_predictions():
    return render_template("stock_predictions.html")


# API to get top 5 best-performing predicted stocks
@app.route("/api/stock_predictions", methods=["GET"])
def get_stock_predictions():
    try:
        symbol = request.args.get("symbol", "").upper()

        if not symbol:  # If no stock symbol is entered, show default stocks
            default_symbols = ["BEL.NS", "GRASIM.NS", "EICHERMOT.NS", "JSWSTEEL.NS", "BPCL.NS", "ULTRACEMCO.NS", "APOLLOHOSP.NS", "TRENT.NS",
    "WIPRO.NS", "BHARTIARTL.NS", "HDFCLIFE.NS", "INFY.NS", "BAJFINANCE.NS", "ADANIENT.NS", "NTPC.NS", "LT.NS", "SBIN.NS",
    "M&M.NS", "TATASTEEL.NS", "SHRIRAMFIN.NS", "POWERGRID.NS", "TCS.NS", "KOTAKBANK.NS", "HINDALCO.NS", "ITC.NS",
    "DRREDDY.NS", "ICICIBANK.NS", "TECHM.NS", "TITAN.NS", "TATAMOTORS.NS", "BRITANNIA.NS", "ONGC.NS", "ASIANPAINT.NS",
    "CIPLA.NS", "HCLTECH.NS", "HINDUNILVR.NS", "INDUSINDBK.NS", "SUNPHARMA.NS", "AXISBANK.NS", "TATACONSUM.NS",
    "NESTLEIND.NS", "MARUTI.NS", "SBILIFE.NS", "HEROMOTOCO.NS", "ADANIPORTS.NS", "HDFCBANK.NS", "BAJAJFINSV.NS",
    "RELIANCE.NS", "COALINDIA.NS", "BAJAJ-AUTO.NS"]
            predictions = (
                db.session.query(StockPrediction)
                .filter(StockPrediction.symbol.in_(default_symbols))
                .order_by(desc(StockPrediction.date))
                .limit(50)
                .all()
            )
        else:
            predictions = (
                db.session.query(StockPrediction)
                .filter(StockPrediction.symbol == symbol)
                .order_by(desc(StockPrediction.date))
                .limit(50)
                .all()
            )

        predictions_list = [
            {
                "symbol": p.symbol,
                "date": p.date.strftime("%Y-%m-%d"),
                "close": p.close,
                "predicted_close": p.predicted_close,
            }
            for p in predictions
        ]

        return jsonify(predictions_list)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API to get top 5 best-performing predicted stocks
@app.route("/api/top_predicted_stocks", methods=["GET"])
def get_top_predicted_stocks():
    try:
        top_stocks = db.session.query(
            StockPrediction.symbol, StockPrediction.date,
            StockPrediction.close, StockPrediction.predicted_close
        ).order_by((StockPrediction.predicted_close - StockPrediction.close).desc()).limit(5).all()

        top_stocks_list = [
            {
                "symbol": s.symbol,
                "date": s.date.strftime("%Y-%m-%d"),
                "close": s.close,
                "predicted_close": s.predicted_close,
                "growth": round(s.predicted_close - s.close, 2)  # Growth in price
            }
            for s in top_stocks
        ]

        return jsonify(top_stocks_list)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 🔹 Homepage Route
#@app.route("/")
#def home():
    #return "Welcome to the Stock Portfolio Web App!"


# 🔹 Dashboard Route
from flask_login import login_required

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")

@app.route("/")
def index():
    return render_template("index.html")



# 🔹 Route to Get Latest Stock Data
@app.route("/stocks")
def get_stocks():
    with db.engine.connect() as connection:
        result = connection.execute("SELECT * FROM real_time_data ")
        stocks = [dict(row) for row in result]
        return jsonify(stocks)


@app.route("/real_time_data")
def display_real_time_data():
    try:
        Session = sessionmaker(bind=db.engine)
        session = Session()
        query = text("SELECT * FROM real_time_data ")
        result = session.execute(query).fetchall()
        session.close()

        real_time_data = [dict(row._asdict()) for row in result]

        return render_template("real_time_data.html", data=real_time_data)

    except Exception as e:
        return jsonify({"error": str(e)})


# 🔹 Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
