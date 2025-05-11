import time
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine

# Database connection string (Update with your actual credentials)
DATABASE_URL = "postgresql://postgres:1234@localhost:5432/nifty50_data"

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# List of NIFTY 50 Stocks
nifty50_stocks = ["BEL.NS", "GRASIM.NS", "EICHERMOT.NS", "JSWSTEEL.NS", "BPCL.NS", "ULTRACEMCO.NS", "APOLLOHOSP.NS", "TRENT.NS",
    "WIPRO.NS", "BHARTIARTL.NS", "HDFCLIFE.NS", "INFY.NS", "BAJFINANCE.NS", "ADANIENT.NS", "NTPC.NS", "LT.NS", "SBIN.NS",
    "M&M.NS", "TATASTEEL.NS", "SHRIRAMFIN.NS", "POWERGRID.NS", "TCS.NS", "KOTAKBANK.NS", "HINDALCO.NS", "ITC.NS",
    "DRREDDY.NS", "ICICIBANK.NS", "TECHM.NS", "TITAN.NS", "TATAMOTORS.NS", "BRITANNIA.NS", "ONGC.NS", "ASIANPAINT.NS",
    "CIPLA.NS", "HCLTECH.NS", "HINDUNILVR.NS", "INDUSINDBK.NS", "SUNPHARMA.NS", "AXISBANK.NS", "TATACONSUM.NS",
    "NESTLEIND.NS", "MARUTI.NS", "SBILIFE.NS", "HEROMOTOCO.NS", "ADANIPORTS.NS", "HDFCBANK.NS", "BAJAJFINSV.NS",
    "RELIANCE.NS", "COALINDIA.NS", "BAJAJ-AUTO.NS"]

def fetch_real_time_data():
    real_time_data = pd.DataFrame()

    for stock in nifty50_stocks:
        try:
            # Fetch data for each stock
            data = yf.download(stock, period="1d", interval="1m", multi_level_index=False,auto_adjust=True)
            
            # ✅ Reset index to remove MultiIndex issue
            data = data.reset_index()
            
            # ✅ Add stock symbol column
            data['ticker'] = stock  

            # ✅ Rename columns to match PostgreSQL table schema
            data.rename(columns={
                'Datetime': 'date', 
                'Open': 'open', 
                'High': 'high', 
                'Low': 'low', 
                'Close': 'close', 
                'Volume': 'volume', 
                'ticker': 'symbol'
            }, inplace=True)
            
            # ✅ Convert 'date' column to proper datetime format
            data['date'] = pd.to_datetime(data['date'])

            # ✅ Drop unnamed or unnecessary columns
            data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

            # ✅ Append to the final dataframe
            real_time_data = pd.concat([real_time_data, data])
            print(f"✅ Successfully fetched data for {stock}")
        except Exception as e:
            print(f"⚠️ Could not fetch data for {stock}: {e}")

    # ✅ Ensure DataFrame is not empty before saving
    if not real_time_data.empty:
        try:
            # Save to PostgreSQL using engine
            real_time_data.to_sql('real_time_data', con=engine, if_exists='append', index=False)
            print("✅ Successfully saved data to PostgreSQL")
        except Exception as e:
            print(f"⚠️ Could not save data to PostgreSQL: {e}")
    else:
        print("⚠️ No data fetched. Nothing to save.")

# ✅ Run every 60 seconds
if __name__ == "__main__":
    while True:
        fetch_real_time_data()
        time.sleep(60)  # Update every 60 seconds
