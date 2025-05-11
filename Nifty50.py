#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip uninstall yfinance
#!pip uninstall pandas-datareader
#!pip install yfinance --upgrade --no-cache-dir
#!pip install pandas-datareader
#!pip install matplotlib


# In[2]:


import yfinance as yf  # For fetching historical stock data
import pandas as pd
import numpy as np
import datetime
from alpha_vantage.timeseries import TimeSeries  
import matplotlib.pyplot as plt


# In[3]:


nifty50_stocks = [
    "BEL.NS", "GRASIM.NS", "EICHERMOT.NS", "JSWSTEEL.NS", "BPCL.NS", "ULTRACEMCO.NS", "APOLLOHOSP.NS", "TRENT.NS",
    "WIPRO.NS", "BHARTIARTL.NS", "HDFCLIFE.NS", "INFY.NS", "BAJFINANCE.NS", "ADANIENT.NS", "NTPC.NS", "LT.NS", "SBIN.NS",
    "M&M.NS", "TATASTEEL.NS", "SHRIRAMFIN.NS", "POWERGRID.NS", "TCS.NS", "KOTAKBANK.NS", "HINDALCO.NS", "ITC.NS",
    "DRREDDY.NS", "ICICIBANK.NS", "TECHM.NS", "TITAN.NS", "TATAMOTORS.NS", "BRITANNIA.NS", "ONGC.NS", "ASIANPAINT.NS",
    "CIPLA.NS", "HCLTECH.NS", "HINDUNILVR.NS", "INDUSINDBK.NS", "SUNPHARMA.NS", "AXISBANK.NS", "TATACONSUM.NS",
    "NESTLEIND.NS", "MARUTI.NS", "SBILIFE.NS", "HEROMOTOCO.NS", "ADANIPORTS.NS", "HDFCBANK.NS", "BAJAJFINSV.NS",
    "RELIANCE.NS", "COALINDIA.NS", "BAJAJ-AUTO.NS"
]


# In[4]:


# Create an empty DataFrame to store data
real_time_data = pd.DataFrame()

for stock in nifty50_stocks:
    try:
        data = yf.download(stock,multi_level_index=False, period="1d", interval="1m")
        data['Symbol'] = stock
        data['date'] = data.index
        data.reset_index(drop=True, inplace=True)

        # Ensure 'date' column is in datetime format
        data['date'] = pd.to_datetime(data['date'])

        # Remove any 'Unnamed' columns if present
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

        real_time_data = pd.concat([real_time_data, data])
        print(f"Successfully fetched data for {stock}")
    except Exception as e:
        print(f"Could not fetch data for {stock}: {e}")

# Save the data to a CSV file
real_time_data.to_csv("nifty50_real_time_data.csv", index=False)
print("Data collection completed and saved to 'nifty50_real_time_data.csv'")


# In[5]:


real_time_data.head()


# In[6]:


real_time_data.tail()


# In[7]:


# Fetch historical data for the past 5 years
historical_data = pd.DataFrame()

for stock in nifty50_stocks:
    try:
        data = yf.download(stock, multi_level_index=False,start="2018-01-01", end="2023-12-31")
        data['Symbol'] = stock
        historical_data = pd.concat([historical_data, data])
        print(f"Historical data fetched for {stock}")
    except Exception as e:
        print(f"Failed to fetch historical data for {stock}: {e}")

# Save historical data
historical_data.to_csv("nifty50_historical_data.csv")
print("Historical data saved to 'nifty50_historical_data.csv'")


# In[8]:


historical_data.head()


# In[9]:


# Load the data
real_time_data = pd.read_csv("nifty50_real_time_data.csv")
historical_data = pd.read_csv("nifty50_historical_data.csv")


# In[10]:


# Check for missing values
print("Real-time data missing values:\n", real_time_data.isnull().sum())
print("Historical data missing values:\n", historical_data.isnull().sum())


# In[11]:


print(real_time_data.dtypes)


# In[12]:


print(historical_data.dtypes)


# In[13]:


print("Historical Data Columns:", historical_data.columns.tolist())


# In[14]:


print("Real-Time Data Columns:", real_time_data.columns.tolist())


# In[15]:


historical_data.rename(columns=lambda x: x.strip().lower(), inplace=True)


# In[16]:


print("Historical Data Columns:", historical_data.columns.tolist())


# In[17]:


real_time_data.rename(columns=lambda x: x.strip().lower(), inplace=True)


# In[18]:


print("Real Time Data Columns:", real_time_data.columns.tolist())


# In[19]:


real_time_data.rename(columns={"adj close": "adj_close"}, inplace=True)


# In[20]:


# ✅ Ensure both datasets have the same column order
column_order = ['date', 'close', 'high', 'low', 'open', 'volume', 'symbol']

# Reorder historical data
historical_data = historical_data[column_order]

# Reorder real-time data
real_time_data = real_time_data[column_order]


# In[21]:


print(real_time_data.columns)


# In[22]:


print(historical_data.columns)


# In[23]:


historical_data['date'] = pd.to_datetime(historical_data['date'])


# In[24]:


real_time_data['date'] = pd.to_datetime(real_time_data['date'], utc=True)
real_time_data['date'] = real_time_data['date'].dt.tz_localize(None)  # Remove timezone


# In[25]:


historical_data['date']


# In[26]:


real_time_data['date'] 


# In[27]:


import psycopg2
import pandas as pd

try:
    # ✅ Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname="nifty50_data",
        user="postgres",
        password="1234",
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()
    print("✅ Connected to PostgreSQL database.")

    # ✅ Convert date column to correct format
    real_time_data["date"] = pd.to_datetime(real_time_data["date"])  
    real_time_data["date"] = real_time_data["date"].dt.tz_localize(None)  # Remove timezone

    # ✅ Debug: Check data before inserting
    print("🔍 Checking Real-Time Data Before Insertion:")
    print(real_time_data.head())  # Show first 5 rows
    print(f"Total Rows: {real_time_data.shape[0]}")  # Check row count

    # ✅ Insert Real-Time Data in Batches
    batch_size = 50
    for i, (_, row) in enumerate(real_time_data.iterrows()):
        #print(f"Inserting: {row['date']}, {row['open']}, {row['symbol']}")  # Debug Output

        cursor.execute("""
            INSERT INTO real_time_data ("date", "close", "high", "low", "open","volume", "symbol")
            VALUES (%s, %s, %s, %s, %s, %s, %s);
        """, (row["date"], row["close"], row["high"], row["low"], row["open"], row["volume"], row["symbol"]))

        if (i + 1) % batch_size == 0:  # Commit every 50 rows
            conn.commit()

    conn.commit()  # Final commit
    print("✅ Data inserted successfully.")

except Exception as e:
    print(f"❌ Error: {e}")

finally:
    if conn:
        cursor.close()
        conn.close()
        print("🔒 PostgreSQL connection closed.")


# In[28]:


print(historical_data.describe())


# In[29]:


print(real_time_data.describe())


# In[30]:


# Check for missing values in both real-time and historical datasets
print(real_time_data.isnull().sum())


# In[31]:


print(historical_data.isnull().sum())


# In[32]:


print("Missing values in Historical Data:\n", historical_data.isnull().sum())
print("\nMissing values in Real-Time Data:\n", real_time_data.isnull().sum())


# In[33]:


historical_data.drop_duplicates(inplace=True)
real_time_data.drop_duplicates(inplace=True)


# In[34]:


historical_data.duplicated()


# In[35]:


real_time_data.duplicated()


# In[36]:


print("Historical Data Types:\n", historical_data.dtypes)
print("Real-Time Data Types:\n", real_time_data.dtypes)


# In[37]:


print(historical_data.columns.tolist())
print(real_time_data.columns.tolist())


# In[38]:


# ✅ Convert 'date' column in historical data to datetime
historical_data['date'] = pd.to_datetime(historical_data['date'])

# ✅ Verify the change
print(historical_data.dtypes)


# In[39]:


print(real_time_data.dtypes)


# In[40]:


from sqlalchemy import create_engine

# ✅ Create PostgreSQL Connection
engine = create_engine("postgresql://postgres:1234@localhost:5432/nifty50_data")

# ✅ Store Cleaned Historical Data
historical_data.to_sql('historical_data', engine, if_exists='replace', index=False)
print("✅ Cleaned Historical Data Stored Successfully.")

# ✅ Store Cleaned Real-Time Data
real_time_data.to_sql('real_time_data', engine, if_exists='replace', index=False)
print("✅ Cleaned Real-Time Data Stored Successfully.")



# In[41]:


#Feature Engineering


# In[42]:


#1.Daily Returns


# In[43]:


# Sort by date and symbol
historical_data = historical_data.sort_values(by=['symbol', 'date'])
real_time_data = real_time_data.sort_values(by=['symbol', 'date'])


# In[44]:


# ✅ Calculate Daily Returns
historical_data['daily_return'] = historical_data.groupby('symbol')['close'].pct_change()
real_time_data['daily_return'] = real_time_data.groupby('symbol')['close'].pct_change()


# In[45]:


print(historical_data[['symbol', 'date', 'close', 'daily_return']].head(30))


# In[46]:


print(real_time_data[['symbol', 'date', 'close', 'daily_return']].head(10))


# In[47]:


#Moving averages and volatility


# In[48]:


# ✅ Calculate Moving Averages and Volatility for Historical Data
historical_data['SMA_20'] = historical_data.groupby('symbol')['close'].transform(lambda x: x.rolling(window=20).mean())
historical_data['SMA_50'] = historical_data.groupby('symbol')['close'].transform(lambda x: x.rolling(window=50).mean())
historical_data['volatility_30'] = historical_data.groupby('symbol')['daily_return'].transform(lambda x: x.rolling(window=30).std())


# In[49]:


# ✅ Calculate Moving Averages and Volatility for Real-Time Data
real_time_data['SMA_20'] = real_time_data.groupby('symbol')['close'].transform(lambda x: x.rolling(window=20).mean())
real_time_data['SMA_50'] = real_time_data.groupby('symbol')['close'].transform(lambda x: x.rolling(window=50).mean())
real_time_data['volatility_30'] = real_time_data.groupby('symbol')['daily_return'].transform(lambda x: x.rolling(window=30).std())


# In[50]:


# ✅ Check the first few rows to verify calculations
print("🔍 Checking Feature Engineering Results:")
print(historical_data[['symbol', 'date', 'close', 'SMA_20', 'SMA_50', 'volatility_30']].head(30))


# In[51]:


# ✅ Check the first few rows to verify calculations
print("🔍 Checking Feature Engineering Results:")
print(real_time_data[['symbol', 'date', 'close', 'SMA_20', 'SMA_50', 'volatility_30']].head(30))


# In[52]:


# Check after enough data points are available
print(historical_data[['symbol', 'date', 'close', 'SMA_20', 'SMA_50', 'volatility_30']].dropna().head(30))


# In[53]:


# Check after enough data points are available
print(real_time_data[['symbol', 'date', 'close', 'SMA_20', 'SMA_50', 'volatility_30']].dropna().head(50))


# In[54]:


import pandas as pd


# In[55]:


# Function to calculate advanced technical indicators
def add_technical_indicators(df):
    # ✅ Exponential Moving Averages (EMA)
    df['EMA_20'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    df['EMA_50'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=50, adjust=False).mean())

    # ✅ Relative Strength Index (RSI)
    delta = df.groupby('symbol')['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # ✅ Moving Average Convergence Divergence (MACD)
    df['MACD'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean()) - \
                 df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())

    # ✅ Bollinger Bands
    rolling_mean = df.groupby('symbol')['close'].rolling(window=20).mean().reset_index(level=0, drop=True)
    rolling_std = df.groupby('symbol')['close'].rolling(window=20).std().reset_index(level=0, drop=True)
    df['Bollinger_Upper'] = rolling_mean + (rolling_std * 2)
    df['Bollinger_Lower'] = rolling_mean - (rolling_std * 2)

    return df


# In[56]:


# ✅ Apply feature engineering to historical and real-time data
historical_data = add_technical_indicators(historical_data)
real_time_data = add_technical_indicators(real_time_data)


# In[57]:


# ✅ Check feature calculation results
print("🔍 Checking Feature Engineering Results:")
print(historical_data[['symbol', 'date', 'close', 'EMA_20', 'RSI_14', 'MACD', 'Bollinger_Upper']].head(30))


# In[58]:


# ✅ Check feature calculation results
print("🔍 Checking Feature Engineering Results:")
print(real_time_data[['symbol', 'date', 'close', 'EMA_20', 'RSI_14', 'MACD', 'Bollinger_Upper']].head(30))


# In[59]:


print(historical_data.describe())


# In[60]:


print(real_time_data.describe())


# In[61]:


import numpy as np

# ✅ Replace Inf values in daily_return
historical_data.replace([np.inf, -np.inf], np.nan, inplace=True)
historical_data.dropna(subset=['daily_return'], inplace=True)


# In[62]:


real_time_data.replace([np.inf, -np.inf], np.nan, inplace=True)
real_time_data.dropna(subset=['daily_return'], inplace=True)


# In[63]:


# ✅ Remove rows where `close == 0`
historical_data = historical_data[historical_data['close'] > 0]
real_time_data = real_time_data[real_time_data['close'] > 0]


# In[64]:


# ✅ Fill missing `volume` values without warning
historical_data['volume'] = historical_data['volume'].replace(0, np.nan).ffill()
real_time_data['volume'] = real_time_data['volume'].replace(0, np.nan).ffill()


# In[65]:


# ✅ Cap `volatility_30` to 99th percentile
historical_data['volatility_30'] = np.where(
    historical_data['volatility_30'] > historical_data['volatility_30'].quantile(0.99),
    historical_data['volatility_30'].median(),
    historical_data['volatility_30']
)


# In[66]:


# ✅ Clip `RSI_14` values between 1 and 9
historical_data['RSI_14'] = np.clip(historical_data['RSI_14'], 1, 99)
real_time_data['RSI_14'] = np.clip(real_time_data['RSI_14'], 1, 99)


# In[67]:


historical_data.columns


# In[68]:


from sqlalchemy import create_engine

# ✅ Create PostgreSQL Connection
engine = create_engine("postgresql://postgres:1234@localhost:5432/nifty50_data")

# ✅ Store Cleaned Historical Data
historical_data.to_sql('historical_data', engine, if_exists='replace', index=False)
print("✅ Cleaned Historical Data Stored Successfully.")

# ✅ Store Cleaned Real-Time Data
real_time_data.to_sql('real_time_data', engine, if_exists='replace', index=False)
print("✅ Cleaned Real-Time Data Stored Successfully.")



# In[69]:


import psycopg2
import pandas as pd

try:
    # ✅ Reconnect before fetching data
    conn = psycopg2.connect(
        dbname="nifty50_data",
        user="postgres",
        password="1234",
        host="localhost",
        port="5432"
    )
    print("✅ Connected to PostgreSQL database.")

    # ✅ Fetch Data from PostgreSQL
    stock_data = pd.read_sql("SELECT date, symbol, daily_return FROM historical_data", con=conn)
    nifty_data = stock_data.copy()  # Since all Nifty 50 stocks are included

    print("✅ Data fetched successfully.")

except Exception as e:
    print(f"❌ Error: {e}")

finally:
    if conn:
        conn.close()
        print("🔒 PostgreSQL connection closed.")


# In[70]:


#from historical_data


# In[71]:


import numpy as np
import pandas as pd

# ✅ Extract Nifty 50 Index Daily Returns
nifty_returns = nifty_data.groupby('date')['daily_return'].mean()  # Average return of all stocks

# ✅ Calculate Beta for Each Stock
betas = {}

for symbol in stock_data['symbol'].unique():
    stock_returns = stock_data[stock_data['symbol'] == symbol].set_index('date')['daily_return']
    
    # ✅ Ensure both stock and index returns have the same dates
    combined_df = pd.merge(stock_returns, nifty_returns, left_index=True, right_index=True, how='inner')
    
    if len(combined_df) > 1:  # Ensure we have enough data points
        market_var = np.var(combined_df.iloc[:, 1], ddof=1)  # Variance of Nifty 50 returns (column 1)
        market_cov = np.cov(combined_df.iloc[:, 0], combined_df.iloc[:, 1])[0, 1]  # Covariance with market
        beta = market_cov / market_var if market_var != 0 else 0
    else:
        beta = np.nan  # Not enough data
    
    betas[symbol] = beta

# Convert to DataFrame
beta_df = pd.DataFrame.from_dict(betas, orient='index', columns=['Beta']).dropna()
print("✅ Beta values computed:\n", beta_df)


# In[72]:


# ✅ Define Parameters
risk_free_rate = 0.07  # 7% (Adjust based on India’s bond yield)
market_return = nifty_data['daily_return'].mean()  # Expected return of Nifty 50

# ✅ Compute Expected Returns for Each Stock
beta_df['Expected Return'] = risk_free_rate + beta_df['Beta'] * (market_return - risk_free_rate)
print("✅ Expected Returns computed:\n", beta_df)


# In[73]:


# ✅ Compute Standard Deviation (Risk) for Each Stock
stock_volatility = stock_data.groupby('symbol')['daily_return'].std()

# ✅ Compute Sharpe Ratio
beta_df['Sharpe Ratio'] = (beta_df['Expected Return'] - risk_free_rate) / stock_volatility
print("✅ Sharpe Ratios computed:\n", beta_df)


# In[74]:


# ✅ Compute Downside Standard Deviation
downside_risk = stock_data[stock_data['daily_return'] < 0].groupby('symbol')['daily_return'].std()

# ✅ Compute Sortino Ratio
beta_df['Sortino Ratio'] = (beta_df['Expected Return'] - risk_free_rate) / downside_risk
print("✅ Sortino Ratios computed:\n", beta_df)


# In[75]:


# ✅ Compute Cumulative Returns for Each Stock
cumulative_returns = stock_data.groupby(['symbol', 'date'])['daily_return'].sum().groupby(level=0).cumsum()

# ✅ Compute Maximum Drawdown
max_drawdowns = {}

for symbol in stock_data['symbol'].unique():
    stock_cum_returns = cumulative_returns.loc[symbol]
    rolling_max = stock_cum_returns.cummax()
    drawdown = (stock_cum_returns - rolling_max) / rolling_max
    max_drawdowns[symbol] = drawdown.min()  # Get worst drawdown

# ✅ Convert to DataFrame
beta_df['Max Drawdown'] = beta_df.index.map(max_drawdowns)
print("✅ Maximum Drawdowns computed:\n", beta_df)


# In[76]:


import psycopg2

# ✅ PostgreSQL Connection
conn = psycopg2.connect(
    dbname="nifty50_data",
    user="postgres",
    password="1234",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

# ✅ Insert financial metrics into stock_metrics table
for index, row in beta_df.iterrows():
    cursor.execute("""
        INSERT INTO stock_metrics (symbol, beta, expected_return, sharpe_ratio, sortino_ratio, max_drawdown)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol) DO UPDATE 
        SET beta = EXCLUDED.beta,
            expected_return = EXCLUDED.expected_return,
            sharpe_ratio = EXCLUDED.sharpe_ratio,
            sortino_ratio = EXCLUDED.sortino_ratio,
            max_drawdown = EXCLUDED.max_drawdown;
    """, (index, row['Beta'], row['Expected Return'], row['Sharpe Ratio'], row['Sortino Ratio'], row['Max Drawdown']))

# ✅ Commit and Close Connection
conn.commit()
cursor.close()
conn.close()

print("✅ Financial metrics stored in stock_metrics table successfully!")


# In[77]:


from sqlalchemy import create_engine

# ✅ Create SQLAlchemy engine
engine = create_engine("postgresql://postgres:1234@localhost:5432/nifty50_data")

# ✅ Use engine instead of raw psycopg2 connection
metrics_df = pd.read_sql("SELECT symbol, expected_return FROM stock_metrics;", engine)
prices_df = pd.read_sql("SELECT date, symbol, close FROM historical_data;", engine)


# In[78]:


# ✅ Print Data
print("Stock Metrics:\n", metrics_df.head())
print("Stock Prices:\n", prices_df.head())


# In[79]:


# ✅ Pivot the price data to have stocks as columns & dates as index
prices_pivot = prices_df.pivot(index="date", columns="symbol", values="close")

# ✅ Compute daily returns
returns = prices_pivot.pct_change().dropna()

# ✅ Compute annualized volatility (Standard Deviation)
stock_volatility = returns.std() * np.sqrt(252)  # 252 trading days

# ✅ Compute covariance matrix (for portfolio risk calculation)
cov_matrix = returns.cov()

# ✅ Print results
print("Volatility:\n", stock_volatility)
print("\nCovariance Matrix:\n", cov_matrix)


# In[80]:


expected_returns = beta_df['Expected Return']


# In[81]:


# Ensure covariance matrix and expected returns use the same symbols
common_symbols = expected_returns.index.intersection(cov_matrix.index)

# Filter both to use only common symbols
expected_returns = expected_returns.loc[common_symbols]
cov_matrix = cov_matrix.loc[common_symbols, common_symbols]


# In[82]:


print("Expected Returns Index:", expected_returns.index)
print("Covariance Matrix Index:", cov_matrix.index)


# In[83]:


extra_stocks = set(cov_matrix.index) - set(expected_returns.index)
print("Extra stocks in cov_matrix:", extra_stocks)


# In[84]:


print("Shape of expected_returns:", expected_returns.shape)
print("Shape of cov_matrix:", cov_matrix.shape)


# In[85]:


# ✅ Check the type BEFORE processing
print("Type of cov_matrix:", type(cov_matrix))  
print("Type of expected_returns:", type(expected_returns))

# If cov_matrix is already a NumPy array, reload it from DataFrame
if isinstance(cov_matrix, np.ndarray):
    print("Reloading cov_matrix as DataFrame...")
    cov_matrix = returns.cov()  # Compute covariance matrix again


# In[86]:


# ✅ Reload covariance matrix as DataFrame if needed
if isinstance(cov_matrix, np.ndarray):
    print("Reloading cov_matrix as DataFrame...")
    cov_matrix = returns.cov()  # Recompute covariance matrix

# ✅ Now, ensure expected_returns only contains stocks in cov_matrix BEFORE conversion
common_symbols = expected_returns.index.intersection(cov_matrix.index)

expected_returns = expected_returns.loc[common_symbols]
cov_matrix = cov_matrix.loc[common_symbols, common_symbols]  # ✅ Ensure same stocks

print("Filtered Expected Returns Index:", expected_returns.index)
print("Filtered Covariance Matrix Index:", cov_matrix.index)

# ✅ Convert covariance matrix to NumPy array (ONLY AFTER FILTERING)
cov_matrix = cov_matrix.to_numpy()

print("✅ Covariance matrix successfully converted to NumPy array!")


# In[87]:


# ✅ Number of Portfolios to Simulate
num_portfolios = 10000  
portfolio_results = []  # Ensure portfolio_results list is initialized

# ✅ Markowitz Efficient Simulation
for _ in range(num_portfolios):
    weights = np.random.dirichlet(np.ones(len(expected_returns)), size=1)[0]  # Corrected length

    # Compute Portfolio Return & Risk
    port_return = np.dot(weights, expected_returns)
    port_volatility = np.sqrt(weights @ cov_matrix @ weights.T)  # No shape mismatch

    # Compute Sharpe Ratio
    sharpe_ratio = port_return / port_volatility if port_volatility != 0 else 0  

    # Store the results
    portfolio_results.append([port_return, port_volatility, sharpe_ratio] + list(weights))

# ✅ Convert to DataFrame
columns = ["Return", "Volatility", "Sharpe Ratio"] + list(expected_returns.index)
portfolio_df = pd.DataFrame(portfolio_results, columns=columns)

# ✅ Display top portfolios
print(portfolio_df.head())


# In[88]:


import numpy as np

# ✅ Number of Portfolios to Simulate
#num_portfolios = 10000  

# ✅ Extract Expected Returns & Covariance Matrix
#expected_returns = metrics_df.set_index("symbol")["expected_return"]
#cov_matrix = returns.cov()

# ✅ Store Portfolio Metrics
#portfolio_results = []

# ✅ Monte Carlo Simulation Loop
#for _ in range(num_portfolios):
    #weights = np.random.dirichlet(np.ones(len(expected_returns)), size=1)[0]  # Random weights

    # Compute Portfolio Return & Risk
    #port_return = np.dot(weights, expected_returns)
    #port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Compute Sharpe Ratio (Risk-Free Rate = 0 for now)
    #sharpe_ratio = port_return / port_volatility if port_volatility != 0 else 0  

    # Store the results
    #portfolio_results.append([port_return, port_volatility, sharpe_ratio] + list(weights))

# ✅ Convert to DataFrame
#columns = ["Return", "Volatility", "Sharpe Ratio"] + list(expected_returns.index)
#portfolio_df = pd.DataFrame(portfolio_results, columns=columns)

# ✅ Display top portfolios
#print(portfolio_df.head())


# In[89]:


# ✅ Portfolio with Maximum Sharpe Ratio
optimal_portfolio = portfolio_df.loc[portfolio_df["Sharpe Ratio"].idxmax()]

# ✅ Print the Best Portfolio Weights
print("Best Portfolio Allocation:")
print(optimal_portfolio)


# In[90]:


# Drop unnecessary columns (keeping only stock allocation weights)
optimal_weights = optimal_portfolio.drop(["Return", "Volatility", "Sharpe Ratio"])


# In[91]:


best_portfolio_return = optimal_portfolio["Return"]
best_portfolio_volatility = optimal_portfolio["Volatility"]
best_portfolio_sharpe_ratio = optimal_portfolio["Sharpe Ratio"]


# In[92]:


print("Optimal Portfolio Weights:")
print(optimal_weights)


# In[93]:


print("Optimal Portfolio Data:")
print(optimal_portfolio)


# In[94]:


optimal_portfolio.count()


# In[95]:


print("Formatted Portfolio DataFrame before insertion:")
print(portfolio_df)


# In[96]:


if not optimal_weights.empty:
    portfolio_df = pd.DataFrame({
        "symbol": optimal_weights.index,
        "allocation": optimal_weights.values
    })
    
    # Add scalar values to all rows
    portfolio_df["return"] = optimal_portfolio["Return"]
    portfolio_df["volatility"] = optimal_portfolio["Volatility"]
    portfolio_df["sharpe_ratio"] = optimal_portfolio["Sharpe Ratio"]
    
    print("🚀 Final Portfolio DataFrame Before Insertion:")
    print(portfolio_df)
else:
    print("❌ No valid portfolio weights found. Check portfolio optimization!")


# In[97]:


import psycopg2

# 🔹 Step 1: Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="nifty50_data",
    user="postgres",
    password="1234",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

# 🔹 Step 2: Create the table if it doesn't exist
cursor.execute("""
    CREATE TABLE IF NOT EXISTS optimized_portfolio (
        symbol TEXT PRIMARY KEY,
        allocation FLOAT,
        return FLOAT,
        volatility FLOAT,
        sharpe_ratio FLOAT
    );
""")
conn.commit()

# 🔹 Step 3: Insert Data into PostgreSQL
for _, row in portfolio_df.iterrows():
    cursor.execute("""
        INSERT INTO optimized_portfolio (symbol, allocation, return, volatility, sharpe_ratio)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (symbol) DO UPDATE
        SET allocation = EXCLUDED.allocation,
            return = EXCLUDED.return,
            volatility = EXCLUDED.volatility,
            sharpe_ratio = EXCLUDED.sharpe_ratio;
    """, (row["symbol"], row["allocation"], row["return"], row["volatility"], row["sharpe_ratio"]))

# 🔹 Step 4: Commit & Close Connection
conn.commit()
cursor.close()
conn.close()

print("✅ Data successfully inserted into PostgreSQL!")


# In[98]:


from sqlalchemy import create_engine

# ✅ Create PostgreSQL Connection
engine = create_engine("postgresql://postgres:1234@localhost:5432/nifty50_data")

# ✅ Store Cleaned Historical Data
historical_data.to_sql('historical_data', engine, if_exists='replace', index=False)
print("✅ Cleaned Historical Data Stored Successfully.")

# ✅ Store Cleaned Real-Time Data
real_time_data.to_sql('real_time_data', engine, if_exists='replace', index=False)
print("✅ Cleaned Real-Time Data Stored Successfully.")



# In[99]:


historical_data['symbol']


# In[100]:


print(portfolio_df["symbol"].tolist())  # Check all symbols before inserting


# In[101]:


historical_data.columns


# In[102]:


##STOCK PREDICTIONS USING XGBOOST MODELS


# In[103]:


##new features of XGBOOST


# In[104]:


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sqlalchemy import create_engine

# 🔹 PostgreSQL Connection
DB_URI = "postgresql://postgres:1234@localhost:5432/nifty50_data"
engine = create_engine(DB_URI)

# 🔹 Load Real-Time Stock Data
def load_real_time_data():
    query = "SELECT * FROM real_time_data ORDER BY date DESC LIMIT 500;"  # Fetch last 500 records
    return pd.read_sql(query, engine)

# 🔹 Feature Engineering
def preprocess_data(df):
    df = df.copy()

    # ✅ Moving Averages
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()

    # ✅ Momentum Indicators
    df['RSI_14'] = 100 - (100 / (1 + df['close'].pct_change().rolling(14).mean()))
    df['MACD'] = df['EMA_50'] - df['close'].ewm(span=26, adjust=False).mean()

    # ✅ New Features
    df['volatility_10'] = df['close'].rolling(window=10).std()
    df['momentum_10'] = df['close'].diff(10)
    df['williams_r'] = ((df['high'].rolling(14).max() - df['close']) / 
                         (df['high'].rolling(14).max() - df['low'].rolling(14).min())) * -100
    df['rolling_mean_10'] = df['close'].rolling(10).mean()
    df['normalized_volume'] = df['volume'] / df['volume'].max()

    selected_features = ['open', 'low', 'high', 'SMA_20', 'EMA_50', 'RSI_14', 'MACD', 
                         'volatility_10', 'momentum_10', 'williams_r', 'rolling_mean_10', 'normalized_volume']

    df_processed = df[selected_features].fillna(0)
    return df_processed, df  # Return processed & original data

# 🔹 Train XGBoost Model
def train_xgboost():
    print("🚀 Training XGBoost Model...")

    df_hist = pd.read_sql("SELECT * FROM historical_data", con=engine)
    df_real = load_real_time_data()

    df_combined = pd.concat([df_hist, df_real]).reset_index(drop=True)
    df_processed, df_original = preprocess_data(df_combined)

    X = df_processed
    y = df_original['close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [300, 500],  
        'max_depth': [3, 4],  
        'learning_rate': [0.01, 0.03]
    }

    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    print(f"✅ RMSE (Train Data): {rmse:.4f}")
    print(f"✅ MAPE (Train Data): {mape:.2f}%")

    best_model.save_model("xgboost_model_final.json")
    print("✅ Model Trained & Saved as 'xgboost_model_final.json'.")

# 🔹 Fine-Tune XGBoost Model
def fine_tune_xgboost():
    """Fine-tune the trained model using latest real-time stock data."""
    print("🔄 Fine-Tuning Model with Latest Real-Time Data...")

    df_real = load_real_time_data()
    df_processed, df_original = preprocess_data(df_real)

    # ✅ Load Pre-Trained Model  
    model = xgb.XGBRegressor()
    model.load_model("xgboost_model_final.json")

    # ✅ Splitting Real-Time Data for Fine-Tuning
    X_train, X_val, y_train, y_val = train_test_split(
        df_processed, df_original['close'], test_size=0.2, random_state=42
    )

    # ✅ Fine-Tune with Validation Set
    model.set_params(early_stopping_rounds=10)  # Set early stopping globally
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

    # ✅ Save Updated Model  
    model.save_model("xgboost_model_finetuned_final.json")
    print("✅ Fine-Tuned Model Saved as 'xgboost_model_finetuned_final.json'.")


# 🔹 Evaluate Model Performance
def evaluate_predictions(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    print(f"✅ RMSE: {rmse:.4f}")
    print(f"✅ MAPE: {mape:.2f}%")
    return rmse, mape  # ✅ **Fixed the missing return statement**

# 🔹 Make Real-Time Predictions
def predict_real_time():
    """Make real-time stock price predictions and save them to PostgreSQL."""
    print("🚀 Loading Real-Time Data...")
    df_real_time = load_real_time_data()
    df_real_time_processed, df_original = preprocess_data(df_real_time)

    print("🚀 Loading Fine-Tuned XGBoost Model...")
    model = xgb.XGBRegressor()
    model.load_model("xgboost_model_finetuned_final.json")  # Load fine-tuned model

    print("🚀 Making Predictions...")
    predictions = model.predict(df_real_time_processed)

    # ✅ Add Predictions to Original DataFrame  
    df_original['predicted_close'] = predictions

    print("🚀 Evaluating Predictions...")
    rmse, mape = evaluate_predictions(df_original['close'], df_original['predicted_close'])

    print("🚀 Saving Predictions to PostgreSQL...")
    df_original[['symbol', 'date', 'close', 'predicted_close']].to_sql(
        'stock_predictions', con=engine, if_exists='replace', index=False
    )

    print("✅ Predictions Saved to PostgreSQL Successfully!")
    print("🚀 Predictions Completed! Here's a Preview:")
    print(df_original[['symbol', 'date', 'close', 'predicted_close']].head(30))

    return df_original, rmse, mape  # Return results

# 🔹 Run Pipeline
if __name__ == "__main__":
    print("🔄 Training the Final Best Model...")
    train_xgboost()

    print("\n🔄 Fine-Tuning with Latest Real-Time Data...")
    fine_tune_xgboost()

    print("\n🔄 Running Real-Time Predictions...")
    predict_real_time()


# In[105]:


import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# 🔹 PostgreSQL Connection
DB_URI = "postgresql://postgres:1234@localhost:5432/nifty50_data"
engine = create_engine(DB_URI)

# 🔹 Load Real-Time Data
def load_real_time_data():
    query = "SELECT * FROM real_time_data ORDER BY date DESC LIMIT 500;"  # Fetch last 500 records
    df = pd.read_sql(query, engine)
    return df

# 🔹 Feature Engineering
def preprocess_data(df):
    df = df.copy()

    # ✅ Moving Averages
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()

    # ✅ Momentum Indicators
    df['RSI_14'] = 100 - (100 / (1 + df['close'].pct_change().rolling(14).mean()))
    df['MACD'] = df['EMA_50'] - df['close'].ewm(span=26, adjust=False).mean()

    # ✅ New Features
    df['volatility_10'] = df['close'].rolling(window=10).std()  
    df['momentum_10'] = df['close'].diff(10)  
    df['williams_r'] = ((df['high'].rolling(14).max() - df['close']) / 
                         (df['high'].rolling(14).max() - df['low'].rolling(14).min())) * -100  
    df['rolling_mean_10'] = df['close'].rolling(10).mean()  

    df['normalized_volume'] = df['volume'] / df['volume'].max()

    selected_features = ['open', 'low', 'high', 'SMA_20', 'EMA_50', 'RSI_14', 'MACD', 
                         'volatility_10', 'momentum_10', 'williams_r', 'rolling_mean_10', 'normalized_volume']

    df_processed = df[selected_features].fillna(0)  
    return df_processed, df  

# 🔹 Prediction Function
def predict_real_time():
    print("🚀 Loading Real-Time Data...")
    df_real_time = load_real_time_data()
    df_real_time_processed, df_original = preprocess_data(df_real_time)

    print("🚀 Loading Fine-Tuned XGBoost Model...")
    model = xgb.XGBRegressor()
    model.load_model("xgboost_model_finetuned_final.json")

    print("🚀 Making Predictions...")
    predictions = model.predict(df_real_time_processed)

    df_original['predicted_close'] = predictions

    print("🚀 Evaluating Predictions...")
    rmse, mape = evaluate_predictions(df_original['close'], df_original['predicted_close'])

    print("🚀 Saving Predictions to PostgreSQL...")
    df_original[['symbol', 'date', 'close', 'predicted_close']].to_sql(
        'stock_predictions', con=engine, if_exists='replace', index=False
    )

    print("✅ Predictions Saved Successfully!")
    
    print("🚀 Generating Visualizations...")
    visualize_predictions(df_original)

    return df_original, rmse, mape 

# 🔹 Evaluation Function
def evaluate_predictions(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    print(f"✅ RMSE: {rmse:.4f}")
    print(f"✅ MAPE: {mape:.2f}%")
    return rmse, mape

# 🔹 Visualization Function
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_predictions(df):
    fig, axes = plt.subplots(4, 1, figsize=(15, 25))  # 🔹 4 Rows, 1 Column, More Height

    # ✅ 1️⃣ Actual vs Predicted Prices (Line Chart)
    axes[0].plot(df['date'], df['close'], label='Actual Close Price', color='blue', marker='o')
    axes[0].plot(df['date'], df['predicted_close'], label='Predicted Close Price', color='red', linestyle='dashed', marker='x')
    axes[0].set_xlabel("Date", fontsize=14)
    axes[0].set_ylabel("Stock Price", fontsize=14)
    axes[0].set_title("Actual vs Predicted Stock Prices", fontsize=16, fontweight='bold')
    axes[0].legend(fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)

    # ✅ 2️⃣ Prediction Error Distribution (Histogram)
    error = df['close'] - df['predicted_close']
    sns.histplot(error, bins=30, kde=True, color='purple', ax=axes[1])
    axes[1].set_xlabel("Prediction Error (Close - Predicted)", fontsize=14)
    axes[1].set_ylabel("Count", fontsize=14)
    axes[1].set_title("Prediction Error Distribution", fontsize=16, fontweight='bold')

    # ✅ 3️⃣ Stock-wise Actual vs Predicted Prices (Bar Chart)
    df_grouped = df.groupby("symbol").agg({"close": "mean", "predicted_close": "mean"})
    df_grouped.plot(kind="bar", ax=axes[2], width=0.7)
    axes[2].set_xlabel("Stock Symbol", fontsize=14)
    axes[2].set_ylabel("Price", fontsize=14)
    axes[2].set_title("Stock-wise Actual vs Predicted Prices", fontsize=16, fontweight='bold')
    axes[2].tick_params(axis='x', rotation=45)

    # ✅ 4️⃣ Stock Price Trend Analysis (Moving Average)
    df_sorted = df.sort_values("date")
    axes[3].plot(df_sorted['date'], df_sorted['close'].rolling(10).mean(), label="10-day Rolling Mean (Actual)", color="blue")
    axes[3].plot(df_sorted['date'], df_sorted['predicted_close'].rolling(10).mean(), label="10-day Rolling Mean (Predicted)", color="red")
    axes[3].set_xlabel("Date", fontsize=14)
    axes[3].set_ylabel("Stock Price", fontsize=14)
    axes[3].set_title("Stock Price Trend Analysis (Moving Average)", fontsize=16, fontweight='bold')
    axes[3].legend(fontsize=12)
    axes[3].tick_params(axis='x', rotation=45)

    plt.subplots_adjust(hspace=0.7)  # 🔹 Adds More Vertical Space Between Charts
    plt.show()


# 🔹 Run Everything
if __name__ == "__main__":
    print("\n🔄 Running Real-Time Predictions & Visualizations...")
    predict_real_time()


# In[106]:


import os
print(os.getcwd())  # Check current working directory
print(os.listdir()) # List all files in the directory


# In[107]:


from sqlalchemy import create_engine

# ✅ Create PostgreSQL Connection
engine = create_engine("postgresql://postgres:1234@localhost:5432/nifty50_data")

# ✅ Store Cleaned Historical Data
historical_data.to_sql('historical_data', engine, if_exists='replace', index=False)
print("✅ Cleaned Historical Data Stored Successfully.")

# ✅ Store Cleaned Real-Time Data
real_time_data.to_sql('real_time_data', engine, if_exists='replace', index=False)
print("✅ Cleaned Real-Time Data Stored Successfully.")



# In[108]:


historical_data.columns


# In[109]:


real_time_data.columns


# In[110]:





# In[ ]:






