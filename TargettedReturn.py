import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from gamspy import Container, Set, Parameter, Variable, Equation, Model, Sum, Sense, Options
import numpy as np
import sys
import gamspy as gp
import math
import matplotlib.pyplot as plt
import pickle
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import pickle

# Step 1: Fetch the list of S&P 500 companies from Wikipedia
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find the table with the tickers
table = soup.find('table', {'class': 'wikitable'})

# Extract all tickers and limit to the first 5
all_rows = table.findAll('tr')[1:]  # Skip the header row
limited_rows = all_rows[:5]  # Consider only the first 5 rows (stocks)
tickers = [row.findAll('td')[0].text.strip() for row in limited_rows]

# File paths for saving/loading data
price_data_pickle_path = 'price_data.pkl'
tickers_pickle_path = 'tickers.pkl'

# Check if the pickle files exist
try:
    # Try loading the data from pickle files
    with open(price_data_pickle_path, 'rb') as f:
        price_data = pickle.load(f)
    with open(tickers_pickle_path, 'rb') as f:
        loaded_tickers = pickle.load(f)
    
    # Ensure only the first 5 tickers are used
    tickers = tickers[:5]
    price_data = {ticker: price_data[ticker] for ticker in tickers if ticker in price_data}
    print("Data loaded from pickle files.")
except FileNotFoundError:
    # If pickle files don't exist, fetch the data
    print("Pickle files not found. Fetching data...")
    price_data = {}  # Dictionary to store price data for each ticker
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(months=13)

    # Step 2: Fetch historical price data for the first 5 tickers (last 13 months)
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)  # Fetch data for the last 13 months
            
            # Resample to get the last closing price of each month
            monthly_prices = hist['Close'].resample('ME').last()  # Use 'ME' for month-end frequency
            
            # Ensure we have exactly 13 months of data
            if len(monthly_prices) >= 13:
                price_data[ticker] = monthly_prices[-13:]  # Keep only the last 13 months
            else:
                print(f"Insufficient data for {ticker}, skipping.")
        except Exception as e:
            print(f"Failed to fetch data for {ticker}: {e}")

    # Save the fetched data to pickle files
    with open(price_data_pickle_path, 'wb') as f:
        pickle.dump(price_data, f)
    with open(tickers_pickle_path, 'wb') as f:
        pickle.dump(tickers, f)
    print("Data saved to pickle files.")

# Step 3: Combine the data into a single DataFrame
price_df = pd.DataFrame(price_data).T  # Transpose so that rows are stocks and columns are months

# Step 4: Save the data to a CSV file
price_df.to_csv('sp500_last_day_monthly_prices.csv')

# Step 5: Load the data back again
price_df = pd.read_csv('sp500_last_day_monthly_prices.csv', index_col=0)

# Step 6: Convert the DataFrame to a NumPy array
price_array = price_df.to_numpy()

# Step 7: Print the resulting array
print("Price array shape:", price_array.shape)  # Should be (5, 13)
print("Tickers shape:", len(tickers))  # Number of successfully loaded tickers


# Portfolio Configuration Parameters
max_exposure = 0.15  # Maximum exposure per stock
buy_exposure = 0.10  # Maximum buying limit as a fraction of NAV
max_return = 1.80  # Cap on maximum return per stock
PT = 0.50  # Portfolio turnover rate
cash_min = 0.001  # Minimum cash position as a fraction of NAV
bfee = 1.01001  # Buy fee multiplier
sfee = 0.99  # Sell fee multiplier
min_buy = 0.001  # Minimum purchase amount
expected_return = 1.30  # Target return for the portfolio
init_cash = 1000000  # Initial cash balance
eps = 0.0001
M = 100000000

buy_minimum = 40000


# Define the model container
m = Container()

# Sets
time_index = 13  # Total number of time periods
stocks = Set(container=m, name="stocks", records=tickers)  # Set of stocks
time = Set(container=m, name="time", records=np.arange(0, time_index, 1))  # Time periods
subtime = Set(container=m, domain=time, records=np.arange(1, time_index, 1))  # Excludes first period
subtime2 = Set(container=m, domain=time, records=np.arange(2, time_index, 1))  # Excludes first two periods

# Data: Stock prices indexed by stocks and time
prices = Parameter(
    container=m,
    name="prices",
    domain=[stocks, time],
    records=price_array,
)

# Variables
C = Variable(container=m, name="C", domain=time, type="Positive")  # Cash balance
V = Variable(container=m, name="V", domain=time, type="Positive")  # Net Asset Value
x = Variable(container=m, name="x", domain=[stocks, time], type="Positive")  # Holdings
x.fx[stocks, '0'] = 0  # No initial holdings

b = Variable(container=m, name="b", domain=[stocks, time], type="Positive")  # Stocks bought
b.fx[stocks, '0'] = 0  # No initial buying

s = Variable(container=m, name="s", domain=[stocks, time], type="Positive")  # Stocks sold

Z = Variable(container=m, name="Z", type="free")  # Objective variable
Sr = Variable(container=m, name="Sr", domain=[stocks], type="Positive")  # Total sales
Br = Variable(container=m, name="Br", domain=[stocks], type="Positive")  # Total purchases
#Br.lo[stocks] = 20000


profit = Variable(container=m, name="profit", domain=[stocks], type="Positive")  # Profit per stock

## Binary variable if stock i is a winner
omega = Variable(container=m, name="omega", domain=[stocks], type="Binary")

## Binary variable if stock i was a trade
omega_trades = Variable(container=m, name="omega_stock", domain=[stocks], type="Binary")


## winner_determination of stock i
winner_determination = Equation(
    m,
    name="winner_determination",
    domain=[stocks],
    description="Determines if investing in stock i was a winner through the lifetime of the fund"
)
winner_determination[stocks] = (
    (Sr[stocks] + x[stocks, str(time_index-1)]*prices[stocks, str(time_index-1)]) >= Br[stocks] + eps*omega[stocks] - M*(1-omega[stocks])
)



## loser_determination of stock i
loser_determination = Equation(
    m,
    name="loser_determination",
    domain=[stocks],
    description="Determines if investing in stock i was a loser through the lifetime of the fund"
)
loser_determination[stocks] = (
    (Sr[stocks] + x[stocks, str(time_index-1)]*prices[stocks, str(time_index-1)]) -  Br[stocks]  <=  -eps*(1-omega[stocks]) + M*omega[stocks]
)


## Determination of omega_trades for stock i
omega_trades_determination = Equation(
    m,
    name="omega_trades_determination",
    domain=[stocks],
    description="Determines if Br[stocks] > 1 and sets omega_trades to 1 in that case"
)
omega_trades_determination[stocks] = (
    Br[stocks] <= 1000000 * omega_trades[stocks]
)



## Determination of omega_trades for stock i
omega_trades_max = Equation(
    m,
    name="omega_trades_max",
    domain=[stocks],
    description="Determines the maximum number of stocks that can be bought"
)
omega_trades_max[stocks] = (
    Br[stocks] >= buy_minimum * omega_trades[stocks]
)



# Equations: Total sales and purchases
total_sales = Equation(
    m,
    name="total_sales",
    domain=[stocks],
    description="Total revenue from selling stock",
)
total_sales[stocks] = Sr[stocks] == Sum(time, prices[stocks, time] * s[stocks, time])

total_purchases = Equation(
    m,
    name="total_purchases",
    domain=[stocks],
    description="Total cost from buying stock",
)
total_purchases[stocks] = Br[stocks] == Sum(time, prices[stocks, time] * b[stocks, time])

# Profit calculation
pnl = Equation(
    m,
    name="pnl",
    domain=[stocks],
    description="Calculate profit for each stock",
)
pnl[stocks] = (
    profit[stocks] ==
    (Sr[stocks] + x[stocks, str(time_index - 1)] * prices[stocks, str(time_index - 1)]) - Br[stocks]
)

# Turnover constraint
portfolio_turnover = Equation(
    m,
    name="portfolio_turnover",
    description="Portfolio turnover calculation",
)
portfolio_turnover[...] = (
    PT * (V["1"] + V[str(time_index - 1)]) ==
    Sum(stocks, Sum(subtime2, b[stocks, subtime2] * prices[stocks, subtime2])) * 2
)

# Max returns per stock
max_returns = Equation(
    m,
    name="max_returns",
    domain=[stocks],
    description="Cap on maximum returns per stock",
)
max_returns[stocks] = (
    (Sr[stocks] + x[stocks, str(time_index - 1)] * prices[stocks, str(time_index - 1)]) <=
    Br[stocks] * max_return
)

# Holdings and cash balance equations
holdings_balance = Equation(
    m,
    name="holdings_balance",
    domain=[stocks, time],
    description="Holdings balance over time",
)
holdings_balance[stocks, subtime] = (
    x[stocks, subtime] == x[stocks, subtime - 1] + b[stocks, subtime] - s[stocks, subtime]
)

holdings_balance_initial = Equation(
    m,
    name="holdings_balance_initial",
    domain=[stocks],
    description="Initial stock holdings",
)
holdings_balance_initial[stocks] = x[stocks, "0"] == 0

cash_balance = Equation(
    m,
    name="cash_balance",
    domain=[time],
    description="Cash balance at each time period",
)
cash_balance[time] = (
    C[time] == C[time - 1] -
    Sum(stocks, prices[stocks, time] * bfee * b[stocks, time]) +
    Sum(stocks, prices[stocks, time] * sfee * s[stocks, time])
)

cash_balance_initial = Equation(
    m,
    name="cash_balance_initial",
    description="Initial cash balance",
)
cash_balance_initial[...] = C["0"] == init_cash

# NAV calculation
nav = Equation(
    m,
    name="nav",
    domain=[time],
    description="Net Asset Value calculation",
)
nav[time] = V[time] == C[time] + Sum(stocks, prices[stocks, time] * x[stocks, time])

# Risk constraints
risk_constraint = Equation(
    m,
    name="risk_constraint",
    domain=[stocks, time],
    description="Limit position exposure",
)
risk_constraint[stocks, time] = max_exposure * V[time] >= prices[stocks, time] * x[stocks, time]

buy_risk_constraint = Equation(
    m,
    name="buy_risk_constraint",
    domain=[stocks, time],
    description="Limit buying exposure",
)
buy_risk_constraint[stocks, time] = buy_exposure * V[time] >= prices[stocks, time] * b[stocks, time]

# Deviation and objective function
Z_plus = Variable(container=m, name="Z_plus", type="Positive")
Z_minus = Variable(container=m, name="Z_minus", type="Positive")
constraint_deviation = Equation(
    container=m,
    name="constraint_deviation",
)
constraint_deviation[...] = (
    Sum(stocks, omega[stocks]) == Z_plus #- Z_minus
)


# Deviation and objective function
portfolio_returns = Equation(
    container=m,
    name="portfolio_returns",
)
portfolio_returns[...] = (
    V[str(time_index - 1)] - expected_return * init_cash  == 0
)




obj_function = Equation(
    container=m,
    name="obj_function",
)
obj_function[...] = Z == Z_plus 

# Model definition
b1 = Model(
    container=m,
    name="b1",
    equations=m.getEquations(),
    problem="MIP",
    sense=Sense.MIN,
    objective=Z,
)

# Solve the model
gdx_path = m.gdxOutputPath()
b1.solve(
    output=sys.stdout,
    options=Options(report_solution=1),
    solver_options={
        "reslim": "100",
         "SolnPoolReplace": 2,
         "SolnPoolIntensity": 4,
         "SolnPoolPop": 2,
         "PopulateLim": 1000,
         "solnpoolmerge": "mysol.gdx",
    }
)


print(omega_trades.records.level.sum())