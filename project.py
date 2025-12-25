import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt


# =====================================================================
# DATA LOADING AND MERGING
# =====================================================================

# Load S&P 500 price data and VIX data
# Assumption:
# - sp.csv contains Date and S&P 500 price (Close_x after merge)
# - vix.csv contains Date and VIX index level (Close_y after merge)
sp = pd.read_csv("sp.csv", parse_dates=["Date"])
Close_y = pd.read_csv("vix.csv", parse_dates=["Date"])

# Merge datasets on date to ensure aligned observations
df = sp.merge(Close_y, on="Date", how="inner")

# Sort chronologically and reset index
df = df.sort_values("Date").reset_index(drop=True)


# =====================================================================
# PRICE-BASED TECHNICAL INDICATORS
# =====================================================================

# Compute short-term and medium-term moving averages
# These capture trend direction in the equity market
df["MA10"] = talib.SMA(df["Close_x"], timeperiod=10)
df["MA30"] = talib.SMA(df["Close_x"], timeperiod=30)

# Compute Relative Strength Index (RSI)
# Used to avoid entering positions during extreme momentum conditions
df["RSI"] = talib.RSI(df["Close_x"], timeperiod=14)


# =====================================================================
# PRICE-ONLY TRADING SIGNAL (BASELINE PORTFOLIO)
# =====================================================================

# Price-based signal:
# - Positive trend (MA10 > MA30)
# - Momentum not overbought or oversold (30 < RSI < 70)
df["Signal_Price"] = (
    (df["MA10"] > df["MA30"]) &
    (df["RSI"] > 30) &
    (df["RSI"] < 70)
).astype(int)


# =====================================================================
# VIX-BASED VOLATILITY REGIME CONSTRUCTION
# =====================================================================

# Normalize VIX using a rolling z-score
# This allows volatility to be interpreted relative to recent conditions
df["VIX_Mean"] = df["Close_y"].rolling(60).mean()
df["VIX_Std"] = df["Close_y"].rolling(60).std()
df["VIX_z"] = (df["Close_y"] - df["VIX_Mean"]) / df["VIX_Std"]


# =====================================================================
# VOLATILITY REGIMES AND EXPOSURE SCALING
# =====================================================================

# Define volatility regimes and map them to equity exposure levels:
# - Low volatility: full exposure (1.0)
# - Medium volatility: reduced exposure (0.5)
# - High volatility: no exposure (0.0)
df["Vol_Regime"] = np.select(
    [
        df["VIX_z"] < 0,
        (df["VIX_z"] >= 0) & (df["VIX_z"] < 1),
        df["VIX_z"] >= 1
    ],
    [1.0, 0.5, 0.0]
)


# =====================================================================
# FINAL PORTFOLIO SIGNALS
# =====================================================================

# Portfolio 1: Price-only strategy
df["Final_Price"] = df["Signal_Price"]

# Portfolio 2: Price strategy conditioned on volatility regime
df["Final_Price_VIX"] = df["Signal_Price"] * df["Vol_Regime"]


# =====================================================================
# RETURN CALCULATION (NO LOOK-AHEAD BIAS)
# =====================================================================

# Compute daily market returns
df["Market_Return"] = df["Close_x"].pct_change()

# Strategy returns:
# Signals are lagged by one period to ensure decisions use only
# information available at time t
df["Ret_Price"] = df["Final_Price"].shift(1) * df["Market_Return"]
df["Ret_Price_VIX"] = df["Final_Price_VIX"].shift(1) * df["Market_Return"]

# Drop initial NaN values created by rolling calculations and shifts
df = df.dropna()


# =====================================================================
# PERFORMANCE METRICS FUNCTION
# =====================================================================

def performance_metrics(returns, periods_per_year=252):
    """
    Compute standard portfolio performance metrics.

    Parameters
    ----------
    returns : pd.Series
        Series of portfolio returns.
    periods_per_year : int
        Number of trading periods per year (252 for daily data).

    Returns
    -------
    tuple
        Total return, annualized return, Sharpe ratio, maximum drawdown.
    """
    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    annualized_return = cumulative.iloc[-1] ** (periods_per_year / len(returns)) - 1
    annualized_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = annualized_return / annualized_vol if annualized_vol != 0 else np.nan
    drawdown = cumulative / cumulative.cummax() - 1
    max_dd = drawdown.min()
    return total_return, annualized_return, sharpe, max_dd


# =====================================================================
# PERFORMANCE EVALUATION
# =====================================================================

# Compute metrics for each strategy and buy-and-hold benchmark
price_metrics = performance_metrics(df["Ret_Price"])
vix_metrics = performance_metrics(df["Ret_Price_VIX"])
bh_metrics = performance_metrics(df["Market_Return"])


# =====================================================================
# RESULTS TABLE
# =====================================================================

results = pd.DataFrame(
    [price_metrics, vix_metrics, bh_metrics],
    columns=["Total Return", "Annualized Return", "Sharpe Ratio", "Max Drawdown"],
    index=["Price Only", "Price + VIX", "Buy & Hold"]
).round(4)

print(results)

# Save results for reporting
results.to_excel("portfolio_comparison.xlsx")


# =====================================================================
# CUMULATIVE PERFORMANCE PLOT
# =====================================================================

df["Cum_Price"] = (1 + df["Ret_Price"]).cumprod()
df["Cum_Price_VIX"] = (1 + df["Ret_Price_VIX"]).cumprod()
df["Cum_BH"] = (1 + df["Market_Return"]).cumprod()

plt.figure(figsize=(12,6))
plt.plot(df["Date"], df["Cum_Price"], label="Price Only")
plt.plot(df["Date"], df["Cum_Price_VIX"], label="Price + VIX")
plt.plot(df["Date"], df["Cum_BH"], label="Buy & Hold", linestyle="--")
plt.legend()
plt.title("Cumulative Portfolio Performance")
plt.grid(True)
plt.show()
