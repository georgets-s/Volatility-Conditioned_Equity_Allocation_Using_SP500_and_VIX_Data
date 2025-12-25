
import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt


#%%


# Load data
sp = pd.read_csv("sp.csv", parse_dates=["Date"])
Close_y = pd.read_csv("vix.csv", parse_dates=["Date"])

# Merge datasets
df = sp.merge(Close_y, on="Date", how="inner")
df = df.sort_values("Date").reset_index(drop=True)


#%%


# Moving averages
df["MA10"] = talib.SMA(df["Close_x"], timeperiod=10)
df["MA30"] = talib.SMA(df["Close_x"], timeperiod=30)

# RSI
df["RSI"] = talib.RSI(df["Close_x"], timeperiod=14)


#%%


df["Signal_Price"] = (
    (df["MA10"] > df["MA30"]) &
    (df["RSI"] > 30) &
    (df["RSI"] < 70)
).astype(int)


#%%


df["VIX_Mean"] = df["Close_y"].rolling(60).mean()
df["VIX_Std"] = df["Close_y"].rolling(60).std()
df["VIX_z"] = (df["Close_y"] - df["VIX_Mean"]) / df["VIX_Std"]


#%%


df["Vol_Regime"] = np.select(
    [
        df["VIX_z"] < 0,
        (df["VIX_z"] >= 0) & (df["VIX_z"] < 1),
        df["VIX_z"] >= 1
    ],
    [1.0, 0.5, 0.0]  # exposure levels
)


#%%


# Portfolio 1: price only
df["Final_Price"] = df["Signal_Price"]

# Portfolio 2: price + volatility filter
df["Final_Price_VIX"] = df["Signal_Price"] * df["Vol_Regime"]


#%%


df["Market_Return"] = df["Close_x"].pct_change()

df["Ret_Price"] = df["Final_Price"].shift(1) * df["Market_Return"]
df["Ret_Price_VIX"] = df["Final_Price_VIX"].shift(1) * df["Market_Return"]

df = df.dropna()


#%%


def performance_metrics(returns, periods_per_year=252):
    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    annualized_return = cumulative.iloc[-1] ** (periods_per_year / len(returns)) - 1
    annualized_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = annualized_return / annualized_vol if annualized_vol != 0 else np.nan
    drawdown = cumulative / cumulative.cummax() - 1
    max_dd = drawdown.min()
    return total_return, annualized_return, sharpe, max_dd


#%%


price_metrics = performance_metrics(df["Ret_Price"])
vix_metrics = performance_metrics(df["Ret_Price_VIX"])
bh_metrics = performance_metrics(df["Market_Return"])


#%%


results = pd.DataFrame(
    [price_metrics, vix_metrics, bh_metrics],
    columns=["Total Return", "Annualized Return", "Sharpe Ratio", "Max Drawdown"],
    index=["Price Only", "Price + VIX", "Buy & Hold"]
).round(4)

print(results)
results.to_excel("portfolio_comparison.xlsx")


#%%


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


#%%












