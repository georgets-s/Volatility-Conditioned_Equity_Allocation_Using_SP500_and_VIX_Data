# Volatility-Conditioned_Equity_Allocation_Using_SP500_and_VIX_Data


# Project Description
This project evaluates whether conditioning equity exposure on market volatility improves portfolio performance relative to a price only strategy. Using daily S&P 500 data (from the early 2000 to late 2024), a baseline portfolio is constructed based on standard technical indicators (moving averages and RSI). A second portfolio applies the same price based signals but dynamically adjusts exposure using volatility regimes derived from the VIX index.

The VIX is standardized using a rolling window to identify low, medium, and high volatility environments, which are mapped to different equity exposure levels. Portfolio performance is assessed through cumulative returns, Sharpe ratios, and maximum drawdowns, with comparisons against a buy-and-hold benchmark.

The project focuses on risk management and regime based allocation rather than return prediction, reflecting institutional approaches used in asset management and insurance portfolios.



# Definition and Mapping of volatility Regimes

| Regime            | Condition     | Interpretation           |
| ----------------- | ------------- | ------------------------ |
| Low volatility    | VIX_z < 0     | Risk-on environment      |
| Medium volatility | 0 ≤ VIX_z < 1 | Transitional uncertainty |
| High volatility   | VIX_z ≥ 1     | Stress / risk-off regime |

Low volatility → 100% equity exposure
Medium volatility → 50% equity exposure
High volatility → 0% equity exposure (capital preservation)


# Volatility integration with price based signals
Final Signal(t)=Price Signal(t)×Volatility Exposure(t)

