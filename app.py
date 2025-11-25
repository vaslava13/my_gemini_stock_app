import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.plotting import plot_efficient_frontier

# --- CONFIGURATION ---
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# --- HELPER FUNCTIONS (PORTFOLIO) ---

def optimize_portfolio(tickers, current_holdings, start_date='2020-01-01', end_date=None):
    """Fetches stock prices and calculates optimal portfolios."""
    try:
        with st.spinner("Downloading historical stock data..."):
            df = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    df = df['Close']
                except KeyError:
                    df = df.xs('Close', level=0, axis=1)

        if df.empty:
            st.error("❌ No data downloaded. Please check your ticker symbols.")
            return None, None, None, None

        df = df.dropna(axis=1, how='all').dropna() 
        if df.shape[1] < 2:
            st.error("⚠️ Not enough valid stocks to form a portfolio. At least two are required.")
            return None, None, None, None

        latest_prices = df.iloc[-1]
        valid_tickers_held = [t for t in tickers if t in current_holdings and t in latest_prices.index]
        
        if not valid_tickers_held:
            st.error("❌ None of your tickers could be found in the data.")
            return None, None, None, None

        total_portfolio_value = sum(current_holdings[ticker] * latest_prices[ticker] for ticker in valid_tickers_held)
        current_weights = {ticker: (current_holdings[ticker] * latest_prices[ticker]) / total_portfolio_value for ticker in valid_tickers_held}

        mu = expected_returns.mean_historical_return(df)
        S = risk_models.sample_cov(df)

        portfolios = {}
        # 1. Low Risk
        ef_low = EfficientFrontier(mu, S)
        ef_low.min_volatility()
        portfolios['low_risk'] = {'weights': ef_low.clean_weights(), 'performance': ef_low.portfolio_performance()}

        # 2. Medium Risk
        ef_med = EfficientFrontier(mu, S)
        ef_med.max_sharpe()
        portfolios['medium_risk'] = {'weights': ef_med.clean_weights(), 'performance': ef_med.portfolio_performance()}

        # 3. High Risk
        min_ret, _, _ = portfolios['low_risk']['performance']
        max_ret = mu.max()
        target_return = min_ret + 0.8 * (max_ret - min_ret)
        ef_high = EfficientFrontier(mu, S)
        try:
            ef_high.efficient_return(target_return)
            portfolios['high_risk'] = {'weights': ef_high.clean_weights(), 'performance': ef_high.portfolio_performance()}
        except:
            portfolios['high_risk'] = portfolios['medium_risk']

        # Plotting
        current_weights_series = pd.Series(current_weights).reindex(df.columns, fill_value=0)
        curr_ret = np.sum(mu * current_weights_series)
        curr_std = np.sqrt(np.dot(current_weights_series.T, np.dot(S, current_weights_series)))

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_efficient_frontier(EfficientFrontier(mu, S), ax=ax, show_assets=False)
        
        scenarios = [('low_risk', 'Lowest Risk', 'green', 'o'), ('medium_risk', 'Max Sharpe', 'blue', '*'), ('high_risk', 'High Risk', 'red', 'X')]
        for key, label, color, marker in scenarios:
            r, v, _ = portfolios[key]['performance']
            ax.scatter(v, r, marker=marker, s=150, c=color, label=label, zorder=5)
            
        ax.scatter(curr_std, curr_ret, marker='D', s=150, c='yellow', edgecolors='black', label='Current Portfolio', zorder=5)
        ax.set_title("Efficient Frontier")
