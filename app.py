import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.plotting import plot_efficient_frontier

# --- CONFIGURATION ---
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# --- HELPER FUNCTIONS ---

def optimize_portfolio(tickers, current_holdings, start_date='2020-01-01', end_date=None):
    """
    Fetches stock prices and calculates optimal portfolios.
    """
    try:
        # Download historical price data
        with st.spinner("Downloading historical stock data..."):
            # Auto-adjust=True is important for splits/dividends
            df = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
            
            # Handle MultiIndex if necessary (yfinance update)
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    df = df['Close']
                except KeyError:
                    # Fallback if 'Close' isn't top level
                    df = df.xs('Close', level=0, axis=1)

        if df.empty:
            st.error("❌ No data downloaded. Please check your ticker symbols.")
            return None, None, None

        # Remove stocks with missing data (NaNs)
        df = df.dropna(axis=1, how='all') # Drop cols that are ALL NaN
        df = df.dropna() # Drop rows with any NaN (clean timeline)
        
        if df.shape[1] < 2:
            st.error("⚠️ Not enough valid stocks to form a portfolio. At least two are required.")
            return None, None, None

        # --- Calculate current portfolio value and weights ---
        latest_prices = df.iloc[-1]
        
        # Filter holdings to only include what we successfully downloaded
        valid_tickers_held = [t for t in tickers if t in current_holdings and t in latest_prices.index]

        if not valid_tickers_held:
            st.error("❌ None of your tickers could be found in the data.")
            return None, None, None

        total_portfolio_value = sum(current_holdings[ticker] * latest_prices[ticker] for ticker in valid_tickers_held)
        
        current_weights = {
            ticker: (current_holdings[ticker] * latest_prices[ticker]) / total_portfolio_value
            for ticker in valid_tickers_held
        }

        # --- Core Calculations ---
        mu = expected_returns.mean_historical_return(df)
        S = risk_models.sample_cov(df)

        portfolios = {}

        # 1. Lowest Risk (Minimum Volatility)
        ef_low = EfficientFrontier(mu, S)
        ef_low.min_volatility()
        portfolios['low_risk'] = {'weights': ef_low.clean_weights(), 'performance': ef_low.portfolio_performance()}

        # 2. Medium Risk (Maximum Sharpe Ratio)
        ef_med = EfficientFrontier(mu, S)
        ef_med.max_sharpe()
        portfolios['medium_risk'] = {'weights': ef_med.clean_weights(), 'performance': ef_med.portfolio_performance()}

        # 3. High Risk (Target High Return)
        min_ret, _, _ = portfolios['low_risk']['performance']
        max_ret = mu.max()
        target_return = min_ret + 0.8 * (max_ret - min_ret)
        
        ef_high = EfficientFrontier(mu, S)
        try:
            ef_high.efficient_return(target_return)
            portfolios['high_risk'] = {'weights': ef_high.clean_weights(), 'performance': ef_high.portfolio_performance()}
        except:
            # Fallback if aggressive target fails
            portfolios['high_risk'] = portfolios['medium_risk']

        # --- Plotting ---
        # We calculate current portfolio stats for the plot
        current_weights_series = pd.Series(current_weights).reindex(df.columns, fill_value=0)
        curr_ret = np.sum(mu * current_weights_series)
        curr_std = np.sqrt(np.dot(current_weights_series.T, np.dot(S, current_weights_series)))

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_efficient_frontier(EfficientFrontier(mu, S), ax=ax, show_assets=False)
        
        # Add markers
        scenarios = [
            ('low_risk', 'Lowest Risk', 'green', 'o'),
            ('medium_risk', 'Max Sharpe', 'blue', '*'),
            ('high_risk', 'High Risk', 'red', 'X')
        ]
        for key, label, color, marker in scenarios:
            r, v, _ = portfolios[key]['performance']
            ax.scatter(v, r, marker=marker, s=150, c=color, label=label, zorder=5)
            
        # Add Current Portfolio Marker
        ax.scatter(curr_std, curr_ret, marker='D', s=150, c='yellow', edgecolors='black', label='Current Portfolio', zorder=5)
        
        ax.set_title("Efficient Frontier")
        ax.legend()
        plt.tight_layout()

        return portfolios, total_portfolio_value, latest_prices, fig

    except Exception as e:
        st.error(f"An error occurred during optimization: {e}")
        return None, None, None, None

def calculate_rebalancing_plan(weights, latest_prices, current_holdings, total_value, expected_return):
    """Generates the Buy/Sell table."""
    rebal_data = []
    for ticker in latest_prices.index:
        optimal_weight = weights.get(ticker, 0)
        current_shares = current_holdings.get(ticker, 0)
        target_value = total_value * optimal_weight
        current_price = latest_prices.get(ticker, 0)
        
        if current_price > 0:
            target_shares = target_value / current_price
            shares_to_trade = target_shares - current_shares
            price_target = current_price * (1 + expected_return)
            
            if abs(shares_to_trade) > 0.01:
                rebal_data.append({
                    "Ticker": ticker,
                    "Action": "Buy" if shares_to_trade > 0 else "Sell",
                    "Shares to Trade": float(f"{abs(shares_to_trade):.2f}"),
                    "Target Price (1Y)": float(f"{price_target:.2f}")
                })
    return rebal_data

def display_portfolio_results(tab, name, perf, weights, rebalancing_data):
    """Renders the results in a specific Streamlit tab."""
    with tab:
        st.subheader(f"{name} Strategy")
        c1, c2, c3 = st.columns(3)
        c1.metric("Expected Return", f"{perf[0]*100:.2f}%")
        c2.metric("Volatility (Risk)", f"{perf[1]*100:.2f}%")
        c3.metric("Sharpe Ratio", f"{perf[2]:.2f}")

        st.markdown("### 📊 Recommended Allocation")
        active_weights = {k: v*100 for k, v in weights.items() if v > 0.001}
        st.bar_chart(pd.DataFrame.from_dict(active_weights, orient='index', columns=['Weight (%)']))

        st.markdown("### 🔄 Rebalancing Plan")
        if rebalancing_data:
            st.dataframe(pd.DataFrame(rebalancing_data), use_container_width=True)
        else:
            st.info("No significant rebalancing needed.")

# --- MAIN APPLICATION LOGIC ---

# 1. Initialize Session State for the Portfolio Data
if 'portfolio_data' not in st.session_state:
    # Default example data
    st.session_state.portfolio_data = pd.DataFrame([
        {"Ticker": "AAPL", "Shares": 15},
        {"Ticker": "MSFT", "Shares": 20},
        {"Ticker": "GOOGL", "Shares": 30},
        {"Ticker": "NVDA", "Shares": 48},
        {"Ticker": "SLF", "Shares": 95},
        {"Ticker": "ENB", "Shares": 47},
        {"Ticker": "AMZN", "Shares": 5}
    ])

st.title("💰 AI Portfolio Optimizer")

# 2. Create Top-Level Tabs
input_tab, results_tab = st.tabs(["✏️ Edit Portfolio", "📈 Analysis Results"])

# --- TAB 1: INPUT ---
with input_tab:
    st.markdown("### Enter your Portfolio")
    st.caption("Double-click a cell to edit. Click '+ ' below the last row to add a new stock.")
    
    # The Data Editor allowing users to add/delete rows
    edited_df = st.data_editor(
        st.session_state.portfolio_data,
        num_rows="dynamic",
        use_container_width=True
    )
    
    # Save the edited data back to session state so it persists
    st.session_state.portfolio_data = edited_df

    # Analyze Button
    if st.button("🚀 Analyze Portfolio", type="primary"):
        # Validate Inputs
        if edited_df.empty:
            st.error("Please add at least one stock.")
        else:
            # Prepare data for the optimizer
            user_tickers = [t.upper() for t in edited_df["Ticker"].tolist() if t] # Ensure uppercase, no empty strings
            user_holdings = {
                row["Ticker"].upper(): row["Shares"] 
                for _, row in edited_df.iterrows() 
                if row["Ticker"]
            }
            
            # Run Optimization
            results = optimize_portfolio(user_tickers, user_holdings, start_date='2023-01-01')
            
            # Store results in session state
            if results[0] is not None:
                st.session_state.results = results
                st.success("Optimization Complete! Switch to the 'Analysis Results' tab.")
            else:
                st.session_state.results = None

# --- TAB 2: RESULTS ---
with results_tab:
    if 'results' in st.session_state and st.session_state.results is not None:
        portfolios, total_val, latest_prices, fig = st.session_state.results
        
        st.success(f"**Current Portfolio Value:** ${total_val:,.2f}")
        
        # Display the Efficient Frontier Plot
        st.pyplot(fig)
        
        st.divider()
        
        # Create the specific result tabs
        t1, t2, t3 = st.tabs(["🛡️ Low Risk", "⚖️ Balanced", "🚀 High Risk"])
        
        scenarios = [
            (t1, 'low_risk', "Lowest Risk"),
            (t2, 'medium_risk', "Balanced"),
            (t3, 'high_risk', "High Risk")
        ]
        
        for tab, key, name in scenarios:
            p_data = portfolios[key]
            rebal_plan = calculate_rebalancing_plan(
                p_data['weights'], latest_prices, 
                {row["Ticker"].upper(): row["Shares"] for _, row in st.session_state.portfolio_data.iterrows()},
                total_val, p_data['performance'][0]
            )
            display_portfolio_results(tab, name, p_data['performance'], p_data['weights'], rebal_plan)
            
    else:
        st.info("👈 Please go to the 'Edit Portfolio' tab and click 'Analyze Portfolio' to see results.")
