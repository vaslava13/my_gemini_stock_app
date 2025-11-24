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

        # Clean data
        df = df.dropna(axis=1, how='all')
        df = df.dropna() 
        
        if df.shape[1] < 2:
            st.error("⚠️ Not enough valid stocks to form a portfolio. At least two are required.")
            return None, None, None, None

        # --- Calculate current portfolio value and weights ---
        latest_prices = df.iloc[-1]
        valid_tickers_held = [t for t in tickers if t in current_holdings and t in latest_prices.index]

        if not valid_tickers_held:
            st.error("❌ None of your tickers could be found in the data.")
            return None, None, None, None

        total_portfolio_value = sum(current_holdings[ticker] * latest_prices[ticker] for ticker in valid_tickers_held)
        
        current_weights = {
            ticker: (current_holdings[ticker] * latest_prices[ticker]) / total_portfolio_value
            for ticker in valid_tickers_held
        }

        # --- Core Calculations ---
        mu = expected_returns.mean_historical_return(df)
        S = risk_models.sample_cov(df)

        portfolios = {}

        # 1. Lowest Risk
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

        # --- Plotting ---
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

# --- NEW FUNCTION FOR COMPARISON TAB ---
def analyze_stock(tickers, period):
    """
    Downloads data for one or more tickers and prepares comparison stats.
    """
    try:
        with st.spinner(f"Fetching data for {period}..."):
            df = yf.download(tickers, period=period, progress=False, auto_adjust=True)
            
            # Handle Data Format
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    df_close = df['Close']
                except KeyError:
                    # Fallback
                    df_close = df.xs('Close', level=0, axis=1)
            else:
                # Single ticker case usually returns a DataFrame with 'Close' as a column
                if 'Close' in df.columns:
                    df_close = df[['Close']]
                else:
                    # If yfinance returns just the close data directly (rare but possible versions)
                    df_close = df

        if df_close.empty:
            return None, None

        # Normalize data (Percentage Growth from start of period)
        # Formula: (Price / Start_Price - 1) * 100
        normalized_df = (df_close / df_close.iloc[0] - 1) * 100
        
        return df_close, normalized_df

    except Exception as e:
        st.error(f"Error analyzing stocks: {e}")
        return None, None

# --- MAIN APPLICATION LOGIC ---

if 'portfolio_data' not in st.session_state:
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

# 2. Create Top-Level Tabs (Added 3rd Tab here)
input_tab, results_tab, compare_tab = st.tabs(["✏️ Edit Portfolio", "📈 Analysis Results", "🔍 Stock Comparison"])

# --- TAB 1: INPUT ---
with input_tab:
    st.markdown("### Enter your Portfolio")
    st.caption("Double-click a cell to edit. Click '+ ' below the last row to add a new stock.")
    
    edited_df = st.data_editor(st.session_state.portfolio_data, num_rows="dynamic", use_container_width=True)
    st.session_state.portfolio_data = edited_df

    if st.button("🚀 Analyze Portfolio", type="primary"):
        if edited_df.empty:
            st.error("Please add at least one stock.")
        else:
            user_tickers = [t.upper() for t in edited_df["Ticker"].tolist() if t]
            user_holdings = {row["Ticker"].upper(): row["Shares"] for _, row in edited_df.iterrows() if row["Ticker"]}
            
            results = optimize_portfolio(user_tickers, user_holdings, start_date='2023-01-01')
            
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
        st.pyplot(fig)
        st.divider()
        
        t1, t2, t3 = st.tabs(["🛡️ Low Risk", "⚖️ Balanced", "🚀 High Risk"])
        scenarios = [(t1, 'low_risk', "Lowest Risk"), (t2, 'medium_risk', "Balanced"), (t3, 'high_risk', "High Risk")]
        
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

# --- TAB 3: STOCK COMPARISON (NEW) ---
with compare_tab:
    st.header("Compare Stock Performance")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Get portfolio tickers for easy selection
        default_tickers = st.session_state.portfolio_data["Ticker"].unique().tolist()
        default_tickers = [t for t in default_tickers if t] # filter empty
        
        # User input: Multiselect for portfolio stocks + Text input for extras
        selected_tickers = st.multiselect("Select stocks from your portfolio:", default_tickers, default=default_tickers[:3])
        extra_tickers = st.text_input("Add other tickers (comma separated, e.g., TSLA, BTC-USD):")
    
    with col2:
        period = st.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=3)

    if st.button("🔎 Compare Stocks"):
        # Combine lists
        final_tickers = selected_tickers.copy()
        if extra_tickers:
            extras = [t.strip().upper() for t in extra_tickers.split(",") if t.strip()]
            final_tickers.extend(extras)
        
        # Remove duplicates
        final_tickers = list(set(final_tickers))

        if not final_tickers:
            st.error("Please select or enter at least one ticker.")
        else:
            # Run the new analysis function 

[Image of line chart comparing stock performance]

            raw_data, norm_data = analyze_stock(final_tickers, period)
            
            if raw_data is not None:
                st.subheader(f"📈 Performance Comparison ({period})")
                st.caption("Chart shows percentage growth (normalized) so you can compare different stock prices fairly.")
                st.line_chart(norm_data)
                
                st.subheader("📊 Summary Statistics")
                # Calculate simple stats
                summary_data = {
                    "Current Price": raw_data.iloc[-1],
                    "Start Price": raw_data.iloc[0],
                    "Total Return (%)": ((raw_data.iloc[-1] / raw_data.iloc[0]) - 1) * 100,
                    "Volatility (Std Dev)": raw_data.pct_change().std() * np.sqrt(252) * 100 # Annualized Volatility
                }
                st.dataframe(pd.DataFrame(summary_data).style.format("{:.2f}"))
