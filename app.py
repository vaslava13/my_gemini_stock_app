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
        ax.legend()
        plt.tight_layout()

        return portfolios, total_portfolio_value, latest_prices, fig

    except Exception as e:
        st.error(f"An error occurred during optimization: {e}")
        return None, None, None, None

def calculate_rebalancing_plan(weights, latest_prices, current_holdings, total_value, expected_return):
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
                    "Ticker": ticker, "Action": "Buy" if shares_to_trade > 0 else "Sell",
                    "Shares to Trade": float(f"{abs(shares_to_trade):.2f}"), "Target Price (1Y)": float(f"{price_target:.2f}")
                })
    return rebal_data

def display_portfolio_results(tab, name, perf, weights, rebalancing_data):
    with tab:
        st.subheader(f"{name} Strategy")
        c1, c2, c3 = st.columns(3)
        c1.metric("Expected Return", f"{perf[0]*100:.2f}%")
        c2.metric("Volatility (Risk)", f"{perf[1]*100:.2f}%")
        c3.metric("Sharpe Ratio", f"{perf[2]:.2f}")
        active_weights = {k: v*100 for k, v in weights.items() if v > 0.001}
        st.bar_chart(pd.DataFrame.from_dict(active_weights, orient='index', columns=['Weight (%)']))
        if rebalancing_data:
            st.dataframe(pd.DataFrame(rebalancing_data), use_container_width=True)
        else:
            st.info("No significant rebalancing needed.")

# --- HELPER FUNCTIONS (COMPARISON) ---

def analyze_stock_comparison(tickers, period):
    """Downloads normalized data for comparison."""
    try:
        with st.spinner(f"Fetching data for {period}..."):
            df = yf.download(tickers, period=period, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    df_close = df['Close']
                except KeyError:
                    df_close = df.xs('Close', level=0, axis=1)
            else:
                df_close = df[['Close']] if 'Close' in df.columns else df

        if df_close.empty: return None, None
        normalized_df = (df_close / df_close.iloc[0] - 1) * 100
        return df_close, normalized_df
    except Exception as e:
        st.error(f"Error analyzing stocks: {e}")
        return None, None

# --- NEW HELPER FUNCTIONS (DEEP DIVE FINANCIALS) ---

def plot_financial_metrics(income_stmt, cash_flow, ticker_symbol):
    """Plots key financial metrics using Plotly."""
    try:
        # Prepare Data: Last 4 quarters
        income_stmt = income_stmt.iloc[:, :4].iloc[:, ::-1]
        cash_flow = cash_flow.iloc[:, :4].iloc[:, ::-1]
        dates = [d.strftime('%Y-%m-%d') for d in pd.to_datetime(income_stmt.columns)]

        # Extract data (in millions)
        revenue = income_stmt.loc['Total Revenue'] / 1e6 if 'Total Revenue' in income_stmt.index else pd.Series(0, index=dates)
        ebitda = income_stmt.loc['EBITDA'] / 1e6 if 'EBITDA' in income_stmt.index else pd.Series(0, index=dates)
        net_income = income_stmt.loc['Net Income'] / 1e6 if 'Net Income' in income_stmt.index else pd.Series(0, index=dates)
        fcf = cash_flow.loc['Free Cash Flow'] / 1e6 if 'Free Cash Flow' in cash_flow.index else pd.Series(0, index=dates)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15, 
                            subplot_titles=('Quarterly Income Metrics', 'Quarterly Free Cash Flow'))

        # Trace 1: Income Metrics
        fig.add_trace(go.Bar(x=dates, y=revenue, name='Total Revenue', text=revenue.round(0), textposition='outside'), row=1, col=1)
        fig.add_trace(go.Bar(x=dates, y=ebitda, name='EBITDA', text=ebitda.round(0), textposition='outside'), row=1, col=1)
        fig.add_trace(go.Bar(x=dates, y=net_income, name='Net Income', text=net_income.round(0), textposition='outside'), row=1, col=1)

        # Trace 2: Free Cash Flow
        fig.add_trace(go.Bar(x=dates, y=fcf, name='Free Cash Flow', text=fcf.round(0), textposition='outside', marker_color='teal'), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="grey", row=2, col=1)

        fig.update_layout(title_text=f'Quarterly Financial Metrics for {ticker_symbol}', height=700, barmode='group', template='plotly_dark', hovermode='x unified')
        return fig

    except Exception as e:
        st.warning(f"Could not plot financial metrics: {e}")
        return None

def analyze_single_stock_financials(ticker_symbol, period="2y"):
    """Downloads and displays deep-dive financials for a single stock."""
    try:
        stock = yf.Ticker(ticker_symbol)
        
        # 1. Price History & Rolling Return
        with st.spinner(f"Fetching deep dive data for {ticker_symbol}..."):
            data_period_map = {"3mo": "6mo", "6mo": "9mo", "1y": "15mo", "2y": "27mo"}
            hist = stock.history(period=data_period_map.get(period, "2y"))
            
            if hist.empty:
                st.error(f"Could not download price history for '{ticker_symbol}'.")
                return

            daily_returns = hist['Close'].pct_change()
            rolling_return = (daily_returns.rolling(window=63).sum() * 4) # Approx annualized
            hist_plot = stock.history(period=period) # Trim for plot

            # Plotly Figure for Price
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3], specs=[[{"secondary_y": True}], [{}]])
            fig.add_trace(go.Scatter(x=hist_plot.index, y=hist_plot['Close'], name='Close Price', line=dict(color='cyan')), row=1, col=1, secondary_y=False)
            fig.add_trace(go.Bar(x=hist_plot.index, y=hist_plot['Volume'], name='Volume', marker=dict(color='gray', opacity=0.3)), row=1, col=1, secondary_y=True)
            fig.add_trace(go.Scatter(x=rolling_return.index, y=rolling_return, name='Rolling Ann. Return', line=dict(color='lime')), row=2, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="grey", row=2, col=1)
            fig.update_layout(title_text=f'{ticker_symbol} Performance', height=600, template='plotly_dark', showlegend=False)
            
            st.plotly_chart(fig, use_container_width=True)

            # 2. Financial Statements
            st.subheader("📑 Financial Statements")
            
            income_stmt = stock.quarterly_income_stmt
            balance_sheet = stock.quarterly_balance_sheet
            cash_flow = stock.quarterly_cashflow

            tab_f1, tab_f2, tab_f3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
            
            with tab_f1:
                if not income_stmt.empty:
                    st.dataframe((income_stmt / 1e6).round(2).style.format("{:,.2f} M"))
                else:
                    st.info("Income statement not available.")
            with tab_f2:
                if not balance_sheet.empty:
                    st.dataframe((balance_sheet / 1e6).round(2).style.format("{:,.2f} M"))
                else:
                    st.info("Balance sheet not available.")
            with tab_f3:
                if not cash_flow.empty:
                    st.dataframe((cash_flow / 1e6).round(2).style.format("{:,.2f} M"))
                else:
                    st.info("Cash flow statement not available.")

            # 3. Financial Metrics Plot
            if not income_stmt.empty and not cash_flow.empty:
                st.subheader("📊 Key Financial Metrics")
                fig_metrics = plot_financial_metrics(income_stmt, cash_flow, ticker_symbol)
                if fig_metrics:
                    st.plotly_chart(fig_metrics, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")

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

# CREATE 4 TABS
input_tab, results_tab, compare_tab, deep_dive_tab = st.tabs([
    "✏️ Edit Portfolio", 
    "📈 Analysis Results", 
    "🔍 Stock Comparison",
    "📊 Single Stock Deep Dive"
])

# --- TAB 1: INPUT ---
with input_tab:
    st.markdown("### Enter your Portfolio")
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
        st.info("👈 Please go to the 'Edit Portfolio' tab and click 'Analyze Portfolio'.")

# --- TAB 3: COMPARISON ---
with compare_tab:
    st.header("Compare Stock Performance")
    c1, c2 = st.columns([3, 1])
    with c1:
        default_tickers = [t for t in st.session_state.portfolio_data["Ticker"].unique() if t]
        selected_tickers = st.multiselect("Select portfolio stocks:", default_tickers, default=default_tickers[:3])
        extra_tickers = st.text_input("Add other tickers (comma separated):")
    with c2:
        period = st.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=3)

    if st.button("🔎 Compare Stocks"):
        final_tickers = list(set(selected_tickers + [t.strip().upper() for t in extra_tickers.split(",") if t.strip()]))
        if not final_tickers:
            st.error("Select at least one ticker.")
        else:
            raw_data, norm_data = analyze_stock_comparison(final_tickers, period)
            if raw_data is not None:
                st.line_chart(norm_data)
                st.dataframe(pd.DataFrame({
                    "Current Price": raw_data.iloc[-1],
                    "Total Return (%)": ((raw_data.iloc[-1] / raw_data.iloc[0]) - 1) * 100
                }).style.format("{:.2f}"))

# --- TAB 4: DEEP DIVE (NEW) ---
with deep_dive_tab:
    st.header("🏢 Single Company Deep Dive")
    st.caption("Analyze Price, Volume, Income Statement, and Cash Flow for a single company.")
    
    col_d1, col_d2, col_d3 = st.columns([2, 1, 1])
    with col_d1:
        dd_ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL):", value="AAPL").upper()
    with col_d2:
        dd_period = st.selectbox("History Period", ["3mo", "6mo", "1y", "2y"], index=3, key="dd_period")
    with col_d3:
        st.write("") # Spacer
        if st.button("📊 Analyze Company"):
            analyze_single_stock_financials(dd_ticker, dd_period)
