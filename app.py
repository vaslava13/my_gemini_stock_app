import yfinance as yf
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.plotting import plot_efficient_frontier
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def optimize_portfolio(tickers, current_holdings, start_date='2020-01-01', end_date=None):
    """
    Fetches stock prices, calculates three optimal portfolios (low, medium, high risk),
    plots them on the efficient frontier along with the current portfolio,
    and provides rebalancing advice with theoretical price targets.

    Args:
        tickers (list): A list of stock tickers.
        current_holdings (dict): A dictionary of tickers and the number of shares owned.
        start_date (str): The start date for fetching historical data (YYYY-MM-DD).
        end_date (str): The end date for fetching historical data (YYYY-MM-DD).
                           If None, it will fetch data up to the latest available.

    Returns:
        tuple: A tuple containing portfolio results (dict), total portfolio value (float),
               and latest prices (pd.Series). Returns (None, None, None) if optimization fails.
    """
    try:
        # Download historical price data
        print("Downloading historical stock data...")
        df = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']

        if df.empty:
            print("No data downloaded. Please check the tickers and date range.")
            return None, None, None

        # Remove stocks with missing data for simplicity
        df = df.dropna(axis=1)
        if df.shape[1] < 2:
            print("Not enough valid stocks to form a portfolio. At least two are required.")
            return None, None, None

        print("\nSuccessfully downloaded data for:", list(df.columns))

        # --- Calculate current portfolio value and weights ---
        latest_prices = df.iloc[-1]
        valid_tickers_held = [t for t in tickers if t in current_holdings and t in latest_prices]

        if not valid_tickers_held:
            print("None of the tickers in current_holdings are in the downloaded data.")
            return None, None, None

        total_portfolio_value = sum(current_holdings[ticker] * latest_prices[ticker] for ticker in valid_tickers_held)
        print(f"\nTotal current portfolio value: ${total_portfolio_value:,.2f}")

        current_weights = {
            ticker: (current_holdings[ticker] * latest_prices[ticker]) / total_portfolio_value
            for ticker in valid_tickers_held
        }

        print("\nCurrent Portfolio Weights:")
        for ticker, weight in current_weights.items():
            print(f"{ticker}: {weight:.2%}")

        # --- Core Calculations ---
        mu = expected_returns.mean_historical_return(df)
        S = risk_models.sample_cov(df)

        # --- NEW: Calculate current portfolio performance ---
        current_weights_aligned = pd.Series(current_weights).reindex(df.columns, fill_value=0)
        current_ret = np.sum(mu * current_weights_aligned)
        current_std = np.sqrt(np.dot(current_weights_aligned.T, np.dot(S, current_weights_aligned)))
        print("\nCurrent Portfolio Performance:")
        print(f"  Expected annual return: {current_ret:.2%}")
        print(f"  Annual volatility (risk): {current_std:.2%}")
        # --- END NEW ---

        portfolios = {}

        # 1. Lowest Risk (Minimum Volatility) Portfolio
        ef_low_risk = EfficientFrontier(mu, S)
        ef_low_risk.min_volatility()
        portfolios['low_risk'] = {
            'weights': ef_low_risk.clean_weights(),
            'performance': ef_low_risk.portfolio_performance()
        }

        # 2. Medium Risk (Maximum Sharpe Ratio) Portfolio
        ef_medium_risk = EfficientFrontier(mu, S)
        ef_medium_risk.max_sharpe()
        portfolios['medium_risk'] = {
            'weights': ef_medium_risk.clean_weights(),
            'performance': ef_medium_risk.portfolio_performance()
        }

        # 3. High Risk (Targeting a high return) Portfolio
        min_ret, _, _ = portfolios['low_risk']['performance']
        max_ret = mu.max()
        target_return = min_ret + 0.8 * (max_ret - min_ret)

        ef_high_risk = EfficientFrontier(mu, S)
        ef_high_risk.efficient_return(target_return)
        portfolios['high_risk'] = {
            'weights': ef_high_risk.clean_weights(),
            'performance': ef_high_risk.portfolio_performance()
        }

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(12, 7))
        ef_plot = EfficientFrontier(mu, S)
        plot_efficient_frontier(ef_plot, ax=ax, show_assets=False)

        # Plot the three optimal portfolios
        risk_levels = ['low_risk', 'medium_risk', 'high_risk']
        labels = ['Lowest Risk', 'Max Sharpe Ratio', 'High Risk']
        colors = ['green', 'blue', 'red']
        markers = ['o', '*', 'X']

        for i, risk in enumerate(risk_levels):
            ret, std, _ = portfolios[risk]['performance']
            ax.scatter(std, ret, marker=markers[i], s=150, c=colors[i], label=labels[i], zorder=5)

        # --- NEW: Plot the current portfolio on the graph ---
        ax.scatter(current_std, current_ret, marker='D', s=150, c='yellow',
                   edgecolors='black', label='Current Portfolio', zorder=5)
        # --- END NEW ---

        ax.set_title("Efficient Frontier with Portfolio Options")
        ax.legend()
        plt.tight_layout()

        print("\nPlotting the efficient frontier. Please close the plot window to continue.")
        plt.show()

        return portfolios, total_portfolio_value, latest_prices

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None

def print_portfolio_details(name, data, total_value, latest_prices, current_holdings):
    """Helper function to print details for a single portfolio."""
    weights = data['weights']
    performance = data['performance']
    expected_return = performance[0]

    print("\n" + "="*50)
    print(f"      {name.upper()} PORTFOLIO")
    print("="*50)

    print("\n  Portfolio Performance:")
    print(f"    Expected annual return: {expected_return:.2%}")
    print(f"    Annual volatility: {performance[1]:.2%}")
    print(f"    Sharpe Ratio: {performance[2]:.2f}")

    print("\n  Optimal Portfolio Weights:")
    for ticker, weight in weights.items():
        if weight > 0:
            print(f"    {ticker}: {weight:.2%}")

    print("\n  Rebalancing and Price Target Recommendations:")
    print("  ---------------------------------------------")
    for ticker in latest_prices.index:
        optimal_weight = weights.get(ticker, 0)
        current_shares = current_holdings.get(ticker, 0)
        current_value = current_shares * latest_prices.get(ticker, 0)
        target_value = total_value * optimal_weight
        target_shares = target_value / latest_prices.get(ticker, 1)
        shares_to_trade = target_shares - current_shares
        action = "Buy" if shares_to_trade > 0 else "Sell"
        price_target = latest_prices.get(ticker, 0) * (1 + expected_return)

        print(f"\n  {ticker}:")
        print(f"    Current Shares: {current_shares:.2f} (${current_value:,.2f})")
        print(f"    Target Shares:  {target_shares:.2f} (${target_value:,.2f})")
        if abs(shares_to_trade) > 0.01:
            print(f"    Action: {action} {abs(shares_to_trade):.2f} shares")
        print(f"    Theoretical 1-Year Price Target: ${price_target:,.2f}")

# Helper function to display portfolio data nicely
def display_portfolio_results(tab, name, expected_return, volatility, sharpe, weights, rebalancing_data):
    with tab:
        st.subheader(f"{name} Strategy")
        
        # 1. Display Top-Level Metrics in Columns
        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Annual Return", f"{expected_return:.2f}%")
        col2.metric("Annual Volatility (Risk)", f"{volatility:.2f}%")
        col3.metric("Sharpe Ratio", f"{sharpe:.2f}")

        # 2. Display Weights (Pie Chart or Bar Chart)
        st.markdown("### 📊 Recommended Allocation")
        
        # Convert weights dict to DataFrame for plotting
        # Filter out 0% weights to make the chart cleaner
        active_weights = {k: v for k, v in weights.items() if v > 0.0001}
        df_weights = pd.DataFrame.from_dict(active_weights, orient='index', columns=['Weight'])
        df_weights['Weight'] = df_weights['Weight'] * 100 # Convert to percentage
        
        st.bar_chart(df_weights)

        # 3. Display Rebalancing Logic as a DataFrame (Instead of text list)
        st.markdown("### 🔄 Rebalancing Action Plan")
        
        if rebalancing_data:
            df_rebal = pd.DataFrame(rebalancing_data)
            st.dataframe(
                df_rebal, 
                use_container_width=True,
                column_config={
                    "Ticker": "Ticker",
                    "Action": "Action",
                    "Amount": "Shares to Trade",
                    "Target Price (1Y)": st.column_config.NumberColumn(format="$%.2f")
                }
            )
        else:
            st.info("No rebalancing required.")

if __name__ == '__main__':
    # --- User Input ---
    portfolio_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'SLF', 'ENB', 'BEPC', 'LAZR', 'BIPC', 'PYPL']

    current_holdings = {
        'AAPL': 15, 'MSFT': 20, 'GOOGL': 30, 'AMZN': 5, 'NVDA': 48,
        'SLF': 95, 'ENB': 47, 'LAZR': 1, 'BEPC': 15, 'BIPC': 12, 'PYPL': 5
    }
    start = '2023-01-01' # Using a longer start date for more robust historical data

    print("Starting portfolio optimization...")
    portfolios, total_value, latest_prices = optimize_portfolio(
        portfolio_tickers, current_holdings, start_date=start
    )

    if portfolios:
        print("\n\n" + "#"*60)
        print("                   PORTFOLIO OPTIMIZATION SUMMARY")
        print("#"*60)

        print_portfolio_details("Lowest Risk", portfolios['low_risk'], total_value, latest_prices, current_holdings)
        print_portfolio_details("Medium Risk (Balanced)", portfolios['medium_risk'], total_value, latest_prices, current_holdings)
        print_portfolio_details("High Risk", portfolios['high_risk'], total_value, latest_prices, current_holdings)

        print("\n\n" + "="*50)
        print("Disclaimer:")
        print("The 'Theoretical 1-Year Price Target' is not financial advice or a price prediction. It is calculated by applying the portfolio's overall expected annual return to the stock's current price. This is based on historical data and assumes past performance will continue, which is not guaranteed.")
        print("="*50)


    print("\nOptimization complete.")

    # ... (Your existing code where you calculate metrics) ...

    st.divider()
    st.header("🎯 Portfolio Optimization Results")

    # Create 3 Tabs for the different strategies
    tab1, tab2, tab3 = st.tabs(["🛡️ Low Risk", "⚖️ Balanced", "🚀 High Risk"])

    # --- EXAMPLE OF HOW TO CALL THIS FUNCTION IN YOUR LOOP ---
    
    # You likely have a loop or sequence where you calculate these. 
    # You need to gather the data into a list instead of printing it directly.
    
    # Example Data Structure construction (Adapt this inside your logic):
    # For "Low Risk":
    low_risk_rebal_list = []
    # ... inside your rebalancing calculation loop for low risk ...
    # instead of print(f"Action: Sell {shares}"), do:
    # low_risk_rebal_list.append({
    #     "Ticker": ticker,
    #     "Action": "Sell",
    #     "Amount": 6.02,
    #     "Target Price (1Y)": 325.20
    # })
    
    # Then call the display function:
    display_portfolio_results(
        tab=tab1,
        name="Minimum Volatility",
        expected_return=19.78,  # Replace with your variable: e.g., perf[0]*100
        volatility=13.38,       # Replace with your variable: e.g., perf[1]*100
        sharpe=1.48,            # Replace with your variable
        weights={'AAPL': 0.0586, 'ENB': 0.4594}, # Replace with your `cleaned_weights` dictionary
        rebalancing_data=low_risk_rebal_list
    )
    
    # Repeat for tab2 (Balanced) and tab3 (High Risk) using their specific variables.









