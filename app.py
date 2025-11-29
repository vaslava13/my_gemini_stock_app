import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import requests
import json
import textwrap
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.plotting import plot_efficient_frontier

# --- CONFIGURATION ---
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# --- API KEYS (PASTE YOUR KEYS HERE) ---
# Securely load keys
try:
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("Secrets not found. Please add them to .streamlit/secrets.toml (local) or Streamlit Cloud Secrets.")
    st.stop()

# --- HELPER FUNCTIONS (NEWS & AI) ---

def fetch_news_from_api(ticker_symbol, company_name):
    """Fetches news articles from NewsAPI."""
    if NEWS_API_KEY == "YOUR_NEWS_API_KEY":
        return None
        
    one_week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    search_query = f'("{ticker_symbol}" OR "{company_name}")'
    
    url = (f'https://newsapi.org/v2/everything?'
           f'q={search_query}&'
           f'from={one_week_ago}&'
           f'sortBy=publishedAt&'
           f'language=en&'
           f'apiKey={NEWS_API_KEY}')
    
    try:
        response = requests.get(url)
        if response.status_code != 200: return None
        data = response.json()
        return data.get('articles', [])
    except Exception:
        return None

def get_gemini_analysis(ticker, news_articles):
    """Analyzes news using Gemini API with Aggregation."""
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        return None

    headlines = ""
    # Limit to top 20 articles to provide enough context for grouping
    for i, article in enumerate(news_articles[:20]): 
        title = article.get('title')
        url = article.get('url')
        if title and url:
            headlines += f"- {title} [URL: {url}]\n"

    if not headlines: return None

    # Updated Prompt for Categorization and Aggregation
    system_prompt = textwrap.dedent("""
        You are a senior financial analyst. Your task is to synthesize news headlines into a structured report.
        
        1. Group the provided headlines into these categories: 
           - Financial Performance
           - Products & Services
           - Partnerships & Deals
           - Legal & Regulatory
           - Market Sentiment / Analyst Ratings
           - Other
        
        2. For EACH category that contains news:
           - Determine the overall 'impact' (Positive, Negative, or Neutral).
           - Write a 'main_message': A concise paragraph summarizing the key theme across all articles in this category.
           - Create a list of 'articles' containing the 'title' and 'url' for every article used in that category.

        3. Return ONLY a JSON object with the following structure:
        {
            "categories": [
                {
                    "name": "Category Name",
                    "impact": "Positive",
                    "main_message": "Summary of the category...",
                    "articles": [
                        {"title": "Headline 1", "url": "http..."},
                        {"title": "Headline 2", "url": "http..."}
                    ]
                }
            ]
        }
    """).strip()

    user_query = f"Analyze these news headlines for {ticker}:\n\n{headlines}"
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {"responseMimeType": "application/json"}
    }

    try:
        response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
        if response.status_code != 200: return None
        result = response.json()
        text_resp = result['candidates'][0]['content']['parts'][0]['text']
        return json.loads(text_resp)
    except Exception:
        return None

# --- HELPER FUNCTIONS (CALCULATIONS) ---
def calculate_technical_indicators(df):
    """Calculates RSI, MACD, and SMAs."""
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    return df

def make_chart_responsive(fig, height=400):
    """
    Optimizes Plotly charts for mobile viewing:
    - Sets 'Pan' as default (instead of Zoom)
    - Moves legend to bottom to save vertical space
    - Reduces margins
    """
    fig.update_layout(
        height=height,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(
            orientation="h", 
            yanchor="bottom", y=-0.3, 
            xanchor="center", x=0.5
        ),
        dragmode='pan', # Crucial for mobile scrolling
        hovermode="x unified"
    )
    return fig

def optimize_portfolio(tickers, current_holdings, start_date='2020-01-01', end_date=None):
    """Fetches stock prices and calculates optimal portfolios."""
    try:
        with st.spinner("Downloading historical stock data..."):
            df = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                try: df = df['Close']
                except KeyError: df = df.xs('Close', level=0, axis=1)

        if df.empty:
            st.error("❌ No data downloaded.")
            return None, None, None, None

        df = df.dropna(axis=1, how='all').dropna() 
        if df.shape[1] < 2:
            st.error("⚠️ Need at least two valid stocks.")
            return None, None, None, None

        latest_prices = df.iloc[-1]
        valid_tickers_held = [t for t in tickers if t in current_holdings and t in latest_prices.index]
        
        if not valid_tickers_held:
            st.error("❌ None of your tickers were found.")
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
    """Downloads data for comparison and calculates Tech Indicators."""
    try:
        with st.spinner(f"Fetching data for {period}..."):
            df = yf.download(tickers, period=period, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                try: df_close = df['Close']
                except KeyError: df_close = df.xs('Close', level=0, axis=1)
            else:
                df_close = df[['Close']] if 'Close' in df.columns else df

        if df_close.empty: return None, None, None
        
        normalized_df = (df_close / df_close.iloc[0] - 1) * 100
        
        tech_summary = []
        for ticker in tickers:
            try:
                if isinstance(df_close, pd.DataFrame) and ticker in df_close.columns:
                    series = df_close[ticker].dropna()
                else:
                    series = df_close.squeeze()
                
                if len(series) > 50:
                    delta = series.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    
                    current_price = series.iloc[-1]
                    current_rsi = rsi.iloc[-1]
                    sma_50 = series.rolling(window=50).mean().iloc[-1]
                    
                    trend = "Bullish 🟢" if current_price > sma_50 else "Bearish 🔴"
                    rsi_status = "Overbought ⚠️" if current_rsi > 70 else ("Oversold 🟢" if current_rsi < 30 else "Neutral")
                    
                    tech_summary.append({
                        "Ticker": ticker,
                        "Price": current_price,
                        "Trend (vs SMA50)": trend,
                        "RSI (14)": f"{current_rsi:.1f}",
                        "RSI Status": rsi_status
                    })
            except Exception:
                continue
                
        return df_close, normalized_df, tech_summary
    except Exception as e:
        st.error(f"Error analyzing stocks: {e}")
        return None, None, None

# --- NEW HELPER FUNCTIONS (DEEP DIVE) ---
def plot_price_history(hist_df, ticker_symbol):
    """Plots Price vs Time (Simple Line Chart)."""
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['Close'], name='Stock Price', line=dict(color='cyan', width=2)))
        fig.update_layout(title_text=f"{ticker_symbol}: Price History", title_x=0.5, height=500, template="plotly_dark", hovermode="x unified", xaxis_title="Date", yaxis_title="Price ($)")
        return fig
    except Exception as e:
        st.warning(f"Could not plot price history: {e}")
        return None

def display_fundamental_metrics(stock):
    """Displays Valuation, Profitability, and Health metrics."""
    try:
        info = stock.info
        def get_metric(key, fmt="{:,.2f}", multiplier=1):
            val = info.get(key)
            if val is None: return "N/A"
            return fmt.format(val * multiplier)

        st.subheader("🏗️ Fundamental Analysis")
        
        st.markdown("**Valuation**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Market Cap", get_metric("marketCap", "{:,.0f}"))
        c2.metric("Trailing P/E", get_metric("trailingPE"))
        c3.metric("Forward P/E", get_metric("forwardPE"))
        c4.metric("Price/Book", get_metric("priceToBook"))

        st.markdown("**Profitability & Health**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Profit Margin", get_metric("profitMargins", "{:.2f}%", 100))
        c2.metric("Return on Equity", get_metric("returnOnEquity", "{:.2f}%", 100))
        c3.metric("Debt/Equity", get_metric("debtToEquity"))
        c4.metric("Current Ratio", get_metric("currentRatio"))

        st.markdown("**Dividends & Targets**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dividend Yield", get_metric("dividendYield", "{:.2f}%", 100))
        c2.metric("52W High", get_metric("fiftyTwoWeekHigh"))
        c3.metric("52W Low", get_metric("fiftyTwoWeekLow"))
        c4.metric("Analyst Target", get_metric("targetMeanPrice"))
        
        st.divider()
    except Exception as e:
        st.warning(f"Could not retrieve fundamental data: {e}")

def plot_financial_metrics(income_stmt, cash_flow, ticker_symbol):
    """Plots key financial metrics."""
    try:
        income_stmt = income_stmt.iloc[:, :4].iloc[:, ::-1]
        cash_flow = cash_flow.iloc[:, :4].iloc[:, ::-1]
        dates = [d.strftime('%Y-%m-%d') for d in pd.to_datetime(income_stmt.columns)]

        revenue = income_stmt.loc['Total Revenue'] / 1e6 if 'Total Revenue' in income_stmt.index else pd.Series(0, index=dates)
        ebitda = income_stmt.loc['EBITDA'] / 1e6 if 'EBITDA' in income_stmt.index else pd.Series(0, index=dates)
        net_income = income_stmt.loc['Net Income'] / 1e6 if 'Net Income' in income_stmt.index else pd.Series(0, index=dates)
        fcf = cash_flow.loc['Free Cash Flow'] / 1e6 if 'Free Cash Flow' in cash_flow.index else pd.Series(0, index=dates)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15, subplot_titles=('Quarterly Income Metrics', 'Quarterly Free Cash Flow'))

        fig.add_trace(go.Bar(x=dates, y=revenue, name='Total Revenue', text=revenue.round(0), textposition='outside'), row=1, col=1)
        fig.add_trace(go.Bar(x=dates, y=ebitda, name='EBITDA', text=ebitda.round(0), textposition='outside'), row=1, col=1)
        fig.add_trace(go.Bar(x=dates, y=net_income, name='Net Income', text=net_income.round(0), textposition='outside'), row=1, col=1)
        fig.add_trace(go.Bar(x=dates, y=fcf, name='Free Cash Flow', text=fcf.round(0), textposition='outside', marker_color='teal'), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="grey", row=2, col=1)

        fig.update_layout(title_text=f'Quarterly Financial Metrics for {ticker_symbol}', title_x=0.5, height=700, barmode='group', template='plotly_dark', hovermode='x unified')
        return fig
    except Exception:
        return None

def analyze_single_stock_financials(ticker_symbol, period="2y"):
    """Downloads and displays deep-dive financials + Simple Price + Complex Technicals."""
    try:
        stock = yf.Ticker(ticker_symbol)
        
        with st.spinner(f"Fetching deep dive data for {ticker_symbol}..."):
            data_period_map = {"3mo": "9mo", "6mo": "1y", "1y": "2y", "2y": "3y"}
            hist = stock.history(period=data_period_map.get(period, "2y"))
            
            if hist.empty:
                st.error(f"Could not download price history for '{ticker_symbol}'.")
                return

            hist = calculate_technical_indicators(hist)
            if period == "3mo": hist_plot = hist.iloc[-63:]
            elif period == "6mo": hist_plot = hist.iloc[-126:]
            elif period == "1y": hist_plot = hist.iloc[-252:]
            else: hist_plot = hist.iloc[-504:]

            # --- CENTERING LOGIC ---
            spacer_left, content_col, spacer_right = st.columns([1, 6, 1])
            
            with content_col:
                # 1. Simple Price Plot 
                st.subheader("📈 Price History")
                fig_price = plot_price_history(hist_plot, ticker_symbol)
                if fig_price: st.plotly_chart(fig_price, use_container_width=True)

                # 2. Complex Technical Analysis Plot
                st.subheader("📉 Technical Analysis")
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2], subplot_titles=(f'{ticker_symbol} Price & SMA', 'RSI (14)', 'MACD'))
                fig.add_trace(go.Candlestick(x=hist_plot.index, open=hist_plot['Open'], high=hist_plot['High'], low=hist_plot['Low'], close=hist_plot['Close'], name='OHLC'), row=1, col=1)
                fig.add_trace(go.Scatter(x=hist_plot.index, y=hist_plot['SMA_50'], name='SMA 50', line=dict(color='cyan', width=1)), row=1, col=1)
                fig.add_trace(go.Scatter(x=hist_plot.index, y=hist_plot['SMA_200'], name='SMA 200', line=dict(color='orange', width=1)), row=1, col=1)
                fig.add_trace(go.Scatter(x=hist_plot.index, y=hist_plot['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                fig.add_trace(go.Bar(x=hist_plot.index, y=hist_plot['MACD_Hist'], name='MACD Hist', marker_color='gray'), row=3, col=1)
                fig.add_trace(go.Scatter(x=hist_plot.index, y=hist_plot['MACD'], name='MACD', line=dict(color='blue', width=1)), row=3, col=1)
                fig.add_trace(go.Scatter(x=hist_plot.index, y=hist_plot['MACD_Signal'], name='Signal', line=dict(color='orange', width=1)), row=3, col=1)
                fig.update_layout(title_text=f'Technical Indicators: {ticker_symbol}', title_x=0.5, height=900, template='plotly_dark', showlegend=False, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

                # 3. Fundamental Analysis
                display_fundamental_metrics(stock)

                # 4. Financial Statements
                st.subheader("📑 Financial Statements")
                income_stmt = stock.quarterly_income_stmt
                balance_sheet = stock.quarterly_balance_sheet
                cash_flow = stock.quarterly_cashflow

                tab_f1, tab_f2, tab_f3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
                with tab_f1:
                    if not income_stmt.empty: st.dataframe((income_stmt / 1e6).round(2).style.format("{:,.2f} M"), use_container_width=True)
                with tab_f2:
                    if not balance_sheet.empty: st.dataframe((balance_sheet / 1e6).round(2).style.format("{:,.2f} M"), use_container_width=True)
                with tab_f3:
                    if not cash_flow.empty: st.dataframe((cash_flow / 1e6).round(2).style.format("{:,.2f} M"), use_container_width=True)

                if not income_stmt.empty and not cash_flow.empty:
                    st.subheader("📊 Key Financial Metrics")
                    fig_metrics = plot_financial_metrics(income_stmt, cash_flow, ticker_symbol)
                    if fig_metrics: st.plotly_chart(fig_metrics, use_container_width=True)

                # 5. AI News Analysis (Aggregated)
                st.divider()
                st.subheader(f"📰 AI-Powered News Analysis: {ticker_symbol}")
                st.caption("Aggregated news categories, sentiment, and summaries using Gemini 2.0.")
                
                with st.status("Fetching and analyzing news...", expanded=True) as status:
                    company_name = stock.info.get('shortName', ticker_symbol)
                    news_articles = fetch_news_from_api(ticker_symbol, company_name)
                    
                    if news_articles:
                        status.write(f"✅ Found {len(news_articles)} articles. Analyzing with Gemini...")
                        analysis = get_gemini_analysis(ticker_symbol, news_articles)
                        
                        if analysis and 'categories' in analysis:
                            status.update(label="Analysis Complete!", state="complete", expanded=False)
                            
                            for cat in analysis['categories']:
                                # Define color based on impact
                                if cat['impact'] == "Positive": color = "green"
                                elif cat['impact'] == "Negative": color = "red"
                                else: color = "gray"
                                
                                with st.expander(f"{cat['name']} | :{color}[{cat['impact']}]"):
                                    st.markdown(f"**Summary:** {cat['main_message']}")
                                    st.markdown("---")
                                    st.markdown("**Sources:**")
                                    for art in cat.get('articles', []):
                                        st.markdown(f"- [{art['title']}]({art['url']})")
                        else:
                            status.update(label="AI Analysis Failed", state="error")
                            st.error("Could not generate analysis.")
                    else:
                        status.update(label="No News Found", state="error")
                        st.warning("No recent news articles found for this stock.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# --- MAIN APPLICATION LOGIC ---
# --- INITIALIZE PORTFOLIO WITH LIVE VALUES ---
if 'portfolio_data' not in st.session_state:
    # 1. Define the initial holdings
    initial_holdings = [
        {"Ticker": "AAPL", "Shares": 15},
        {"Ticker": "MSFT", "Shares": 20},
        {"Ticker": "GOOGL", "Shares": 30},
        {"Ticker": "NVDA", "Shares": 48},
        {"Ticker": "SLF", "Shares": 95},
        {"Ticker": "ENB", "Shares": 47},
        {"Ticker": "AMZN", "Shares": 5}
    ]
    
    df = pd.DataFrame(initial_holdings)
    
    # 2. Fetch live closing prices
    tickers_list = df['Ticker'].tolist()
    try:
        market_data = yf.download(tickers_list, period="1d", progress=False)['Close'].iloc[-1]
        
        # 3. Map prices and calculate values
        df['Price'] = df['Ticker'].apply(lambda t: market_data.get(t, 0.0))
        df['Total Value'] = df['Shares'] * df['Price']
        
        # 4. Calculate Weights
        grand_total = df['Total Value'].sum()
        df['Weight'] = df['Total Value'].apply(lambda x: (x / grand_total * 100) if grand_total > 0 else 0)
        
    except Exception as e:
        df['Price'] = 0.0
        df['Total Value'] = 0.0
        df['Weight'] = 0.0
        print(f"Error fetching initial prices: {e}")

    # 5. Save to session state
    st.session_state.portfolio_data = df

st.title("💰 AI FINANCIAL TOOL & PORTFOLIO OPTIMIZER")

input_tab, compare_tab, deep_dive_tab = st.tabs([
    "✏️ Define & 📊 Optimize Portfolio ", 
    "🔍 Stock Comparison",
    "📊 Single Stock Deep Dive"
])

# --- TAB 1: DASHBOARD (Merged Input + Results) ---
with input_tab:
    st.markdown("### Enter your Portfolio")
    
    # 1. Update Calcs & Display Total at Top
    curr_df = st.session_state.portfolio_data
    if not curr_df.empty:
        curr_df['Total Value'] = curr_df['Shares'] * curr_df['Price']
        grand_total = curr_df['Total Value'].sum()
        
        # Avoid division by zero for weights
        if grand_total > 0:
            curr_df['Weight'] = (curr_df['Total Value'] / grand_total) * 100
        else:
            curr_df['Weight'] = 0.0
            
        st.metric("Total Portfolio Value", f"${grand_total:,.2f}")
    
    st.session_state.portfolio_data = curr_df

    # 2. Editor
    edited_df = st.data_editor(
        st.session_state.portfolio_data,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", required=True),
            "Shares": st.column_config.NumberColumn("Shares", min_value=0, step=1, required=True),
            "Price": st.column_config.NumberColumn("Price", format="$%.2f", disabled=True),
            "Total Value": st.column_config.NumberColumn("Total Value", format="$%.2f", disabled=True),
            "Weight": st.column_config.ProgressColumn("Weight", format="%.2f%%", min_value=0, max_value=100)
        }
    )
    
    # 3. Auto-Reset Results on Edit
    # If the user changes data, we clear the old results to force a re-run
    if not edited_df.equals(st.session_state.portfolio_data):
        st.session_state.portfolio_data = edited_df
        st.session_state.results = None 
        st.rerun()

    # 4. Analyze Button
    if st.button("🚀 Analyze Portfolio", type="primary", use_container_width=True):
        if edited_df.empty: 
            st.error("Add stocks first.")
        else:
            ts = [t.upper() for t in edited_df["Ticker"].tolist() if t]
            hs = {row["Ticker"].upper(): row["Shares"] for _, row in edited_df.iterrows() if row["Ticker"]}
            
            res = optimize_portfolio(ts, hs)
            if res[0]: 
                st.session_state.results = res
                st.success("Optimization Complete!")
            else: 
                st.error("Optimization failed.")

    # 5. Results Section (Only shows AFTER button press)
    if 'results' in st.session_state and st.session_state.results is not None:
        st.divider()
        st.subheader("📊 Optimization Results")
        
        portfolios, total_val, prices, fig = st.session_state.results
        
        # Display Old Style Matplotlib Chart (Square for Mobile)
        st.pyplot(fig, use_container_width=True)
        
        t1, t2, t3 = st.tabs(["🛡️ Low Risk", "⚖️ Balanced", "🚀 High Risk"])
        scenarios = [
            (t1, 'low_risk', "Low Risk"), 
            (t2, 'medium_risk', "Balanced"), 
            (t3, 'high_risk', "High Risk")
        ]
        
        for tab, key, name in scenarios:
            p = portfolios[key]
            plan = calculate_rebalancing_plan(p['weights'], prices, 
                {row["Ticker"].upper(): row["Shares"] for _, row in st.session_state.portfolio_data.iterrows()},
                total_val, p['performance'][0])
            display_portfolio_results(tab, name, p['performance'], p['weights'], plan)

# --- TAB 3: COMPARISON ---
with compare_tab:
    st.header("Compare Stock Performance")
    c1, c2 = st.columns([2, 2])
    with c1:
        default_tickers = [t for t in st.session_state.portfolio_data["Ticker"].unique() if t]
        selected_tickers = st.multiselect("Select portfolio stocks:", default_tickers, default=default_tickers[:3])
        extra_tickers = st.text_input("Add other tickers (comma separated):")
    with c2:
        period = st.selectbox("Time Period", ["3mo", "6mo", "1y", "2y", "5y", "max"], index=3)

    if st.button("🔎 Compare Stocks"):
        final_tickers = list(set(selected_tickers + [t.strip().upper() for t in extra_tickers.split(",") if t.strip()]))
        if not final_tickers:
            st.error("Select at least one ticker.")
        else:
            raw_data, norm_data, tech_summary = analyze_stock_comparison(final_tickers, period)
            if raw_data is not None:
                st.subheader("Performance Chart (Normalized Returns)")
                st.line_chart(norm_data)
                
                st.subheader("Price History (USD)")
                st.line_chart(raw_data)
                
                if tech_summary:
                    st.subheader("Technical Analysis Snapshot")
                    st.dataframe(pd.DataFrame(tech_summary), use_container_width=True)

# --- TAB 4: DEEP DIVE ---
with deep_dive_tab:
    st.header("🏢 Single Company Deep Dive")
    st.caption("Analyze Price, Valuation (P/E), Technicals, and Financials.")
    
    col_d1, col_d2 = st.columns([2, 2])
    with col_d1:
        dd_ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL):", value="AAPL").upper()
    with col_d2:
        dd_period = st.selectbox("History Period", ["3mo", "6mo", "1y", "2y"], index=2, key="dd_period")

    st.write("") # Spacer
    if st.button("📊 Analyze Company"):
        analyze_single_stock_financials(dd_ticker, dd_period)