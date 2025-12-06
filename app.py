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
import matplotlib.ticker as mtick # Added for % formatting
from pypfopt import CLA

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

# --- MOBILE CHART HELPER ---
def make_mobile_chart(fig, height=500, title=None):
    """
    Optimizes Plotly charts for mobile:
    - Pan instead of Zoom (prevents scroll trapping)
    - Bottom Legend (saves horizontal space)
    - Tight margins
    """
    if title: fig.update_layout(title=title)
    fig.update_layout(
        height=height,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.2, # Legend below chart
            xanchor="center", x=0.5
        ),
        dragmode='pan', # Crucial for mobile scrolling
        hovermode="x unified"
    )
    return fig

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

def optimize_portfolio(baseline_holdings, new_holdings=None, start_date='2020-01-01'):
    try:
        # 1. Combine tickers to fetch all data at once
        base_tickers = list(baseline_holdings.keys())
        new_tickers = list(new_holdings.keys()) if new_holdings else []
        all_tickers = list(set(base_tickers + new_tickers))

        with st.spinner("Calculating efficient frontiers..."):
            df = yf.download(all_tickers, start=start_date, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                try: df = df['Close']
                except KeyError: df = df.xs('Close', level=0, axis=1)

        if df.empty or df.shape[1] < 2: return None, None, None, None, None, None, None
        df = df.dropna(axis=1, how='all').dropna()
        latest_prices = df.iloc[-1]

        # --- BASELINE FRONTIER CALC ---
        valid_base = [t for t in base_tickers if t in df.columns]
        df_base = df[valid_base]
        
        mu_b = expected_returns.mean_historical_return(df_base)
        S_b = risk_models.sample_cov(df_base)
        
        cla_b = CLA(mu_b, S_b)
        try:
            cla_b.min_volatility()
            cla_b.max_sharpe()
            ret_b, vol_b, _ = cla_b.efficient_frontier(points=50)
        except:
            ret_b, vol_b = [], []

        val_b = sum(baseline_holdings[t] * latest_prices[t] for t in valid_base)
        w_b = pd.Series({t: (baseline_holdings[t] * latest_prices[t]) / val_b for t in valid_base}).reindex(valid_base, fill_value=0)
        curr_ret_b = np.sum(mu_b * w_b)
        curr_std_b = np.sqrt(np.dot(w_b.T, np.dot(S_b, w_b)))
        curr_sharpe_b = (curr_ret_b - 0.02) / curr_std_b if curr_std_b > 0 else 0

        # --- NEW PORTFOLIO FRONTIER CALC ---
        valid_new = [t for t in new_tickers if t in df.columns]
        df_new = df[valid_new]
        
        mu_n = expected_returns.mean_historical_return(df_new)
        S_n = risk_models.sample_cov(df_new)
        
        cla_n = CLA(mu_n, S_n)
        try:
            cla_n.min_volatility()
            cla_n.max_sharpe()
            ret_n, vol_n, _ = cla_n.efficient_frontier(points=50)
        except:
            ret_n, vol_n = [], []

        val_n = sum(new_holdings[t] * latest_prices[t] for t in valid_new)
        w_n = pd.Series({t: (new_holdings[t] * latest_prices[t]) / val_n for t in valid_new}).reindex(valid_new, fill_value=0)
        curr_ret_n = np.sum(mu_n * w_n)
        curr_std_n = np.sqrt(np.dot(w_n.T, np.dot(S_n, w_n)))
        curr_sharpe_n = (curr_ret_n - 0.02) / curr_std_n if curr_std_n > 0 else 0

        # --- OPTIMIZATION STRATEGIES ---
        portfolios = {}
        ef_low = EfficientFrontier(mu_n, S_n)
        ef_low.min_volatility()
        portfolios['low_risk'] = {'weights': ef_low.clean_weights(), 'performance': ef_low.portfolio_performance()}

        ef_med = EfficientFrontier(mu_n, S_n)
        ef_med.max_sharpe()
        portfolios['medium_risk'] = {'weights': ef_med.clean_weights(), 'performance': ef_med.portfolio_performance()}

        ef_high = EfficientFrontier(mu_n, S_n)
        min_r = portfolios['low_risk']['performance'][0]
        max_r = mu_n.max()
        try:
            ef_high.efficient_return(min_r + 0.8 * (max_r - min_r))
            portfolios['high_risk'] = {'weights': ef_high.clean_weights(), 'performance': ef_high.portfolio_performance()}
        except:
            portfolios['high_risk'] = portfolios['medium_risk']

        # --- IMPROVED PLOTTING (With Plotly) ---
        fig = go.Figure()

        # [Include the Plotly code provided in the previous answer here]
        # (For brevity, I'm skipping repeating the chart code, but ensure 'fig' is created)
        
        # 1. Plot Frontiers
        if len(vol_b) > 0:
            #1. Plot Frontiers
            # Baseline Frontier (Dashed, Lighter)
            fig.add_trace(go.Scatter(
                x=vol_b, y=ret_b, 
                mode='lines', 
                name='Baseline Frontier',
                line=dict(color='#b9e713', width=2, dash='dash'),
                opacity=0.6,
                hovertemplate='<b>Base Frontier</b><br>Risk: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>'
            ))
            #fig.add_trace(go.Scatter(x=vol_b, y=ret_b, mode='lines', name='Baseline Frontier', line=dict(color='#b9e713', width=2, dash='dash'), opacity=0.6))
        if len(vol_n) > 0:
            fig.add_trace(go.Scatter(
                x=vol_n, y=ret_n, 
                mode='lines', 
                name='New Frontier',
                line=dict(color='#2980b9', width=4),
                hovertemplate='<b>New Frontier</b><br>Risk: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>'
            ))
            #fig.add_trace(go.Scatter(x=vol_n, y=ret_n, mode='lines', name='New Frontier', line=dict(color='#2980b9', width=4)))

        # 2. Plot Current Positions
        # Baseline Marker
        fig.add_trace(go.Scatter(
            x=[curr_std_b], y=[curr_ret_b],
            mode='markers+text',
            name='Baseline Hold',
            text=['Base'], textposition="bottom center",
            marker=dict(size=12, color='#bdf53b', line=dict(width=2, color='black')),
            hovertemplate='<b>Baseline Holdings</b><br>Risk: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>'
        ))

        # New Marker
        fig.add_trace(go.Scatter(
            x=[curr_std_n], y=[curr_ret_n],
            mode='markers+text',
            name='New Hold',
            text=['Current'], textposition="top center",
            marker=dict(size=15, color='#0361a0', line=dict(width=2, color='white')),
            hovertemplate='<b>New Holdings</b><br>Risk: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>'
        ))
        #fig.add_trace(go.Scatter(x=[curr_std_b], y=[curr_ret_b], mode='markers+text', name='Baseline Hold', text=['Base'], textposition="bottom center", marker=dict(size=12, color='#bdf53b', line=dict(width=2, color='black'))))
        #fig.add_trace(go.Scatter(x=[curr_std_n], y=[curr_ret_n], mode='markers+text', name='New Hold', text=['Current'], textposition="top center", marker=dict(size=15, color='#0361a0', line=dict(width=2, color='white'))))

        # 3. Plot Optimal Points
        scenarios = [
            ('low_risk', '#27ae60', 'Min Volatility', 'star'), 
            ('medium_risk', '#8e44ad', 'Max Sharpe', 'star'), 
            ('high_risk', '#c0392b', 'Max Return', 'star')
        ]
        #scenarios = [('low_risk', '#27ae60', 'Min Vol'), ('medium_risk', '#8e44ad', 'Max Sharpe'), ('high_risk', '#c0392b', 'High Ret')]

        for key, color, label, symbol in scenarios:
            r, v, _ = portfolios[key]['performance']
            fig.add_trace(go.Scatter(
                x=[v], y=[r],
                mode='markers',
                name=label,
                marker=dict(size=18, color=color, symbol=symbol, line=dict(width=1, color='white')),
                hovertemplate=f'<b>{label}</b><br>Risk: %{{x:.1%}}<br>Return: %{{y:.1%}}<extra></extra>'
            ))

        #for key, color, label in scenarios:
        #    r, v, _ = portfolios[key]['performance']
        #    fig.add_trace(go.Scatter(x=[v], y=[r], mode='markers', name=label, marker=dict(size=18, color=color, symbol='star', line=dict(width=1, color='white'))))

        fig.update_layout(
            title=dict(text="Efficient Frontier Comparison", font=dict(size=20)),
            xaxis=dict(title="Volatility (Risk)", tickformat=".0%", showgrid=True, gridcolor='#444'),
            yaxis=dict(title="Expected Return", tickformat=".0%", showgrid=True, gridcolor='#444'),
            template="plotly_dark",
            height=600,
            legend=dict(
                orientation="h", 
                yanchor="bottom", y=-0.2, 
                xanchor="center", x=0.5,
                bgcolor="rgba(0,0,0,0)"
            ),
            hovermode="closest"
        )
        #fig.update_layout(title="Efficient Frontier Comparison", height=600, template="plotly_dark", legend=dict(orientation="h", y=-0.2, x=0.5))

        # RETURN THE STATS TUPLES AS WELL
        return portfolios, val_b, val_n, latest_prices, fig, (curr_ret_b, curr_std_b, curr_sharpe_b), (curr_ret_n, curr_std_n, curr_sharpe_n)

    except Exception as e:
        st.error(f"Optimization Error: {e}")
        return None, None, None, None, None, None, None

# (Keep calculate_rebalancing_plan and display_portfolio_results exactly as they were)

def calculate_rebalancing_plan(weights, latest_prices, current_holdings, total_value, expected_return):
    """
    Calculates the TARGET portfolio structure:
    Ticker, Target Share Count, and Total Position Value.
    """
    target_data = []
    
    # Sort weights to show largest holdings first
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    
    for ticker, weight in sorted_weights:
        # Only include stocks with a significant allocation (>0.1%)
        if weight > 0.001:
            price = latest_prices.get(ticker, 0)
            
            if price > 0:
                # Calculate Target Value for this stock based on optimal weight
                position_value = total_value * weight
                
                # Calculate Number of Shares to own
                target_shares = position_value / price
                
                target_data.append({
                    "Ticker": ticker,
                    "Shares": float(f"{target_shares:.2f}"),
                    "Total Value": float(f"{position_value:.2f}")
                })
                
    return target_data

def display_portfolio_results(tab, name, perf, weights, rebal_data, total_value):
    with tab:
        st.subheader(f"{name}")
        
        # Updated to 4 Columns to include Total Value
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Return", f"{perf[0]*100:.1f}%")
        c2.metric("Risk", f"{perf[1]*100:.1f}%")
        c3.metric("Sharpe", f"{perf[2]:.2f}")
        c4.metric("Total Value", f"${total_value:,.2f}")
        
        # --- PIE CHART (Standard, No Hole) ---
        active_w = {k: v*100 for k, v in weights.items() if v > 0.001}
        
        fig = go.Figure(data=[go.Pie(
            labels=list(active_w.keys()),
            values=list(active_w.values()),
            hole=0, # 0 = Full Pie
            textinfo='label+percent',
            hoverinfo='label+percent+value',
            marker=dict(line=dict(color='#000000', width=2))
        )])
        
        fig.update_layout(
            title="Recommended Allocation",
            height=400,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # --- SIMPLIFIED TABLE ---
        if rebal_data:
            st.markdown("##### Target Portfolio Structure")
            st.dataframe(
                pd.DataFrame(rebal_data), 
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Ticker": "Ticker",
                    "Shares": st.column_config.NumberColumn("Shares", format="%.2f"),
                    "Total Value": st.column_config.NumberColumn("Total Value", format="$%.2f")
                }
            )
        else: 
            st.info("No allocation data generated.")

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
    
def get_fundamental_comparison(tickers):
    """Fetches key fundamentals (P/E, Market Cap, etc.) for a list of tickers."""
    data = []
    
    # Use yf.Tickers to optimize slightly, though .info still requires individual calls usually
    for symbol in tickers:
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            data.append({
                "Ticker": symbol,
                "Market Cap": info.get("marketCap"),
                "P/E (Trail)": info.get("trailingPE"),
                "P/E (Fwd)": info.get("forwardPE"),
                "PEG Ratio": info.get("pegRatio"),
                "Price/Book": info.get("priceToBook"),
                "ROE": info.get("returnOnEquity"),
                "Div Yield": info.get("dividendYield"),
                "Profit Margin": info.get("profitMargins")
            })
        except Exception:
            continue
            
    df = pd.DataFrame(data)
    if df.empty: return pd.DataFrame()
    
    # --- FORMATTING ---
    # 1. Market Cap (Trillions/Billions)
    def fmt_cap(x):
        if not x or pd.isna(x): return "-"
        if x >= 1e12: return f"${x/1e12:.2f}T"
        if x >= 1e9: return f"${x/1e9:.2f}B"
        return f"${x/1e6:.2f}M"
        
    df["Market Cap"] = df["Market Cap"].apply(fmt_cap)
    
    # 2. Percentages (ROE, Yield, Margins)
    for col in ["ROE", "Div Yield", "Profit Margin"]:
        df[col] = df[col].apply(lambda x: f"{x*100:.2f}%" if x and not pd.isna(x) else "-")
        
    # 3. Decimals (Ratios)
    for col in ["P/E (Trail)", "P/E (Fwd)", "PEG Ratio", "Price/Book"]:
         df[col] = df[col].apply(lambda x: f"{x:.2f}" if x and not pd.isna(x) else "-")
         
    return df.set_index("Ticker")

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
        {"Ticker": "NVDA", "Shares": 798},
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

st.title("💰 AI Financial App")

input_tab, compare_tab, deep_dive_tab = st.tabs([
    "✏️ Define & 📊 Optimize Portfolio ", 
    "🔍 Stock Comparison",
    "📊 Single Stock Deep Dive"
])

# --- TAB 1: DASHBOARD (Comparison & Optimization) ---
with input_tab:
    
    # Initialize Session States for both portfolios
    if 'portfolio_data' not in st.session_state:
        # Default Data
        data = [{"Ticker": "AAPL", "Shares": 15}, {"Ticker": "MSFT", "Shares": 20}, {"Ticker": "GOOGL", "Shares": 30}]
        st.session_state.portfolio_data = pd.DataFrame(data)
    
    if 'new_portfolio_data' not in st.session_state:
        # Default New Portfolio (Copy of baseline initially)
        st.session_state.new_portfolio_data = st.session_state.portfolio_data.copy()

    # --- HELPER TO UPDATE PRICES ---
    def update_prices_for_df(df):
        """Updates prices and totals for a given dataframe."""
        if not df.empty:
            df['Ticker'] = df['Ticker'].astype(str).str.upper()
            for index, row in df.iterrows():
                t = row['Ticker']
                p = row.get('Price', 0.0)
                if t and (pd.isna(p) or p == 0):
                    try:
                        d = yf.download(t, period="1d", progress=False)
                        if not d.empty: df.at[index, 'Price'] = float(d['Close'].iloc[-1])
                    except: pass
            
            df['Total Value'] = df['Shares'] * df['Price']
            total = df['Total Value'].sum()
            df['Weight'] = (df['Total Value'] / total * 100) if total > 0 else 0
        return df, df['Total Value'].sum()

    # --- LAYOUT: TWO COLUMNS ---
    col1, col2 = st.columns(2)

    # === LEFT: BASELINE PORTFOLIO ===
    with col1:
        st.markdown("### 1️⃣ Baseline Portfolio")
        st.caption("Your current holdings.")
        
        base_df = st.session_state.portfolio_data
        base_df, base_total = update_prices_for_df(base_df)
        st.metric("Baseline Value", f"${base_total:,.2f}")
        
        edited_base = st.data_editor(
            base_df,
            num_rows="dynamic",
            use_container_width=True,
            key="base_editor",
            column_config={
                "Ticker": st.column_config.TextColumn(required=True),
                "Shares": st.column_config.NumberColumn(min_value=0, step=1, required=True),
                "Price": st.column_config.NumberColumn(format="$%.2f", disabled=True),
                "Total Value": st.column_config.NumberColumn(format="$%.2f", disabled=True),
                "Weight": st.column_config.ProgressColumn(format="%.2f%%", min_value=0, max_value=100)
            }
        )
        
        # Save state & Reset results if changed
        if not edited_base.equals(st.session_state.portfolio_data):
            st.session_state.portfolio_data = edited_base
            st.session_state.results = None
            st.rerun()

    with col2:    
        st.markdown("### 2️⃣ New Portfolio")
        st.caption("Add stocks or change shares to compare.")
        
        new_df = st.session_state.new_portfolio_data
        new_df, new_total = update_prices_for_df(new_df)
        
        # Show difference in value
        delta_val = new_total - base_total
        # NEW (This handles the arrow automatically)
        st.metric("New Value", f"${new_total:,.2f}", delta=f"{delta_val:,.2f}")

        edited_new = st.data_editor(
            new_df,
            num_rows="dynamic",
            use_container_width=True,
            key="new_editor",
            column_config={
                "Ticker": st.column_config.TextColumn(required=True),
                "Shares": st.column_config.NumberColumn(min_value=0, step=1, required=True),
                "Price": st.column_config.NumberColumn(format="$%.2f", disabled=True),
                "Total Value": st.column_config.NumberColumn(format="$%.2f", disabled=True),
                "Weight": st.column_config.ProgressColumn(format="%.2f%%", min_value=0, max_value=100)
            }
        )

        if not edited_new.equals(st.session_state.new_portfolio_data):
            st.session_state.new_portfolio_data = edited_new
            st.session_state.results = None
            st.rerun()

    # --- ACTIONS ---
    st.divider()
    
    # Button to sync New with Baseline (Reset)
    if st.button("🔄 Reset 'New' to match 'Baseline'"):
        st.session_state.new_portfolio_data = st.session_state.portfolio_data.copy()
        st.session_state.results = None
        st.rerun()

    if st.button("🚀 Compare & Optimize Both", type="primary", use_container_width=True):
        if edited_base.empty:
            st.error("Baseline is empty.")
        else:
            # Prepare inputs
            base_tickers = [t for t in edited_base["Ticker"].tolist() if t]
            base_holdings = {row["Ticker"]: row["Shares"] for _, row in edited_base.iterrows() if row["Ticker"]}
            
            new_tickers = [t for t in edited_new["Ticker"].tolist() if t]
            new_holdings = {row["Ticker"]: row["Shares"] for _, row in edited_new.iterrows() if row["Ticker"]}

            # Run Optimization on BOTH
            res = optimize_portfolio(base_holdings, new_holdings)
            
            if res[0]: 
                st.session_state.results = res
                st.success("Comparison Complete! Scroll down.")
            else: 
                st.error("Optimization failed.")

# --- RESULTS SECTION ---
    if 'results' in st.session_state and st.session_state.results is not None:
        st.divider()
        st.subheader("📊 Optimization Results")
        
        # Unpack all variables, including the new stats tuples
        portfolios, total_val, total_val_new, latest_prices, fig, base_stats, new_stats = st.session_state.results
        
        # --- NEW: DISPLAY SIDE-BY-SIDE METRICS ---
        st.markdown("#### 🆚 Portfolio Comparison")
        
        # Create columns for the comparison
        # Layout: Label | Return | Risk | Sharpe | Total Value
        
        # Baseline Row
        st.markdown(f"**:grey[BASELINE PORTFOLIO]**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Expected Return", f"{base_stats[0]*100:.1f}%")
        c2.metric("Annual Risk", f"{base_stats[1]*100:.1f}%")
        c3.metric("Sharpe Ratio", f"{base_stats[2]:.2f}")
        c4.metric("Total Value", f"${total_val:,.2f}")

        # New Portfolio Row (with deltas where applicable)
        st.markdown(f"**:blue[NEW PORTFOLIO]**")
        d1, d2, d3, d4 = st.columns(4)
        
        # Calculate Deltas
        delta_ret = (new_stats[0] - base_stats[0]) * 100
        delta_risk = (new_stats[1] - base_stats[1]) * 100
        delta_sharpe = new_stats[2] - base_stats[2]
        delta_val = total_val_new - total_val

        d1.metric("Expected Return", f"{new_stats[0]*100:.1f}%", delta=f"{delta_ret:+.1f}%")
        d2.metric("Annual Risk", f"{new_stats[1]*100:.1f}%", delta=f"{delta_risk:+.1f}%", delta_color="inverse") # Inverse because less risk is better
        d3.metric("Sharpe Ratio", f"{new_stats[2]:.2f}", delta=f"{delta_sharpe:+.2f}")
        d4.metric("Total Value", f"${total_val_new:,.2f}", delta=f"${delta_val:,.2f}")

        st.divider()
        
        # --- CHART ---
        st.plotly_chart(fig, use_container_width=True)
        
        # --- OPTIMAL TABS ---
        t1, t2, t3 = st.tabs(["🛡️ Low Risk", "⚖️ Balanced", "🚀 High Risk"])
        scenarios = [(t1, 'low_risk', "Low Risk"), (t2, 'medium_risk', "Balanced"), (t3, 'high_risk', "High Risk")]
        
        for tab, key, name in scenarios:
            p = portfolios[key]
            # Use the NEW Portfolio total value for rebalancing target
            plan = calculate_rebalancing_plan(p['weights'], latest_prices, 
                {row["Ticker"].upper(): row["Shares"] for _, row in st.session_state.portfolio_data.iterrows()},
                total_val_new, p['performance'][0])
            display_portfolio_results(tab, name, p['performance'], p['weights'], plan, total_val_new)

# --- TAB 3: COMPARISON ---
with compare_tab:
    st.header("Compare Stock Performance")
    c1, c2 = st.columns([2, 2])
    with c1:
        default_tickers = [t for t in st.session_state.portfolio_data["Ticker"].unique() if t]
        defaults = default_tickers[:3] if default_tickers else []
        selected_tickers = st.multiselect("Select portfolio stocks:", default_tickers, default=defaults)
        extra_tickers = st.text_input("Add other tickers (comma separated):")
    with c2:
        period = st.selectbox("Time Period", ["3mo", "6mo", "1y", "2y", "5y", "max"], index=3)

    if st.button("🔎 Compare Stocks"):
        final_tickers = list(set(selected_tickers + [t.strip().upper() for t in extra_tickers.split(",") if t.strip()]))
        
        if not final_tickers:
            st.error("Select at least one ticker.")
        else:
            # 1. Fetch Technical Data
            raw_data, norm_data, tech_summary = analyze_stock_comparison(final_tickers, period)
            
            if raw_data is not None:
                # --- PLOT 1: NORMALIZED RETURNS ---
                st.subheader("Performance Chart (Normalized Returns)")
                fig_norm = go.Figure()
                for ticker in norm_data.columns:
                    fig_norm.add_trace(go.Scatter(
                        x=norm_data.index, y=norm_data[ticker], mode='lines', name=ticker,
                        hovertemplate=f'<b>{ticker}</b>: %{{y:.2f}}%<extra></extra>'
                    ))
                fig_norm.update_layout(template="plotly_dark", height=500, xaxis_title="Date", yaxis_title="Return (%)", hovermode="x unified", legend=dict(orientation="h", y=-0.2, x=0.5))
                st.plotly_chart(fig_norm, use_container_width=True)
                
                # --- PLOT 2: RAW PRICE HISTORY ---
                st.subheader("Price History (USD)")
                fig_price = go.Figure()
                for ticker in raw_data.columns:
                    fig_price.add_trace(go.Scatter(
                        x=raw_data.index, y=raw_data[ticker], mode='lines', name=ticker,
                        hovertemplate=f'<b>{ticker}</b>: $%{{y:.2f}}<extra></extra>'
                    ))
                fig_price.update_layout(template="plotly_dark", height=500, xaxis_title="Date", yaxis_title="Price ($)", hovermode="x unified", legend=dict(orientation="h", y=-0.2, x=0.5))
                st.plotly_chart(fig_price, use_container_width=True)
                
                # --- TECHNICAL SUMMARY ---
                if tech_summary:
                    st.divider()
                    st.subheader("Technical Analysis Snapshot")
                    st.dataframe(pd.DataFrame(tech_summary), use_container_width=True)

                # --- NEW: FUNDAMENTAL COMPARISON TABLE ---
                st.divider()
                st.subheader("🏗️ Fundamental Comparison")
                with st.spinner("Fetching fundamental data..."):
                    fund_df = get_fundamental_comparison(final_tickers)
                    if not fund_df.empty:
                        st.dataframe(fund_df, use_container_width=True)
                    else:
                        st.warning("Could not retrieve fundamental data.")
"""
with compare_tab:
    st.header("Compare Stock Performance")
    c1, c2 = st.columns([2, 2])
    with c1:
        default_tickers = [t for t in st.session_state.portfolio_data["Ticker"].unique() if t]
        # Handle case where default_tickers might be empty or smaller than 3
        defaults = default_tickers[:3] if default_tickers else []
        selected_tickers = st.multiselect("Select portfolio stocks:", default_tickers, default=defaults)
        extra_tickers = st.text_input("Add other tickers (comma separated):")
    with c2:
        period = st.selectbox("Time Period", ["3mo", "6mo", "1y", "2y", "5y", "max"], index=3)

    if st.button("🔎 Compare Stocks"):
        final_tickers = list(set(selected_tickers + [t.strip().upper() for t in extra_tickers.split(",") if t.strip()]))
        
        if not final_tickers:
            st.error("Select at least one ticker.")
        else:
            # 1. Fetch Data
            raw_data, norm_data, tech_summary = analyze_stock_comparison(final_tickers, period)
            
            if raw_data is not None:
                # --- PLOT 1: NORMALIZED RETURNS (%) ---
                st.subheader("Performance Chart (Normalized Returns)")
                
                fig_norm = go.Figure()
                for ticker in norm_data.columns:
                    fig_norm.add_trace(go.Scatter(
                        x=norm_data.index, 
                        y=norm_data[ticker],
                        mode='lines',
                        name=ticker,
                        hovertemplate=f'<b>{ticker}</b>: %{{y:.2f}}%<extra></extra>'
                    ))
                
                fig_norm.update_layout(
                    template="plotly_dark",
                    height=500,
                    xaxis_title="Date",
                    yaxis_title="Return (%)",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_norm, use_container_width=True)
                
                # --- PLOT 2: RAW PRICE HISTORY ($) ---
                st.subheader("Price History (USD)")
                
                fig_price = go.Figure()
                for ticker in raw_data.columns:
                    fig_price.add_trace(go.Scatter(
                        x=raw_data.index, 
                        y=raw_data[ticker],
                        mode='lines',
                        name=ticker,
                        hovertemplate=f'<b>{ticker}</b>: $%{{y:.2f}}<extra></extra>'
                    ))
                
                fig_price.update_layout(
                    template="plotly_dark",
                    height=500,
                    xaxis_title="Date",
                    yaxis_title="Stock Price ($)",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_price, use_container_width=True)
                
                # --- TECHNICAL SUMMARY ---
                if tech_summary:
                    st.divider()
                    st.subheader("Technical Analysis Snapshot")
                    st.dataframe(pd.DataFrame(tech_summary), use_container_width=True)
"""
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