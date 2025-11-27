import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import json
import textwrap
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
from pypfopt import EfficientFrontier, risk_models, expected_returns

# --- CONFIGURATION ---
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# --- API KEYS ---
try:
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    NEWS_API_KEY = None
    GEMINI_API_KEY = None

# --- 📱 MOBILE GRAPH HELPER ---
def make_chart_responsive(fig, height=450, title=None):
    """
    Apply mobile-friendly settings to any Plotly chart.
    """
    if title:
        fig.update_layout(title=title)
        
    fig.update_layout(
        height=height,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=50, b=10), # Small margins for phone screens
        legend=dict(
            orientation="h",        # Horizontal legend
            yanchor="bottom", y=-0.3, # Move below the chart
            xanchor="center", x=0.5
        ),
        dragmode='pan', # Default to 'pan' so mobile users can scroll sideways
        hovermode="x unified"
    )
    return fig

# --- HELPER FUNCTIONS (NEWS & AI) ---

def fetch_news_from_api(ticker_symbol, company_name):
    """Fetches news articles from NewsAPI."""
    if not NEWS_API_KEY: return None
    one_week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    search_query = f'("{ticker_symbol}" OR "{company_name}")'
    url = (f'https://newsapi.org/v2/everything?q={search_query}&from={one_week_ago}&sortBy=relevancy&language=en&apiKey={NEWS_API_KEY}')
    try:
        response = requests.get(url)
        return response.json().get('articles', []) if response.status_code == 200 else None
    except: return None

def get_gemini_analysis(ticker, news_articles):
    """Analyzes news using Gemini API."""
    if not GEMINI_API_KEY: return None
    headlines = ""
    for article in news_articles[:15]:
        headlines += f"- {article.get('title')} [URL: {article.get('url')}]\n"
    if not headlines: return None

    system_prompt = textwrap.dedent("""
        Analyze these headlines. Group into categories (Financials, Products, Sentiment, etc.).
        For each category, provide:
        - "impact": "Positive", "Negative", or "Neutral"
        - "main_message": 1-sentence summary.
        - "articles": List of title/url.
        Return ONLY JSON format.
    """).strip()

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": f"Analyze headlines for {ticker}:\n{headlines}"}]}], 
               "systemInstruction": {"parts": [{"text": system_prompt}]}, "generationConfig": {"responseMimeType": "application/json"}}
    try:
        resp = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
        return json.loads(resp.json()['candidates'][0]['content']['parts'][0]['text']) if resp.status_code == 200 else None
    except: return None

# --- HELPER FUNCTIONS (CALCULATIONS) ---

def calculate_technical_indicators(df):
    """Calculates RSI, MACD, and SMAs."""
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df

def optimize_portfolio(tickers, current_holdings, start_date='2020-01-01'):
    try:
        with st.spinner("Downloading historical stock data..."):
            df = yf.download(tickers, start=start_date, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                try: df = df['Close']
                except KeyError: df = df.xs('Close', level=0, axis=1)

        if df.empty or df.shape[1] < 2: return None, None, None, None
        df = df.dropna(axis=1, how='all').dropna()

        latest_prices = df.iloc[-1]
        valid_tickers = [t for t in tickers if t in current_holdings and t in latest_prices.index]
        if not valid_tickers: return None, None, None, None

        total_val = sum(current_holdings[t] * latest_prices[t] for t in valid_tickers)
        current_weights = {t: (current_holdings[t] * latest_prices[t]) / total_val for t in valid_tickers}

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
        ef_high = EfficientFrontier(mu, S)
        try:
            ef_high.efficient_return(min_ret + 0.8 * (mu.max() - min_ret))
            portfolios['high_risk'] = {'weights': ef_high.clean_weights(), 'performance': ef_high.portfolio_performance()}
        except:
            portfolios['high_risk'] = portfolios['medium_risk']

        # --- PLOTLY EFFICIENT FRONTIER (Mobile Replacement) ---
        # Generate random portfolios for context
        n_samples = 200
        w = np.random.dirichlet(np.ones(len(mu)), n_samples)
        rets = w.dot(mu)
        stds = np.sqrt(np.diag(w @ S @ w.T))
        sharpes = rets / stds

        fig = go.Figure()
        
        # Background dots (Possible Portfolios)
        fig.add_trace(go.Scatter(
            x=stds, y=rets, mode='markers',
            marker=dict(color=sharpes, colorscale='Viridis', showscale=False, size=4, opacity=0.4),
            name='Possible Allocation', hoverinfo='skip'
        ))

        # Optimal Markers
        scenarios = [('low_risk', 'Lowest Risk', 'green', 'circle'), ('medium_risk', 'Max Sharpe', 'blue', 'star'), ('high_risk', 'High Risk', 'red', 'x')]
        for key, label, color, symbol in scenarios:
            r, v, _ = portfolios[key]['performance']
            fig.add_trace(go.Scatter(
                x=[v], y=[r], mode='markers', 
                marker=dict(color=color, size=14, symbol=symbol, line=dict(color='white', width=1)), 
                name=label,
                hovertemplate=f"<b>{label}</b><br>Risk: %{{x:.1%}}<br>Return: %{{y:.1%}}<extra></extra>"
            ))

        # Current Portfolio Marker
        curr_series = pd.Series(current_weights).reindex(df.columns, fill_value=0)
        curr_ret = np.sum(mu * curr_series)
        curr_std = np.sqrt(np.dot(curr_series.T, np.dot(S, curr_series)))
        fig.add_trace(go.Scatter(
            x=[curr_std], y=[curr_ret], mode='markers', 
            marker=dict(color='yellow', size=14, symbol='diamond', line=dict(color='black', width=1)), 
            name='Current Portfolio',
            hovertemplate="<b>Current</b><br>Risk: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>"
        ))

        fig.update_layout(xaxis_title="Risk (Volatility)", yaxis_title="Expected Return")
        fig = make_chart_responsive(fig, title="Efficient Frontier")

        return portfolios, total_val, latest_prices, fig

    except Exception: return None, None, None, None

def calculate_rebalancing_plan(weights, latest_prices, current_holdings, total_value, expected_return):
    rebal_data = []
    for t in latest_prices.index:
        diff = (total_value * weights.get(t, 0) / latest_prices.get(t, 1)) - current_holdings.get(t, 0)
        if abs(diff) > 0.01:
            rebal_data.append({
                "Ticker": t, "Action": "Buy" if diff > 0 else "Sell",
                "Shares": float(f"{abs(diff):.2f}"), "Target Price (1Y)": float(f"{latest_prices[t]*(1+expected_return):.2f}")
            })
    return rebal_data

def display_portfolio_results(tab, name, perf, weights, rebal_data):
    with tab:
        st.subheader(f"{name}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Return", f"{perf[0]*100:.1f}%")
        c2.metric("Risk", f"{perf[1]*100:.1f}%")
        c3.metric("Sharpe", f"{perf[2]:.2f}")
        
        # Mobile Friendly Bar Chart
        active_w = {k: v*100 for k, v in weights.items() if v > 0.001}
        df_w = pd.DataFrame.from_dict(active_w, orient='index', columns=['Weight'])
        
        fig = go.Figure(go.Bar(
            x=df_w.index, y=df_w['Weight'],
            marker_color='#4CAF50',
            text=df_w['Weight'].apply(lambda x: f"{x:.1f}%"),
            textposition='auto'
        ))
        fig.update_layout(xaxis_title="Ticker", yaxis_title="Allocation (%)")
        st.plotly_chart(make_chart_responsive(fig, height=300, title="Allocation"), use_container_width=True)
        
        if rebal_data: st.dataframe(pd.DataFrame(rebal_data), use_container_width=True)
        else: st.info("No rebalancing needed.")

# --- HELPER FUNCTIONS (COMPARISON) ---

def analyze_stock_comparison(tickers, period):
    try:
        with st.spinner(f"Fetching data..."):
            df = yf.download(tickers, period=period, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                try: df_close = df['Close']
                except KeyError: df_close = df.xs('Close', level=0, axis=1)
            else: df_close = df[['Close']] if 'Close' in df.columns else df

        if df_close.empty: return None, None, None
        
        # Normalize Data
        norm_df = (df_close / df_close.iloc[0] - 1) * 100
        
        # Mobile Friendly Line Chart
        fig = go.Figure()
        for col in norm_df.columns:
            fig.add_trace(go.Scatter(x=norm_df.index, y=norm_df[col], name=col))
        
        fig.update_layout(yaxis_title="% Return")
        fig = make_chart_responsive(fig, title=f"Performance ({period})")

        # Technical Snapshot
        tech_summary = []
        for t in tickers:
            try:
                s = df_close[t].dropna() if isinstance(df_close, pd.DataFrame) else df_close.dropna()
                if len(s) > 50:
                    delta = s.diff()
                    rsi = 100 - (100/(1+(delta.where(delta>0,0).rolling(14).mean()/(-delta.where(delta<0,0).rolling(14).mean())))).iloc[-1]
                    tech_summary.append({"Ticker": t, "Price": s.iloc[-1], "RSI": f"{rsi:.1f}"})
            except: continue

        return df_close, fig, tech_summary
    except: return None, None, None

def analyze_single_stock_financials(ticker_symbol, period="2y"):
    try:
        stock = yf.Ticker(ticker_symbol)
        with st.spinner(f"Analyzing {ticker_symbol}..."):
            hist = stock.history(period="2y")
            if hist.empty: 
                st.error("No data found.")
                return

            hist = calculate_technical_indicators(hist)
            start_idx = -63 if period == "3mo" else -252 if period == "1y" else -504
            plot_df = hist.iloc[start_idx:]

            # 1. Price Chart 

[Image of line chart comparing stock performance]

            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Close'], name='Price', line=dict(color='cyan')))
            fig_price.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA_50'], name='SMA 50', line=dict(color='orange', width=1)))
            st.plotly_chart(make_chart_responsive(fig_price, title=f"{ticker_symbol} Price"), use_container_width=True)

            # 2. Fundamentals
            info = stock.info
            c1, c2, c3 = st.columns(3)
            c1.metric("P/E", info.get("trailingPE", "N/A"))
            c2.metric("Mkt Cap", f"{info.get('marketCap', 0)/1e9:.1f}B")
            c3.metric("Div Yield", f"{info.get('dividendYield', 0)*100:.2f}%")

            # 3. News
            st.divider()
            st.subheader(f"📰 AI News")
            news = fetch_news_from_api(ticker_symbol, stock.info.get('shortName', ticker_symbol))
            if news:
                analysis = get_gemini_analysis(ticker_symbol, news)
                if analysis and 'categories' in analysis:
                    for cat in analysis['categories']:
                        color = "green" if cat['impact'] == "Positive" else "red" if cat['impact'] == "Negative" else "gray"
                        with st.expander(f"{cat['name']} | :{color}[{cat['impact']}]"):
                            st.write(cat['main_message'])
                            for art in cat.get('articles', []):
                                st.caption(f"- [{art['title']}]({art['url']})")
                else: st.info("AI analysis unavailable.")
            else: st.info("No news found.")

    except Exception as e: st.error(f"Error: {e}")

# --- MAIN APP ---
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = pd.DataFrame([
        {"Ticker": "AAPL", "Shares": 15}, {"Ticker": "MSFT", "Shares": 20},
        {"Ticker": "GOOGL", "Shares": 30}, {"Ticker": "NVDA", "Shares": 48},
        {"Ticker": "SLF", "Shares": 95}, {"Ticker": "ENB", "Shares": 47},
        {"Ticker": "AMZN", "Shares": 5}
    ])

st.title("💰 AI Portfolio Optimizer")
input_tab, results_tab, compare_tab, deep_dive_tab = st.tabs(["✏️ Edit", "📈 Results", "🔍 Compare", "📊 Deep Dive"])

with input_tab:
    st.caption("Edit your portfolio:")
    edited_df = st.data_editor(st.session_state.portfolio_data, num_rows="dynamic", use_container_width=True)
    st.session_state.portfolio_data = edited_df
    if st.button("🚀 Analyze", type="primary", use_container_width=True):
        if edited_df.empty: st.error("Add stocks first.")
        else:
            ts = [t.upper() for t in edited_df["Ticker"].tolist() if t]
            hs = {row["Ticker"].upper(): row["Shares"] for _, row in edited_df.iterrows() if row["Ticker"]}
            res = optimize_portfolio(ts, hs)
            if res[0]: 
                st.session_state.results = res
                st.success("Success! Check Results tab.")
            else: st.error("Failed.")

with results_tab:
    if 'results' in st.session_state:
        portfolios, total_val, prices, fig = st.session_state.results
        st.metric("Portfolio Value", f"${total_val:,.2f}")
        st.plotly_chart(fig, use_container_width=True)
        t1, t2, t3 = st.tabs(["🛡️ Low", "⚖️ Mid", "🚀 High"])
        scenarios = [(t1, 'low_risk', "Low Risk"), (t2, 'medium_risk', "Balanced"), (t3, 'high_risk', "High Risk")]
        for tab, key, name in scenarios:
            p = portfolios[key]
            plan = calculate_rebalancing_plan(p['weights'], prices, 
                {row["Ticker"].upper(): row["Shares"] for _, row in st.session_state.portfolio_data.iterrows()},
                total_val, p['performance'][0])
            display_portfolio_results(tab, name, p['performance'], p['weights'], plan)

with compare_tab:
    c1, c2 = st.columns([2, 2])
    with c1: 
        def_t = [t for t in st.session_state.portfolio_data["Ticker"].unique() if t]
        sel_t = st.multiselect("Stocks:", def_t, default=def_t[:3])
        ext_t = st.text_input("Add others:")
    with c2: per = st.selectbox("Period", ["3mo", "6mo", "1y"], index=2)
    
    if st.button("Compare", use_container_width=True):
        final = list(set(sel_t + [t.strip().upper() for t in ext_t.split(",") if t.strip()]))
        if final:
            _, fig, _ = analyze_stock_comparison(final, per)
            if fig: st.plotly_chart(fig, use_container_width=True)

with deep_dive_tab:
    c1, c2 = st.columns([2, 2])
    with c1: tick = st.text_input("Ticker:", value="AAPL").upper()
    with c2: hist_p = st.selectbox("Range", ["3mo", "6mo", "1y", "2y"], index=2)
    if st.button("Analyze Stock", use_container_width=True):
        analyze_single_stock_financials(tick, hist_p)