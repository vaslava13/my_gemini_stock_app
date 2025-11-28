import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import requests
import json
import textwrap
import math
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.plotting import plot_efficient_frontier

# --- CONFIGURATION ---
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# --- API KEYS ---
try:
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    NEWS_API_KEY = None
    GEMINI_API_KEY = None

# --- HELPER FUNCTIONS (NEWS & AI) ---

def fetch_news_from_api(ticker_symbol, company_name):
    if not NEWS_API_KEY: return None
    one_week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    search_query = f'("{ticker_symbol}" OR "{company_name}")'
    url = (f'https://newsapi.org/v2/everything?q={search_query}&from={one_week_ago}&sortBy=relevancy&language=en&apiKey={NEWS_API_KEY}')
    try:
        response = requests.get(url)
        return response.json().get('articles', []) if response.status_code == 200 else None
    except: return None

def get_gemini_analysis(ticker, news_articles):
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

# --- HELPER FUNCTIONS (CALCULATIONS & VALUATION) ---

def calculate_technical_indicators(df):
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

def calculate_dcf(info):
    """
    Calculates Intrinsic Value using Discounted Cash Flow (5Y Projection).
    """
    try:
        # 1. Inputs & Assumptions
        fcf = info.get('freeCashflow') # Look for raw FCF
        if not fcf:
            # Fallback: Operating Cash Flow - CapEx (Approx)
            ocf = info.get('operatingCashflow')
            # CapEx isn't always in 'info', we stick to fcf if available or fail safely
            if not ocf: return None, "Missing Cash Flow Data"
            fcf = ocf * 0.85 # Crude approx if CapEx missing

        shares = info.get('sharesOutstanding')
        beta = info.get('beta', 1.1) # Default to 1.1 if missing
        growth_rate = info.get('earningsGrowth', 0.10) # Default expectation
        
        if not shares or shares == 0: return None, "Missing Shares Data"

        # Conservative Growth Cap (Max 20% for calculation safety)
        growth_rate = min(growth_rate, 0.20)
        
        # WACC / Discount Rate Estimation (CAPM)
        risk_free = 0.042 # 4.2% Treasury
        market_prem = 0.05 # 5% Equity Premium
        discount_rate = risk_free + (beta * market_prem)

        # 2. Project 5 Years
        future_cash_flows = []
        projected_fcf = fcf
        for i in range(1, 6):
            projected_fcf = projected_fcf * (1 + growth_rate)
            future_cash_flows.append(projected_fcf)

        # 3. Terminal Value (Perpetual Growth 2.5%)
        terminal_growth = 0.025
        terminal_val = future_cash_flows[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)

        # 4. Discount to Present
        dcf_value = 0
        for i, val in enumerate(future_cash_flows):
            dcf_value += val / ((1 + discount_rate) ** (i + 1))
        
        dcf_value += terminal_val / ((1 + discount_rate) ** 5)

        # 5. Per Share
        intrinsic_value = dcf_value / shares
        
        return intrinsic_value, {
            "Growth Rate": f"{growth_rate:.1%}",
            "Discount Rate": f"{discount_rate:.1%}",
            "Shares": f"{shares/1e9:.2f}B",
            "Starting FCF": f"${fcf/1e9:.2f}B"
        }

    except Exception as e:
        return None, str(e)

def display_valuation_models(info):
    """Displays DCF, Graham, and Lynch models."""
    st.subheader("💎 Intrinsic Value Analysis")
    
    try:
        curr_price = info.get('currentPrice', 0)
        eps = info.get('trailingEps')
        book_val = info.get('bookValue')
        growth_raw = info.get('earningsGrowth', 0)
        
        # --- 1. DCF Model ---
        dcf_val, dcf_details = calculate_dcf(info)
        
        col_main, col_detail = st.columns([1, 2])
        if dcf_val:
            delta_dcf = ((dcf_val - curr_price) / curr_price) * 100
            col_main.metric("DCF Fair Value", f"${dcf_val:.2f}", f"{delta_dcf:.1f}%")
            with col_detail.expander("See DCF Assumptions"):
                st.write(dcf_details)
                st.caption("Method: 5-Year FCF Projection + Terminal Value (2.5% growth). Discounted using CAPM.")
        else:
            col_main.info("DCF Unavailable (Missing Cash Flow data)")

        st.divider()
        
        # --- 2. Graham & Lynch ---
        c1, c2 = st.columns(2)
        
        # Graham
        if eps and book_val and eps > 0 and book_val > 0:
            graham_val = math.sqrt(22.5 * eps * book_val)
            delta_g = ((graham_val - curr_price) / curr_price) * 100
            c1.metric("Graham Number (Value)", f"${graham_val:.2f}", f"{delta_g:.1f}%")
        else: c1.caption("Graham Number N/A")

        # Lynch
        if eps and growth_raw and eps > 0:
            # Lynch Fair Value = EPS * GrowthRate (PEG=1 logic)
            # Cap growth at 25 for safety
            lynch_val = eps * min(growth_raw * 100, 25)
            delta_l = ((lynch_val - curr_price) / curr_price) * 100
            c2.metric("Lynch Fair Value (Growth)", f"${lynch_val:.2f}", f"{delta_l:.1f}%")
        else: c2.caption("Lynch Value N/A")
        
        st.divider()

    except Exception as e:
        st.warning(f"Valuation error: {e}")

# --- OPTIMIZATION ENGINE ---

def optimize_portfolio(tickers, current_holdings, start_date='2020-01-01'):
    try:
        with st.spinner("Calculating optimal portfolio..."):
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

        # --- PLOTTING (Classic Matplotlib - Mobile Optimized) ---
        plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14, 'axes.titlesize': 16, 'legend.fontsize': 12})
        # Square aspect ratio for phones
        fig, ax = plt.subplots(figsize=(8, 8)) 
        plot_efficient_frontier(EfficientFrontier(mu, S), ax=ax, show_assets=False)
        
        scenarios = [('low_risk', 'Lowest Risk', 'green', 'o'), ('medium_risk', 'Max Sharpe', 'blue', '*'), ('high_risk', 'High Risk', 'red', 'X')]
        for key, label, color, marker in scenarios:
            r, v, _ = portfolios[key]['performance']
            ax.scatter(v, r, marker=marker, s=200, c=color, label=label, zorder=5)
        
        curr_series = pd.Series(current_weights).reindex(df.columns, fill_value=0)
        curr_ret = np.sum(mu * curr_series)
        curr_std = np.sqrt(np.dot(curr_series.T, np.dot(S, curr_series)))
        ax.scatter(curr_std, curr_ret, marker='D', s=200, c='yellow', edgecolors='black', label='Current', zorder=5)

        # Legend BELOW the chart
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2)
        ax.set_title("Efficient Frontier", pad=20)
        plt.tight_layout()

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
        
        # Native Streamlit Chart (Old Style)
        active_w = {k: v*100 for k, v in weights.items() if v > 0.001}
        st.caption("Recommended Allocation (%)")
        st.bar_chart(pd.DataFrame.from_dict(active_w, orient='index', columns=['Weight']))
        
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
        norm_df = (df_close / df_close.iloc[0] - 1) * 100
        
        tech_summary = []
        for t in tickers:
            try:
                s = df_close[t].dropna() if isinstance(df_close, pd.DataFrame) else df_close.dropna()
                if len(s) > 50:
                    delta = s.diff()
                    rsi = 100 - (100/(1+(delta.where(delta>0,0).rolling(14).mean()/(-delta.where(delta<0,0).rolling(14).mean())))).iloc[-1]
                    tech_summary.append({"Ticker": t, "Price": s.iloc[-1], "RSI": f"{rsi:.1f}"})
            except: continue

        return df_close, norm_df, tech_summary
    except: return None, None, None

def display_fundamental_metrics(stock):
    try:
        info = stock.info
        def get_metric(key, fmt="{:,.2f}", multiplier=1):
            val = info.get(key)
            if val is None: return "N/A"
            return fmt.format(val * multiplier)

        st.subheader("🏗️ Fundamentals")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mkt Cap", get_metric("marketCap", "{:,.0f}"))
        c2.metric("P/E", get_metric("trailingPE"))
        c3.metric("Profit %", get_metric("profitMargins", "{:.2f}%", 100))
        c4.metric("Div %", get_metric("dividendYield", "{:.2f}%", 100))
        st.divider()
    except: st.warning("Fundamental data unavailable.")

def plot_financial_metrics(income_stmt, cash_flow, ticker_symbol):
    try:
        income_stmt = income_stmt.iloc[:, :4].iloc[:, ::-1]
        cash_flow = cash_flow.iloc[:, :4].iloc[:, ::-1]
        dates = [d.strftime('%Y-%m-%d') for d in pd.to_datetime(income_stmt.columns)]

        rev = income_stmt.loc['Total Revenue'] / 1e6 if 'Total Revenue' in income_stmt.index else pd.Series(0, index=dates)
        net = income_stmt.loc['Net Income'] / 1e6 if 'Net Income' in income_stmt.index else pd.Series(0, index=dates)
        fcf = cash_flow.loc['Free Cash Flow'] / 1e6 if 'Free Cash Flow' in cash_flow.index else pd.Series(0, index=dates)

        # Plotly for Financials (Better than Matplotlib for Bars)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15, subplot_titles=('Revenue & Net Income', 'Free Cash Flow'))
        fig.add_trace(go.Bar(x=dates, y=rev, name='Revenue'), row=1, col=1)
        fig.add_trace(go.Bar(x=dates, y=net, name='Net Income'), row=1, col=1)
        fig.add_trace(go.Bar(x=dates, y=fcf, name='Free Cash Flow', marker_color='teal'), row=2, col=1)
        
        # Mobile Config
        fig.update_layout(height=600, template="plotly_dark", margin=dict(l=10, r=10, t=50, b=10),
                          legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                          dragmode='pan', hovermode="x unified", title=f'Financials (M USD): {ticker_symbol}')
        return fig
    except: return None

def analyze_single_stock_financials(ticker_symbol, period="2y"):
    """Deep Dive with Technicals, Financials, Valuation (DCF), and News."""
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

            # 1. Price Chart (Plotly for Pan/Zoom)
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Close'], name='Price', line=dict(color='cyan')))
            fig_price.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA_50'], name='SMA 50', line=dict(color='orange')))
            fig_price.update_layout(title=f"{ticker_symbol} Price", height=400, template="plotly_dark", 
                                    margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2), dragmode='pan')
            st.plotly_chart(fig_price, use_container_width=True)

            # 2. Fundamentals & NEW DCF VALUATION
            display_fundamental_metrics(stock)
            display_valuation_models(stock.info)

            # 3. Technicals (Plotly)
            st.subheader("📉 Technical Analysis")
            fig_tech = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.05)
            fig_tech.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI'], name='RSI', line=dict(color='purple')), row=1, col=1)
            fig_tech.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
            fig_tech.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
            fig_tech.add_trace(go.Bar(x=plot_df.index, y=plot_df['MACD_Hist'], name='MACD'), row=2, col=1)
            fig_tech.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MACD'], name='MACD Line', line=dict(color='blue')), row=2, col=1)
            fig_tech.update_layout(height=500, template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10), 
                                   legend=dict(orientation="h", y=-0.2), dragmode='pan')
            st.plotly_chart(fig_tech, use_container_width=True)

            # 4. Financials
            st.subheader("📑 Quarterly Reports")
            inc = stock.quarterly_income_stmt
            bal = stock.quarterly_balance_sheet
            cf = stock.quarterly_cashflow
            
            tab_f1, tab_f2, tab_f3 = st.tabs(["Income", "Balance", "Cash Flow"])
            with tab_f1: 
                if not inc.empty: st.dataframe((inc/1e6).round(2), use_container_width=True)
            with tab_f2: 
                if not bal.empty: st.dataframe((bal/1e6).round(2), use_container_width=True)
            with tab_f3: 
                if not cf.empty: st.dataframe((cf/1e6).round(2), use_container_width=True)

            if not inc.empty and not cf.empty:
                fig_fin = plot_financial_metrics(inc, cf, ticker_symbol)
                if fig_fin: st.plotly_chart(fig_fin, use_container_width=True)

            # 5. News
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
                st.success("Success! Go to Results tab.")
            else: st.error("Optimization failed.")

with results_tab:
    if 'results' in st.session_state:
        portfolios, total_val, prices, fig = st.session_state.results
        st.metric("Portfolio Value", f"${total_val:,.2f}")
        
        # Display Classic Matplotlib Chart (Mobile Friendly)
        st.pyplot(fig, use_container_width=True)
        
        t1, t2, t3 = st.tabs(["🛡️ Low", "⚖️ Mid", "🚀 High"])
        scenarios = [(t1, 'low_risk', "Low Risk"), (t2, 'medium_risk', "Balanced"), (t3, 'high_risk', "High Risk")]
        for tab, key, name in scenarios:
            p = portfolios[key]
            plan = calculate_rebalancing_plan(p['weights'], prices, 
                {row["Ticker"].upper(): row["Shares"] for _, row in st.session_state.portfolio_data.iterrows()},
                total_val, p['performance'][0])
            display_portfolio_results(tab, name, p['performance'], p['weights'], plan)

with compare_tab:
    c1, c2 = st.columns([2, 1])
    with c1: 
        def_t = [t for t in st.session_state.portfolio_data["Ticker"].unique() if t]
        sel_t = st.multiselect("Stocks:", def_t, default=def_t[:3])
        ext_t = st.text_input("Add others:")
    with c2: per = st.selectbox("Period", ["3mo", "6mo", "1y"], index=2)
    
    if st.button("Compare", use_container_width=True):
        final = list(set(sel_t + [t.strip().upper() for t in ext_t.split(",") if t.strip()]))
        if final:
            raw_p, norm_p, _ = analyze_stock_comparison(final, per)
            if norm_p is not None:
                st.subheader("Performance (%)")
                st.line_chart(norm_p) # Native Streamlit Chart 

                st.subheader("Price ($)")
                st.line_chart(raw_p) # Native Streamlit Chart 

with deep_dive_tab:
    c1, c2 = st.columns([2, 1])
    with c1: tick = st.text_input("Ticker:", value="AAPL").upper()
    with c2: hist_p = st.selectbox("Range", ["3mo", "6mo", "1y", "2y"], index=2)
    if st.button("Analyze Stock", use_container_width=True):
        analyze_single_stock_financials(tick, hist_p)