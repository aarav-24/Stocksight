import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="StockSight", page_icon="◈", layout="wide")

# ══════════════════════════════════════════════════
# STYLING
# ══════════════════════════════════════════════════
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;600;700&display=swap');
.stApp { background-color: #050810; }
h1,h2,h3 { font-family:'Outfit',sans-serif !important; color:#f0f4f8 !important; }
p,li,span,label { color:#c0ccd8; }
.block-container { max-width:1100px; }
div[data-testid="stMetric"] { background:#0f1628; border:1px solid #1a2744; border-radius:10px; padding:12px 16px; }
div[data-testid="stMetric"] label { font-size:11px !important; font-weight:700; letter-spacing:1.5px; color:#4e6380 !important; font-family:'IBM Plex Mono',monospace; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-family:'IBM Plex Mono',monospace; font-weight:700; }
.section-label { font-size:10px; font-weight:700; letter-spacing:1.5px; color:#4e6380; font-family:'IBM Plex Mono',monospace; }
</style>""", unsafe_allow_html=True)

ACCENT = '#00e5a0'
BLUE = '#38bdf8'
RED = '#ff6b6b'
AMBER = '#fbbf24'
BG = '#0d1117'

plt.style.use('dark_background')
plt.rcParams.update({
    'figure.figsize':(14,7),'figure.dpi':120,'font.family':'monospace','font.size':11,
    'axes.facecolor':BG,'figure.facecolor':BG,'axes.edgecolor':'#1a2940',
    'axes.grid':True,'grid.color':'#1a2940','grid.alpha':0.5,
    'text.color':'#c8d3e0','axes.labelcolor':'#8393a7',
    'xtick.color':'#8393a7','ytick.color':'#8393a7',
})

# ══════════════════════════════════════════════════
# API KEY
# ══════════════════════════════════════════════════
try:
    FMP_KEY = st.secrets["FMP_API_KEY"]
except Exception:
    FMP_KEY = None

BASE = "https://financialmodelingprep.com/api/v3"

# ══════════════════════════════════════════════════
# API CALLER — with real error messages
# ══════════════════════════════════════════════════
def fmp_get(endpoint):
    """Call FMP API. Returns data or None. Shows error in app if it fails."""
    url = f"{BASE}/{endpoint}"
    if "apikey=" not in url:
        url += f"{'&' if '?' in url else '?'}apikey={FMP_KEY}"
    try:
        r = requests.get(url, timeout=20)
    except requests.exceptions.Timeout:
        st.error("Request timed out. Try again.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to data provider.")
        return None

    if r.status_code == 401:
        st.error("Invalid API key. Check your FMP_API_KEY in Streamlit secrets.")
        return None
    if r.status_code == 403:
        st.error("API key does not have access to this endpoint. You may need a paid FMP plan for this data.")
        return None
    if r.status_code == 429:
        st.error("Rate limited by FMP. Wait a minute and try again.")
        return None
    if r.status_code != 200:
        st.error(f"API returned status {r.status_code}: {r.text[:300]}")
        return None

    data = r.json()

    # FMP sometimes returns error messages inside a 200 response
    if isinstance(data, dict) and "Error Message" in data:
        st.error(f"FMP error: {data['Error Message']}")
        return None

    return data

# ══════════════════════════════════════════════════
# DATA PULLER — cached 1 hour
# ══════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def pull_all_data(ticker, api_key):
    """Pull all data for a ticker. Returns (data_dict, error_string)."""
    headers_check = requests.get(f"{BASE}/profile/{ticker}?apikey={api_key}", timeout=20)

    if headers_check.status_code != 200:
        return None, f"API error {headers_check.status_code}: {headers_check.text[:200]}"

    profile_data = headers_check.json()
    if not profile_data or (isinstance(profile_data, dict) and "Error Message" in profile_data):
        return None, f"No data found for {ticker}. Check the ticker symbol."

    # Income statement
    inc_resp = requests.get(f"{BASE}/income-statement/{ticker}?period=annual&limit=5&apikey={api_key}", timeout=20)
    income = inc_resp.json() if inc_resp.status_code == 200 else []
    if isinstance(income, dict):
        income = []

    if not income:
        return None, f"No financial statements found for {ticker}."

    # Balance sheet
    bal_resp = requests.get(f"{BASE}/balance-sheet-statement/{ticker}?period=annual&limit=5&apikey={api_key}", timeout=20)
    balance = bal_resp.json() if bal_resp.status_code == 200 else []
    if isinstance(balance, dict):
        balance = []

    # Cash flow
    cf_resp = requests.get(f"{BASE}/cash-flow-statement/{ticker}?period=annual&limit=5&apikey={api_key}", timeout=20)
    cashflow = cf_resp.json() if cf_resp.status_code == 200 else []
    if isinstance(cashflow, dict):
        cashflow = []

    # Stock prices (5 years daily)
    price_resp = requests.get(f"{BASE}/historical-price-full/{ticker}?timeseries=1260&apikey={api_key}", timeout=20)
    if price_resp.status_code == 200:
        price_data = price_resp.json()
        prices = price_data.get('historical', []) if isinstance(price_data, dict) else []
    else:
        prices = []

    # S&P 500 prices
    sp_resp = requests.get(f"{BASE}/historical-price-full/%5EGSPC?timeseries=1260&apikey={api_key}", timeout=20)
    if sp_resp.status_code == 200:
        sp_data = sp_resp.json()
        sp500 = sp_data.get('historical', []) if isinstance(sp_data, dict) else []
    else:
        sp500 = []

    # Risk free rate (fallback to 4.5%)
    rf = 0.045
    try:
        tr_resp = requests.get(f"{BASE}/treasury?from=2024-01-01&to=2026-12-31&apikey={api_key}", timeout=10)
        if tr_resp.status_code == 200:
            tr_data = tr_resp.json()
            if tr_data and isinstance(tr_data, list) and len(tr_data) > 0:
                rf = float(tr_data[0].get('year10', 4.5)) / 100
    except Exception:
        rf = 0.045

    return {
        'profile': profile_data[0] if isinstance(profile_data, list) else profile_data,
        'income': income,
        'balance': balance,
        'cashflow': cashflow,
        'prices': prices,
        'sp500': sp500,
        'rf_rate': rf,
    }, None

# ══════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════
def safe_div(a, b):
    return a / b if a is not None and b is not None and b != 0 else None

def fmt_pct(v):
    return f'{v*100:.1f}%' if v is not None else 'N/A'

def fmt_num(v, s=''):
    return f'{v:.2f}{s}' if v is not None else 'N/A'

def grade(val, thresholds, reverse=False):
    if val is None: return 'N/A', '#4e6380'
    a, b, c, d = thresholds
    if not reverse:
        if val >= a: return 'A', ACCENT
        if val >= b: return 'B', BLUE
        if val >= c: return 'C', AMBER
        if val >= d: return 'D', '#ff9f43'
        return 'F', RED
    else:
        if val <= a: return 'A', ACCENT
        if val <= b: return 'B', BLUE
        if val <= c: return 'C', AMBER
        if val <= d: return 'D', '#ff9f43'
        return 'F', RED

def get_field(data_list, field, idx=0):
    try:
        val = data_list[idx].get(field)
        if val is not None: return float(val)
    except Exception: pass
    return None

def to_monthly_returns(daily_prices):
    if not daily_prices: return pd.Series(dtype=float)
    df = pd.DataFrame(daily_prices)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    monthly = df['close'].resample('MS').last().dropna()
    return monthly.pct_change().dropna()

# ══════════════════════════════════════════════════
# HEADER + INPUT
# ══════════════════════════════════════════════════
st.markdown("## ◈ StockSight")
st.markdown("# Type a ticker. Get the truth.")
st.markdown("Pulls real financials, runs CML + portfolio math, gives you a straight answer.")
st.markdown("")

# Check API key FIRST
if not FMP_KEY:
    st.error("No API key found. Go to your Streamlit app settings, click Secrets, and add:\n\nFMP_API_KEY = \"your_key_from_financialmodelingprep.com\"")
    st.stop()

col_in, col_btn = st.columns([5, 1])
with col_in:
    ticker = st.text_input("Stock ticker", value="", placeholder="AAPL", label_visibility="collapsed")
with col_btn:
    run = st.button("**Analyze**", type="primary", use_container_width=True)

qcols = st.columns(8)
for i, t in enumerate(["AAPL", "TSLA", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "JPM"]):
    with qcols[i]:
        if st.button(t, key=f"q{t}", use_container_width=True):
            ticker = t
            run = True

ticker = ticker.strip().upper()

if not run or not ticker:
    st.markdown("---")
    st.markdown("### What you get")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown("**📊 Health Check**\n\n12 ratios graded A-F")
    c2.markdown("**📈 Growth**\n\n5yr CAGR & trend")
    c3.markdown("**📐 CML Position**\n\nAbove or below the line")
    c4.markdown("**⚖️ Risk Metrics**\n\nSharpe, Beta, Alpha")
    c5.markdown("**🎯 Verdict**\n\nBuy, Hold, or Avoid")
    st.stop()

# ══════════════════════════════════════════════════
# PULL & PROCESS
# ══════════════════════════════════════════════════
with st.spinner(f"Analyzing {ticker}... this takes a few seconds"):

    data, error = pull_all_data(ticker, FMP_KEY)

    if error:
        st.error(error)
        st.stop()

    if data is None:
        st.error("Something went wrong pulling data. Try again.")
        st.stop()

    p = data['profile']
    inc = data['income']
    bal = data['balance']
    cf = data['cashflow']

    company_name = p.get('companyName', ticker)
    sector = p.get('sector', 'N/A')
    price = p.get('price', 'N/A')
    mc = p.get('mktCap', 0)
    if mc and mc > 1e12: mcs = f'${mc/1e12:.2f}T'
    elif mc and mc > 1e9: mcs = f'${mc/1e9:.1f}B'
    elif mc and mc > 1e6: mcs = f'${mc/1e6:.0f}M'
    else: mcs = 'N/A'

    # ── RATIOS ──
    revenue = get_field(inc, 'revenue')
    gross_profit = get_field(inc, 'grossProfit')
    operating_income = get_field(inc, 'operatingIncome')
    net_income = get_field(inc, 'netIncome')
    interest_expense = get_field(inc, 'interestExpense')
    total_assets = get_field(bal, 'totalAssets')
    current_assets = get_field(bal, 'totalCurrentAssets')
    current_liabilities = get_field(bal, 'totalCurrentLiabilities')
    total_debt = get_field(bal, 'totalDebt')
    if total_debt is None: total_debt = get_field(bal, 'longTermDebt')
    equity = get_field(bal, 'totalStockholdersEquity')
    inventory_val = get_field(bal, 'inventory')
    fcf = get_field(cf, 'freeCashFlow')

    ratio_sections = []
    roe = safe_div(net_income, equity); roa = safe_div(net_income, total_assets)
    gross_margin = safe_div(gross_profit, revenue); operating_margin = safe_div(operating_income, revenue); net_margin = safe_div(net_income, revenue)
    ratio_sections.append(('PROFITABILITY', [
        ('Return on Equity', fmt_pct(roe), *grade(roe, (0.20, 0.12, 0.06, 0.02))),
        ('Return on Assets', fmt_pct(roa), *grade(roa, (0.10, 0.06, 0.03, 0.01))),
        ('Gross Margin', fmt_pct(gross_margin), *grade(gross_margin, (0.50, 0.35, 0.20, 0.10))),
        ('Operating Margin', fmt_pct(operating_margin), *grade(operating_margin, (0.25, 0.15, 0.08, 0.03))),
        ('Net Margin', fmt_pct(net_margin), *grade(net_margin, (0.20, 0.10, 0.05, 0.02))),
    ]))
    current_ratio = safe_div(current_assets, current_liabilities)
    quick_ratio = safe_div((current_assets - (inventory_val or 0)), current_liabilities) if current_assets and current_liabilities else None
    ratio_sections.append(('LIQUIDITY', [
        ('Current Ratio', fmt_num(current_ratio), *grade(current_ratio, (2.0, 1.5, 1.0, 0.7))),
        ('Quick Ratio', fmt_num(quick_ratio), *grade(quick_ratio, (1.5, 1.0, 0.7, 0.4))),
    ]))
    de_ratio = safe_div(total_debt, equity)
    interest_cov = safe_div(operating_income, abs(interest_expense)) if interest_expense and interest_expense != 0 else None
    debt_to_assets = safe_div(total_debt, total_assets)
    ratio_sections.append(('SOLVENCY', [
        ('Debt to Equity', fmt_num(de_ratio), *grade(de_ratio, (0.5, 1.0, 2.0, 3.0), reverse=True)),
        ('Interest Coverage', fmt_num(interest_cov, 'x'), *grade(interest_cov, (8, 5, 3, 1.5))),
        ('Debt to Assets', fmt_pct(debt_to_assets), *grade(debt_to_assets, (0.20, 0.35, 0.50, 0.65), reverse=True)),
    ]))
    asset_turnover = safe_div(revenue, total_assets); fcf_margin = safe_div(fcf, revenue)
    ratio_sections.append(('EFFICIENCY', [
        ('Asset Turnover', fmt_num(asset_turnover, 'x'), *grade(asset_turnover, (1.0, 0.7, 0.4, 0.2))),
        ('FCF Margin', fmt_pct(fcf_margin), *grade(fcf_margin, (0.20, 0.10, 0.05, 0.01))),
    ]))

    all_ratios = []
    for _, sr in ratio_sections:
        for item in sr: all_ratios.append(item)
    grade_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
    for _, _, g, _ in all_ratios:
        if g in grade_counts: grade_counts[g] += 1

    # ── GROWTH ──
    years_list, revenues_list, net_incomes_list = [], [], []
    for row in reversed(inc[:5]):
        try:
            yr = int(row.get('calendarYear', str(row.get('date', '2024'))[:4]))
            rev = float(row.get('revenue', 0))
            ni = float(row.get('netIncome', 0))
            if rev > 0:
                years_list.append(yr)
                revenues_list.append(rev / 1e9)
                net_incomes_list.append(ni / 1e9)
        except Exception: continue

    n_years = max(len(revenues_list) - 1, 1)
    revenue_cagr = (revenues_list[-1] / revenues_list[0]) ** (1 / n_years) - 1 if len(revenues_list) >= 2 and revenues_list[0] > 0 else 0
    yoy_growth = []
    for i in range(1, len(revenues_list)):
        yoy_growth.append((revenues_list[i] - revenues_list[i-1]) / revenues_list[i-1] if revenues_list[i-1] > 0 else 0)
    accelerating = len(yoy_growth) >= 2 and yoy_growth[-1] > yoy_growth[-2]

    # ── RETURNS & CML ──
    monthly_returns = to_monthly_returns(data['prices'])
    sp500_returns = to_monthly_returns(data['sp500'])
    rf_rate = data['rf_rate']
    if rf_rate is None or rf_rate <= 0 or rf_rate > 0.20: rf_rate = 0.045

    if len(monthly_returns) < 6:
        st.warning(f"Limited price history for {ticker}. Risk metrics may be less accurate.")

    annual_return = monthly_returns.mean() * 12 if len(monthly_returns) > 0 else 0
    annual_vol = monthly_returns.std() * np.sqrt(12) if len(monthly_returns) > 0 else 0.25
    market_return = sp500_returns.mean() * 12 if len(sp500_returns) > 0 else 0.10
    market_vol = sp500_returns.std() * np.sqrt(12) if len(sp500_returns) > 0 else 0.16

    cml_slope = (market_return - rf_rate) / market_vol if market_vol > 0 else 0
    cml_expected = rf_rate + cml_slope * annual_vol
    cml_distance = annual_return - cml_expected
    sharpe = (annual_return - rf_rate) / annual_vol if annual_vol > 0 else 0
    market_sharpe = (market_return - rf_rate) / market_vol if market_vol > 0 else 0

    aligned = pd.DataFrame({'stock': monthly_returns, 'market': sp500_returns}).dropna()
    if len(aligned) > 12:
        cov_matrix = np.cov(aligned['stock'], aligned['market'])
        beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 1.0
        jensens_alpha = annual_return - (rf_rate + beta * (market_return - rf_rate))
    else:
        beta = 1.0; jensens_alpha = 0

    treynor = (annual_return - rf_rate) / beta if beta != 0 else 0
    downside = monthly_returns[monthly_returns < 0]
    downside_std = downside.std() * np.sqrt(12) if len(downside) > 0 else annual_vol
    sortino = (annual_return - rf_rate) / downside_std if downside_std > 0 else 0

    if data['prices']:
        closes = pd.Series([float(d['close']) for d in sorted(data['prices'], key=lambda x: x['date'])])
        cum = closes / closes.iloc[0]
        rmx = cum.cummax()
        dd = (cum - rmx) / rmx
        max_drawdown = dd.min()
    else:
        max_drawdown = 0

    mvp_vol = market_vol * 0.6
    mvp_return = rf_rate + cml_slope * mvp_vol * 0.85
    max_vol_plot = max(annual_vol, market_vol, 0.15) * 1.5

    # ── VERDICT ──
    score = 50
    gs_map = {'A': 10, 'B': 6, 'C': 2, 'D': -3, 'F': -8}
    for _, _, g, _ in all_ratios: score += gs_map.get(g, 0)
    if cml_distance > 0.03: score += 25
    elif cml_distance > 0.01: score += 12
    elif cml_distance > -0.01: score += 3
    elif cml_distance > -0.03: score -= 10
    else: score -= 20
    if sharpe > 1.0: score += 15
    elif sharpe > 0.5: score += 7
    elif sharpe <= 0: score -= 15
    if revenue_cagr > 0.15: score += 20
    elif revenue_cagr > 0.08: score += 12
    elif revenue_cagr > 0.03: score += 5
    elif revenue_cagr <= 0: score -= 15
    if accelerating: score += 8
    score = max(0, min(100, score))
    if score >= 80: verdict, vc = 'STRONG BUY', ACCENT
    elif score >= 60: verdict, vc = 'BUY', BLUE
    elif score >= 40: verdict, vc = 'HOLD', AMBER
    elif score >= 25: verdict, vc = 'WEAK', '#ff9f43'
    else: verdict, vc = 'AVOID', RED

# ══════════════════════════════════════════════════
# DISPLAY
# ══════════════════════════════════════════════════
st.markdown("---")
st.markdown(f"## {company_name}")
st.markdown(f"**${ticker}** · {sector} · ${price} · {mcs}")

st.markdown(f"""<div style="background:{vc}10;border:1px solid {vc}30;border-radius:12px;padding:24px;margin:16px 0;position:relative;overflow:hidden;">
<div style="position:absolute;top:0;left:0;right:0;height:3px;background:{vc};opacity:0.6;"></div>
<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;">
<div>
<p class="section-label">THE VERDICT</p>
<h2 style="color:{vc} !important;font-size:28px;margin:0 0 8px;">{verdict}</h2>
<p style="font-size:13px;max-width:500px;">{"Strengths outweigh weaknesses. Math favors this one." if verdict in ('STRONG BUY','BUY') else "Roughly fairly priced for the risk." if verdict=='HOLD' else "Not compensating you enough for the risk."}</p>
</div>
<div style="text-align:center;">
<p class="section-label">SCORE</p>
<p style="font-size:40px;font-weight:800;color:{vc};font-family:'IBM Plex Mono',monospace;margin:0;">{score}</p>
</div></div></div>""", unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("CML GAP", f"{cml_distance*100:+.1f}%")
m2.metric("SHARPE", f"{sharpe:.2f}")
m3.metric("CAGR", f"{revenue_cagr*100:.1f}%")
m4.metric("RETURN", f"{annual_return*100:.1f}%")
m5.metric("VOL", f"{annual_vol*100:.1f}%")
st.markdown("---")

left, right = st.columns(2)

with left:
    st.markdown('<p class="section-label">GROWTH</p>', unsafe_allow_html=True)
    st.markdown("### Revenue & Net Income")
    if years_list and revenues_list:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        x = np.arange(len(years_list)); w = 0.35
        ax1.bar(x - w/2, revenues_list, w, color=BLUE, alpha=0.85, label='Revenue ($B)')
        ax1.bar(x + w/2, net_incomes_list, w, color=ACCENT, alpha=0.85, label='Net Income ($B)')
        max_rev = max(revenues_list) if revenues_list else 1
        for bar in ax1.patches[:len(years_list)]:
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max_rev*0.01, f'{bar.get_height():.0f}', ha='center', fontsize=9, color=BLUE)
        for bar in ax1.patches[len(years_list):]:
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max_rev*0.01, f'{bar.get_height():.0f}', ha='center', fontsize=9, color=ACCENT)
        ax1.set_xticks(x); ax1.set_xticklabels([str(y) for y in years_list])
        ax1.set_title('Revenue & Income ($B)', fontsize=12, fontweight='bold', color='white'); ax1.set_ylabel('$ Billions'); ax1.legend(fontsize=8, framealpha=0.3)
        if yoy_growth:
            gx = np.arange(len(yoy_growth)); gc = [ACCENT if g >= 0 else RED for g in yoy_growth]
            ax2.bar(gx, [g*100 for g in yoy_growth], color=gc, alpha=0.85)
            ax2.axhline(y=0, color='white', linewidth=0.5, alpha=0.3)
            ax2.set_xticks(gx); ax2.set_xticklabels([str(y) for y in years_list[1:]])
            for i, g in enumerate(yoy_growth):
                ax2.text(i, g*100+(0.3 if g>=0 else -1.2), f'{g*100:.1f}%', ha='center', fontsize=10, fontweight='bold', color=gc[i])
        ax2.set_title('YoY Growth', fontsize=12, fontweight='bold', color='white')
        plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown(f"**CAGR ({n_years}Y):** {revenue_cagr*100:.1f}% · **Trend:** {'Accelerating' if accelerating else 'Decelerating'}")

with right:
    st.markdown('<p class="section-label">CML POSITIONING</p>', unsafe_allow_html=True)
    st.markdown("### Capital Market Line")
    fig, ax = plt.subplots(figsize=(10, 8))
    cx = np.linspace(0, max_vol_plot, 100); cy = rf_rate + cml_slope * cx
    ax.plot(cx*100, cy*100, color=ACCENT, linewidth=2.5, label='CML', zorder=3)
    ax.fill_between(cx*100, cy*100, 0, alpha=0.05, color=ACCENT)
    if market_vol > mvp_vol:
        efx = np.linspace(mvp_vol, max_vol_plot*0.9, 50)
        efy = mvp_return + (market_return-mvp_return)*((efx-mvp_vol)/(market_vol-mvp_vol))**0.7
        ax.plot(efx*100, efy*100, color='#576678', linewidth=1.5, linestyle='--', alpha=0.6, label='Frontier')
    ax.scatter([0], [rf_rate*100], color=ACCENT, s=80, zorder=5)
    ax.annotate(f'Rf ({rf_rate*100:.1f}%)', (0,rf_rate*100), xytext=(10,10), textcoords='offset points', fontsize=9, color=ACCENT, fontweight='bold')
    ax.scatter([market_vol*100], [market_return*100], color=BLUE, s=140, zorder=5, edgecolors='white', linewidth=1.5)
    ax.annotate('S&P 500', (market_vol*100,market_return*100), xytext=(12,-8), textcoords='offset points', fontsize=9, color=BLUE, fontweight='bold')
    ax.scatter([mvp_vol*100], [mvp_return*100], color=AMBER, s=100, zorder=5, marker='D', edgecolors='white', linewidth=1.5)
    sc = ACCENT if cml_distance > 0 else RED
    ax.scatter([annual_vol*100], [annual_return*100], color=sc, s=220, zorder=6, edgecolors='white', linewidth=2)
    pt = f'+{cml_distance*100:.1f}%' if cml_distance > 0 else f'{cml_distance*100:.1f}%'
    ax.annotate(f'{ticker}\n{pt}', (annual_vol*100,annual_return*100), xytext=(14,6), textcoords='offset points', fontsize=10, color=sc, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=BG, edgecolor=sc, alpha=0.9))
    ax.plot([annual_vol*100]*2, [annual_return*100,cml_expected*100], color=sc, linestyle='--', linewidth=1.5, alpha=0.6)
    ax.set_xlabel('Risk (%)', fontsize=10); ax.set_ylabel('Return (%)', fontsize=10)
    ax.legend(loc='upper left', framealpha=0.3, fontsize=9)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown(f"**Return:** {annual_return*100:.1f}% · **Beta:** {beta:.2f} · **Sharpe:** {sharpe:.2f}")

st.markdown("---")

st.markdown('<p class="section-label">HEALTH CHECK</p>', unsafe_allow_html=True)
st.markdown("### 12 Ratios, Graded")
r1, r2 = st.columns(2)
for idx, (section_name, section_ratios) in enumerate(ratio_sections):
    col = r1 if idx % 2 == 0 else r2
    with col:
        st.markdown(f'<p class="section-label">{section_name}</p>', unsafe_allow_html=True)
        for name, value, g, color in section_ratios:
            st.markdown(f"""<div style="background:#0f1628;border:1px solid #1a2744;border-radius:7px;padding:10px 16px;margin-bottom:4px;display:flex;justify-content:space-between;align-items:center;">
            <span style="font-size:13px;color:#c0ccd8;">{name}</span>
            <span><span style="font-size:13px;font-weight:600;color:#f0f4f8;font-family:'IBM Plex Mono',monospace;margin-right:10px;">{value}</span>
            <span style="width:26px;height:26px;border-radius:6px;display:inline-flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;font-family:'IBM Plex Mono',monospace;color:{color};background:{color}18;">{g}</span></span>
            </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown('<p class="section-label">RISK METRICS</p>', unsafe_allow_html=True)
rm1, rm2, rm3, rm4 = st.columns(4)
rm1.metric("Treynor", f"{treynor:.2f}")
rm2.metric("Sortino", f"{sortino:.2f}")
rm3.metric("Jensen's Alpha", f"{jensens_alpha*100:+.2f}%")
rm4.metric("Max Drawdown", f"{max_drawdown*100:.1f}%")

st.markdown("---")

strengths, weaknesses = [], []
if cml_distance > 0.01: strengths.append(f'{cml_distance*100:.1f}% above CML')
elif cml_distance < -0.02: weaknesses.append(f'{abs(cml_distance)*100:.1f}% below CML')
if sharpe > market_sharpe: strengths.append(f'Sharpe ({sharpe:.2f}) beats market ({market_sharpe:.2f})')
else: weaknesses.append(f'Sharpe ({sharpe:.2f}) trails market ({market_sharpe:.2f})')
if grade_counts['A'] >= 5: strengths.append(f'{grade_counts["A"]} of 12 ratios graded A')
if revenue_cagr > 0.05: strengths.append(f'{revenue_cagr*100:.1f}% revenue CAGR')
elif revenue_cagr > 0: strengths.append(f'Modest growth ({revenue_cagr*100:.1f}%)')
else: weaknesses.append('Revenue declining')
if accelerating: strengths.append('Growth accelerating')

s1, s2 = st.columns(2)
with s1:
    st.markdown("### Strengths")
    for s in strengths: st.markdown(f"<p style='color:{ACCENT};font-size:14px;'>+ {s}</p>", unsafe_allow_html=True)
    if not strengths: st.markdown("<p style='color:#4e6380;'>None identified</p>", unsafe_allow_html=True)
with s2:
    st.markdown("### Weaknesses")
    for w in weaknesses: st.markdown(f"<p style='color:{RED};font-size:14px;'>- {w}</p>", unsafe_allow_html=True)
    if not weaknesses: st.markdown("<p style='color:#4e6380;'>None identified</p>", unsafe_allow_html=True)

if verdict in ('STRONG BUY', 'BUY'): bl = f"Math favors {ticker}."
elif verdict == 'HOLD': bl = f"{ticker} is fairly priced. No strong edge."
else: bl = f"Index fund likely beats {ticker} at this risk."

st.markdown(f"""<div style="text-align:center;padding:20px;background:{vc}0a;border:1px solid {vc}18;border-radius:10px;margin-top:16px;">
<p class="section-label">BOTTOM LINE</p>
<p style="font-size:16px;font-weight:600;color:{vc};margin:8px 0 0;">{bl}</p>
</div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown('<p style="text-align:center;font-size:11px;color:#4e6380;">STOCKSIGHT — not financial advice, just math</p>', unsafe_allow_html=True)
