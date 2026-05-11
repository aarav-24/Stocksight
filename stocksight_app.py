import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="StockSight", page_icon="◈", layout="wide")

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

ACCENT = '#00e5a0'; BLUE = '#38bdf8'; RED = '#ff6b6b'; AMBER = '#fbbf24'; BG = '#0d1117'

plt.style.use('dark_background')
plt.rcParams.update({
    'figure.figsize':(14,7),'figure.dpi':120,'font.family':'monospace','font.size':11,
    'axes.facecolor':BG,'figure.facecolor':BG,'axes.edgecolor':'#1a2940',
    'axes.grid':True,'grid.color':'#1a2940','grid.alpha':0.5,
    'text.color':'#c8d3e0','axes.labelcolor':'#8393a7',
    'xtick.color':'#8393a7','ytick.color':'#8393a7',
})

# ══════════════════════════════════════════════════
# DATA PULLER — cached 1 hour, with retry + sleep
# No API key needed. yfinance is free.
# ══════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner="Pulling financial data...")
def pull_data(ticker_symbol):
    """Pull all data for one ticker. Cached so repeat clicks don't re-fetch."""
    for attempt in range(3):
        try:
            stock = yf.Ticker(ticker_symbol)
            info = stock.info
            if not info or info.get('quoteType') is None:
                return None, f"No data found for {ticker_symbol}."

            time.sleep(1)  # avoid rate limit

            income = stock.income_stmt
            balance = stock.balance_sheet
            cashflow = stock.cashflow

            if income is None or income.empty:
                return None, f"No financial statements for {ticker_symbol}."

            time.sleep(1)

            hist = stock.history(period='5y', interval='1mo')
            if hist is None or hist.empty:
                return None, f"No price history for {ticker_symbol}."

            time.sleep(1)

            sp = yf.Ticker('^GSPC')
            sp_hist = sp.history(period='5y', interval='1mo')

            rf = 0.045
            try:
                time.sleep(1)
                tnx = yf.Ticker('^TNX')
                tnx_info = tnx.info
                rf_val = tnx_info.get('regularMarketPrice', tnx_info.get('previousClose', 4.5))
                if rf_val: rf = float(rf_val) / 100
            except: pass

            return {
                'info': info,
                'income': income,
                'balance': balance,
                'cashflow': cashflow,
                'hist': hist,
                'sp_hist': sp_hist,
                'rf': rf,
            }, None

        except Exception as e:
            if attempt < 2:
                time.sleep(5)  # wait 5 sec before retry
                continue
            return None, f"Failed after 3 attempts: {str(e)[:200]}"

    return None, "Unknown error."

# ══════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════
def safe_get(df, fields, col=0):
    if isinstance(fields, str): fields = [fields]
    for field in fields:
        try:
            if field in df.index:
                loc = df.index.get_loc(field)
                if isinstance(loc, slice): val = df.iloc[loc.start, col]
                elif isinstance(loc, np.ndarray): val = df.iloc[np.where(loc)[0][0], col]
                else: val = df.iloc[loc, col]
                if pd.notna(val): return float(val)
        except: continue
    return None

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

# ══════════════════════════════════════════════════
# HEADER + INPUT
# ══════════════════════════════════════════════════
st.markdown("## ◈ StockSight")
st.markdown("# Type a ticker. Get the truth.")
st.markdown("Pulls real financials, runs CML + portfolio math, gives you a straight answer.")
st.markdown("")

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
# PULL DATA (cached — only fetches once per ticker per hour)
# ══════════════════════════════════════════════════
data, error = pull_data(ticker)

if error:
    st.error(error)
    st.stop()

info = data['info']
income_stmt = data['income']
balance_sheet = data['balance']
cash_flow = data['cashflow']
hist = data['hist']
sp_hist = data['sp_hist']
rf_rate = data['rf']
if rf_rate is None or rf_rate <= 0 or rf_rate > 0.20: rf_rate = 0.045

company_name = info.get('longName', info.get('shortName', ticker))
sector = info.get('sector', 'N/A')
price = info.get('currentPrice', info.get('regularMarketPrice', info.get('previousClose', 'N/A')))
mc = info.get('marketCap', 0)
if mc and mc > 1e12: mcs = f'${mc/1e12:.2f}T'
elif mc and mc > 1e9: mcs = f'${mc/1e9:.1f}B'
else: mcs = 'N/A'

monthly_returns = hist['Close'].pct_change().dropna()
sp500_returns = sp_hist['Close'].pct_change().dropna() if sp_hist is not None and not sp_hist.empty else pd.Series(dtype=float)

# ══════════════════════════════════════════════════
# RATIOS
# ══════════════════════════════════════════════════
revenue = safe_get(income_stmt, ['Total Revenue', 'Revenue'])
gross_profit = safe_get(income_stmt, ['Gross Profit'])
operating_income = safe_get(income_stmt, ['Operating Income', 'EBIT'])
net_income = safe_get(income_stmt, ['Net Income', 'Net Income Common Stockholders'])
interest_expense = safe_get(income_stmt, ['Interest Expense', 'Interest Expense Non Operating'])
total_assets = safe_get(balance_sheet, ['Total Assets'])
current_assets = safe_get(balance_sheet, ['Current Assets', 'Total Current Assets'])
current_liabilities = safe_get(balance_sheet, ['Current Liabilities', 'Total Current Liabilities'])
total_debt = safe_get(balance_sheet, ['Total Debt', 'Long Term Debt', 'Total Non Current Liabilities Net Minority Interest'])
equity = safe_get(balance_sheet, ['Stockholders Equity', 'Total Stockholders Equity', 'Common Stock Equity', 'Total Equity Gross Minority Interest'])
inventory_val = safe_get(balance_sheet, ['Inventory'])
fcf = safe_get(cash_flow, ['Free Cash Flow'])

ratio_sections = []
roe = safe_div(net_income, equity); roa = safe_div(net_income, total_assets)
gm = safe_div(gross_profit, revenue); om = safe_div(operating_income, revenue); nm = safe_div(net_income, revenue)
ratio_sections.append(('PROFITABILITY', [
    ('Return on Equity', fmt_pct(roe), *grade(roe, (0.20, 0.12, 0.06, 0.02))),
    ('Return on Assets', fmt_pct(roa), *grade(roa, (0.10, 0.06, 0.03, 0.01))),
    ('Gross Margin', fmt_pct(gm), *grade(gm, (0.50, 0.35, 0.20, 0.10))),
    ('Operating Margin', fmt_pct(om), *grade(om, (0.25, 0.15, 0.08, 0.03))),
    ('Net Margin', fmt_pct(nm), *grade(nm, (0.20, 0.10, 0.05, 0.02))),
]))
cr = safe_div(current_assets, current_liabilities)
qr = safe_div((current_assets - (inventory_val or 0)), current_liabilities) if current_assets and current_liabilities else None
ratio_sections.append(('LIQUIDITY', [
    ('Current Ratio', fmt_num(cr), *grade(cr, (2.0, 1.5, 1.0, 0.7))),
    ('Quick Ratio', fmt_num(qr), *grade(qr, (1.5, 1.0, 0.7, 0.4))),
]))
de = safe_div(total_debt, equity)
ic = safe_div(operating_income, abs(interest_expense)) if interest_expense and interest_expense != 0 else None
da = safe_div(total_debt, total_assets)
ratio_sections.append(('SOLVENCY', [
    ('Debt to Equity', fmt_num(de), *grade(de, (0.5, 1.0, 2.0, 3.0), reverse=True)),
    ('Interest Coverage', fmt_num(ic, 'x'), *grade(ic, (8, 5, 3, 1.5))),
    ('Debt to Assets', fmt_pct(da), *grade(da, (0.20, 0.35, 0.50, 0.65), reverse=True)),
]))
at = safe_div(revenue, total_assets); fm = safe_div(fcf, revenue)
ratio_sections.append(('EFFICIENCY', [
    ('Asset Turnover', fmt_num(at, 'x'), *grade(at, (1.0, 0.7, 0.4, 0.2))),
    ('FCF Margin', fmt_pct(fm), *grade(fm, (0.20, 0.10, 0.05, 0.01))),
]))

all_ratios = []
for _, sr in ratio_sections:
    for item in sr: all_ratios.append(item)
grade_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
for _, _, g, _ in all_ratios:
    if g in grade_counts: grade_counts[g] += 1

# ══════════════════════════════════════════════════
# GROWTH
# ══════════════════════════════════════════════════
years_list, revenues_list, net_incomes_list = [], [], []
for i in range(min(income_stmt.shape[1], 5)):
    col_date = income_stmt.columns[i]
    yr = col_date.year if hasattr(col_date, 'year') else int(str(col_date)[:4])
    rev = safe_get(income_stmt, ['Total Revenue', 'Revenue'], i)
    ni = safe_get(income_stmt, ['Net Income', 'Net Income Common Stockholders'], i)
    if rev:
        years_list.append(yr); revenues_list.append(rev / 1e9); net_incomes_list.append((ni or 0) / 1e9)
years_list = years_list[::-1]; revenues_list = revenues_list[::-1]; net_incomes_list = net_incomes_list[::-1]
n_years = max(len(revenues_list) - 1, 1)
revenue_cagr = (revenues_list[-1] / revenues_list[0]) ** (1 / n_years) - 1 if len(revenues_list) >= 2 and revenues_list[0] > 0 else 0
yoy_growth = [(revenues_list[i] - revenues_list[i-1]) / revenues_list[i-1] if revenues_list[i-1] > 0 else 0 for i in range(1, len(revenues_list))]
accelerating = len(yoy_growth) >= 2 and yoy_growth[-1] > yoy_growth[-2]

# ══════════════════════════════════════════════════
# CML + RISK
# ══════════════════════════════════════════════════
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
else: beta = 1.0; jensens_alpha = 0

treynor = (annual_return - rf_rate) / beta if beta != 0 else 0
downside = monthly_returns[monthly_returns < 0]
downside_std = downside.std() * np.sqrt(12) if len(downside) > 0 else annual_vol
sortino = (annual_return - rf_rate) / downside_std if downside_std > 0 else 0

price_ret = hist['Close'].pct_change().fillna(0)
cumulative = (1 + price_ret).cumprod(); rolling_max = cumulative.cummax()
drawdown = (cumulative - rolling_max) / rolling_max; max_drawdown = drawdown.min()

mvp_vol = market_vol * 0.6; mvp_return = rf_rate + cml_slope * mvp_vol * 0.85
max_vol_plot = max(annual_vol, market_vol, 0.15) * 1.5

# ══════════════════════════════════════════════════
# VERDICT
# ══════════════════════════════════════════════════
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
        ax1.bar(x-w/2, revenues_list, w, color=BLUE, alpha=0.85, label='Revenue ($B)')
        ax1.bar(x+w/2, net_incomes_list, w, color=ACCENT, alpha=0.85, label='Net Income ($B)')
        mr = max(revenues_list) if revenues_list else 1
        for bar in ax1.patches[:len(years_list)]:
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+mr*0.01, f'{bar.get_height():.0f}', ha='center', fontsize=9, color=BLUE)
        for bar in ax1.patches[len(years_list):]:
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+mr*0.01, f'{bar.get_height():.0f}', ha='center', fontsize=9, color=ACCENT)
        ax1.set_xticks(x); ax1.set_xticklabels([str(y) for y in years_list])
        ax1.set_title('Revenue & Income ($B)', fontsize=12, fontweight='bold', color='white'); ax1.set_ylabel('$ Billions'); ax1.legend(fontsize=8, framealpha=0.3)
        if yoy_growth:
            gx = np.arange(len(yoy_growth)); gc = [ACCENT if g>=0 else RED for g in yoy_growth]
            ax2.bar(gx, [g*100 for g in yoy_growth], color=gc, alpha=0.85)
            ax2.axhline(y=0, color='white', linewidth=0.5, alpha=0.3)
            ax2.set_xticks(gx); ax2.set_xticklabels([str(y) for y in years_list[1:]])
            for i,g in enumerate(yoy_growth):
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
for idx, (sn, sr) in enumerate(ratio_sections):
    col = r1 if idx % 2 == 0 else r2
    with col:
        st.markdown(f'<p class="section-label">{sn}</p>', unsafe_allow_html=True)
        for name, value, g, color in sr:
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
