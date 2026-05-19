import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests as req
import time
import json
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
.insight-box { background:#0a1020; border:1px solid #1a2744; border-radius:10px; padding:18px 20px; margin:10px 0; }
.persona-card { background:#0f1628; border:1px solid #1a2744; border-radius:10px; padding:18px; margin:6px 0; }
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
# API KEY FOR AI FEATURES (optional — analysis works without it)
# ══════════════════════════════════════════════════
try:
    CLAUDE_KEY = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    CLAUDE_KEY = None

# ══════════════════════════════════════════════════
# YFINANCE DATA PULLER — cached, with browser headers
# ══════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner="Pulling financial data...")
def pull_data(ticker_symbol):
    session = req.Session()
    session.headers['User-Agent'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

    for attempt in range(3):
        try:
            stock = yf.Ticker(ticker_symbol, session=session)
            info = stock.info
            if not info or info.get('quoteType') is None:
                return None, f"No data found for {ticker_symbol}."
            time.sleep(2)
            income = stock.income_stmt
            balance = stock.balance_sheet
            cashflow = stock.cashflow
            if income is None or income.empty:
                return None, f"No financial statements for {ticker_symbol}."
            time.sleep(2)
            hist = stock.history(period='5y', interval='1mo')
            if hist is None or hist.empty:
                return None, f"No price history for {ticker_symbol}."
            time.sleep(2)
            sp = yf.Ticker('^GSPC', session=session)
            sp_hist = sp.history(period='5y', interval='1mo')
            rf = 0.045
            try:
                time.sleep(2)
                tnx = yf.Ticker('^TNX', session=session)
                tnx_info = tnx.info
                rf_val = tnx_info.get('regularMarketPrice', tnx_info.get('previousClose', 4.5))
                if rf_val: rf = float(rf_val) / 100
            except: pass
            return {
                'info': info, 'income': income, 'balance': balance,
                'cashflow': cashflow, 'hist': hist, 'sp_hist': sp_hist, 'rf': rf,
            }, None
        except Exception as e:
            if attempt < 2:
                time.sleep(10)
                continue
            return None, f"Failed after 3 attempts: {str(e)[:200]}"
    return None, "Unknown error."

# ══════════════════════════════════════════════════
# AI CALLER — for risk assessment + deep research
# ══════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def call_claude(prompt):
    if not CLAUDE_KEY:
        return None
    try:
        r = req.post("https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json", "x-api-key": CLAUDE_KEY, "anthropic-version": "2023-06-01"},
            json={"model": "claude-sonnet-4-20250514", "max_tokens": 3000,
                  "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=60)
        if r.status_code == 200:
            data = r.json()
            return "\n".join([c.get('text', '') for c in data.get('content', []) if c.get('type') == 'text'])
    except: pass
    return None

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
# REALISTIC SCORING ENGINE (v2)
# ══════════════════════════════════════════════════
def compute_verdict_v2(all_ratios, cml_distance, sharpe, market_sharpe, revenue_cagr, accelerating, annual_return, annual_vol, beta, max_drawdown):
    """
    4 pillars, each scored 0-25, total 0-100.
    Much harder to hit 80+. Most stocks land 35-70.
    """
    grade_pts = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0, 'N/A': 1.5}
    scored_ratios = [r for r in all_ratios if r[2] != 'N/A']
    total_grade_pts = sum(grade_pts.get(r[2], 0) for r in all_ratios)
    max_possible = len(all_ratios) * 4
    fundamental_pct = total_grade_pts / max_possible if max_possible > 0 else 0.5
    fundamental_score = fundamental_pct * 25

    # Pillar 2: CML + Risk-Adjusted (0-25)
    cml_score = 0
    if cml_distance > 0.05: cml_score += 10
    elif cml_distance > 0.02: cml_score += 7
    elif cml_distance > 0: cml_score += 4
    elif cml_distance > -0.02: cml_score += 2
    else: cml_score += 0

    if sharpe > 1.0: cml_score += 8
    elif sharpe > 0.7: cml_score += 6
    elif sharpe > 0.4: cml_score += 4
    elif sharpe > 0: cml_score += 2
    else: cml_score += 0

    sharpe_vs_market = sharpe - market_sharpe
    if sharpe_vs_market > 0.3: cml_score += 7
    elif sharpe_vs_market > 0.1: cml_score += 5
    elif sharpe_vs_market > -0.1: cml_score += 3
    else: cml_score += 0
    cml_score = min(cml_score, 25)

    # Pillar 3: Growth (0-25)
    growth_score = 0
    if revenue_cagr > 0.20: growth_score += 12
    elif revenue_cagr > 0.10: growth_score += 9
    elif revenue_cagr > 0.05: growth_score += 6
    elif revenue_cagr > 0.02: growth_score += 3
    elif revenue_cagr > 0: growth_score += 1
    else: growth_score += 0

    if accelerating: growth_score += 6
    else: growth_score += 1

    # Penalize negative growth harder
    if revenue_cagr < -0.05: growth_score = max(growth_score - 5, 0)

    # Revenue consistency bonus
    growth_score = min(growth_score, 25)

    # Pillar 4: Risk Profile (0-25) — lower vol, smaller drawdown = higher score
    risk_score = 0
    if annual_vol < 0.15: risk_score += 8
    elif annual_vol < 0.25: risk_score += 6
    elif annual_vol < 0.35: risk_score += 4
    elif annual_vol < 0.50: risk_score += 2
    else: risk_score += 0

    dd = abs(max_drawdown) if max_drawdown else 0
    if dd < 0.15: risk_score += 8
    elif dd < 0.25: risk_score += 6
    elif dd < 0.35: risk_score += 4
    elif dd < 0.50: risk_score += 2
    else: risk_score += 0

    if beta is not None:
        if 0.5 <= beta <= 1.2: risk_score += 5
        elif 0.3 <= beta <= 1.5: risk_score += 3
        else: risk_score += 1

    # Positive return bonus
    if annual_return > 0.15: risk_score += 4
    elif annual_return > 0.05: risk_score += 2
    elif annual_return < 0: risk_score = max(risk_score - 3, 0)

    risk_score = min(risk_score, 25)

    total = round(fundamental_score + cml_score + growth_score + risk_score)
    total = max(0, min(100, total))

    # Verdict bands
    if total >= 78: verdict, vc = 'STRONG BUY', ACCENT
    elif total >= 62: verdict, vc = 'BUY', BLUE
    elif total >= 45: verdict, vc = 'HOLD', AMBER
    elif total >= 30: verdict, vc = 'UNDERPERFORM', '#ff9f43'
    else: verdict, vc = 'AVOID', RED

    return {
        'score': total,
        'verdict': verdict,
        'color': vc,
        'pillars': {
            'fundamentals': round(fundamental_score, 1),
            'cml_risk_adj': round(cml_score, 1),
            'growth': round(growth_score, 1),
            'risk_profile': round(risk_score, 1),
        }
    }

# ══════════════════════════════════════════════════
# INVESTOR PERSONA ANALYSIS
# ══════════════════════════════════════════════════
def investor_personas(annual_return, annual_vol, beta, sharpe, cml_distance, revenue_cagr, max_drawdown, de_ratio):
    personas = []

    # Conservative investor
    if annual_vol < 0.20 and abs(max_drawdown) < 0.25 and (de_ratio is None or de_ratio < 1.0):
        con_verdict = "GOOD FIT"
        con_color = ACCENT
        con_reason = f"Low volatility ({annual_vol*100:.0f}%), manageable drawdown ({max_drawdown*100:.0f}%), conservative debt levels. Steady compounder profile."
    elif annual_vol < 0.30 and abs(max_drawdown) < 0.35:
        con_verdict = "MODERATE FIT"
        con_color = AMBER
        con_reason = f"Acceptable volatility ({annual_vol*100:.0f}%) but drawdown of {max_drawdown*100:.0f}% may test patience. Not ideal but not terrible."
    else:
        con_verdict = "POOR FIT"
        con_color = RED
        con_reason = f"Too volatile ({annual_vol*100:.0f}%) with {max_drawdown*100:.0f}% max drawdown. Conservative investors should look elsewhere."
    personas.append(("Conservative Investor", "Prioritizes capital preservation. Wants steady returns, low drawdowns, minimal surprises.", con_verdict, con_color, con_reason))

    # Moderate / balanced
    if sharpe > 0.4 and annual_return > 0.05:
        mod_verdict = "GOOD FIT"
        mod_color = ACCENT
        mod_reason = f"Decent risk-adjusted returns (Sharpe {sharpe:.2f}). Balanced between growth and stability. The kind of stock you hold for 3-5 years."
    elif sharpe > 0.2 and annual_return > 0:
        mod_verdict = "MODERATE FIT"
        mod_color = AMBER
        mod_reason = f"Acceptable Sharpe ({sharpe:.2f}) but returns of {annual_return*100:.1f}% aren't spectacular. Could do better."
    else:
        mod_verdict = "POOR FIT"
        mod_color = RED
        mod_reason = f"Sharpe of {sharpe:.2f} means risk isn't being compensated well. Balanced investors have better options."
    personas.append(("Balanced Investor", "Wants reasonable returns without stomach-churning volatility. Happy with market-beating Sharpe ratio.", mod_verdict, mod_color, mod_reason))

    # Aggressive / growth
    if revenue_cagr > 0.10 and annual_return > 0.15:
        agg_verdict = "GOOD FIT"
        agg_color = ACCENT
        agg_reason = f"Strong revenue growth ({revenue_cagr*100:.1f}% CAGR) with {annual_return*100:.1f}% annual returns. High risk, high reward — exactly what aggressive investors want."
    elif revenue_cagr > 0.05 or annual_return > 0.10:
        agg_verdict = "MODERATE FIT"
        agg_color = AMBER
        agg_reason = f"Some growth ({revenue_cagr*100:.1f}% CAGR) but may not be explosive enough. Aggressive investors might want more upside."
    else:
        agg_verdict = "POOR FIT"
        agg_color = RED
        agg_reason = f"Revenue CAGR of {revenue_cagr*100:.1f}% and returns of {annual_return*100:.1f}% won't satisfy aggressive growth seekers."
    personas.append(("Aggressive / Growth Investor", "Chases high returns. Comfortable with big drawdowns if the upside is there. Wants 15%+ annually.", agg_verdict, agg_color, agg_reason))

    # Income / dividend
    if annual_vol < 0.25 and (de_ratio is None or de_ratio < 2.0):
        inc_verdict = "CHECK DIVIDEND"
        inc_color = BLUE
        inc_reason = "Stable enough for income investors. Check the dividend yield and payout ratio separately."
    else:
        inc_verdict = "NOT IDEAL"
        inc_color = AMBER
        inc_reason = f"Volatility of {annual_vol*100:.0f}% and leverage concerns make this less suitable for income-focused portfolios."
    personas.append(("Income / Dividend Investor", "Wants reliable cash flow. Cares about dividend safety, low debt, and predictability.", inc_verdict, inc_color, inc_reason))

    return personas

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
    c1, c2, c3 = st.columns(3)
    c1.markdown("**📊 Analysis**\n\nRatios, CML, growth, realistic scoring, investor profile matching")
    c2.markdown("**⚠️ Risk Assessment**\n\nAI-powered bear case: accounting risks, competition, concentration")
    c3.markdown("**🔬 Deep Research**\n\nBusiness model, moat, catalysts, asymmetric upside/downside")
    if not CLAUDE_KEY:
        st.info("Add ANTHROPIC_API_KEY to Streamlit secrets to enable AI-powered Risk Assessment and Deep Research tabs.")
    st.stop()

# ══════════════════════════════════════════════════
# PULL DATA
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
industry = info.get('industry', 'N/A')
price = info.get('currentPrice', info.get('regularMarketPrice', info.get('previousClose', 'N/A')))
mc = info.get('marketCap', 0)
if mc and mc > 1e12: mcs = f'${mc/1e12:.2f}T'
elif mc and mc > 1e9: mcs = f'${mc/1e9:.1f}B'
else: mcs = 'N/A'

monthly_returns = hist['Close'].pct_change().dropna()
sp500_returns = sp_hist['Close'].pct_change().dropna() if sp_hist is not None and not sp_hist.empty else pd.Series(dtype=float)

# ══════════════════════════════════════════════════
# COMPUTE ALL METRICS
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
cr_val = safe_div(current_assets, current_liabilities)
qr_val = safe_div((current_assets - (inventory_val or 0)), current_liabilities) if current_assets and current_liabilities else None
ratio_sections.append(('LIQUIDITY', [
    ('Current Ratio', fmt_num(cr_val), *grade(cr_val, (2.0, 1.5, 1.0, 0.7))),
    ('Quick Ratio', fmt_num(qr_val), *grade(qr_val, (1.5, 1.0, 0.7, 0.4))),
]))
de_ratio = safe_div(total_debt, equity)
ic_val = safe_div(operating_income, abs(interest_expense)) if interest_expense and interest_expense != 0 else None
da_val = safe_div(total_debt, total_assets)
ratio_sections.append(('SOLVENCY', [
    ('Debt to Equity', fmt_num(de_ratio), *grade(de_ratio, (0.5, 1.0, 2.0, 3.0), reverse=True)),
    ('Interest Coverage', fmt_num(ic_val, 'x'), *grade(ic_val, (8, 5, 3, 1.5))),
    ('Debt to Assets', fmt_pct(da_val), *grade(da_val, (0.20, 0.35, 0.50, 0.65), reverse=True)),
]))
at_val = safe_div(revenue, total_assets); fm_val = safe_div(fcf, revenue)
ratio_sections.append(('EFFICIENCY', [
    ('Asset Turnover', fmt_num(at_val, 'x'), *grade(at_val, (1.0, 0.7, 0.4, 0.2))),
    ('FCF Margin', fmt_pct(fm_val), *grade(fm_val, (0.20, 0.10, 0.05, 0.01))),
]))

all_ratios = []
for _, sr in ratio_sections:
    for item in sr: all_ratios.append(item)
grade_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
for _, _, g, _ in all_ratios:
    if g in grade_counts: grade_counts[g] += 1

# Growth
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

# CML + Risk
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
# VERDICT V2
# ══════════════════════════════════════════════════
v = compute_verdict_v2(all_ratios, cml_distance, sharpe, market_sharpe, revenue_cagr, accelerating, annual_return, annual_vol, beta, max_drawdown)
score = v['score']; verdict = v['verdict']; vc = v['color']; pillars = v['pillars']

# Investor personas
personas = investor_personas(annual_return, annual_vol, beta, sharpe, cml_distance, revenue_cagr, max_drawdown, de_ratio)

# ══════════════════════════════════════════════════
# DISPLAY — TABS
# ══════════════════════════════════════════════════
st.markdown("---")
st.markdown(f"## {company_name}")
st.markdown(f"**${ticker}** · {sector} · {industry} · ${price} · {mcs}")

tab1, tab2, tab3 = st.tabs(["📊 Analysis", "⚠️ Risk Assessment", "🔬 Deep Research"])

# ══════════════════════════════════════════════════
# TAB 1: ANALYSIS
# ══════════════════════════════════════════════════
with tab1:
    # Verdict banner
    st.markdown(f"""<div style="background:{vc}10;border:1px solid {vc}30;border-radius:12px;padding:24px;margin:16px 0;position:relative;overflow:hidden;">
    <div style="position:absolute;top:0;left:0;right:0;height:3px;background:{vc};opacity:0.6;"></div>
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;">
    <div>
    <p class="section-label">THE VERDICT</p>
    <h2 style="color:{vc} !important;font-size:28px;margin:0 0 8px;">{verdict}</h2>
    </div>
    <div style="text-align:center;">
    <p class="section-label">SCORE</p>
    <p style="font-size:40px;font-weight:800;color:{vc};font-family:'IBM Plex Mono',monospace;margin:0;">{score}</p>
    </div></div></div>""", unsafe_allow_html=True)

    # Score breakdown
    st.markdown('<p class="section-label">SCORE BREAKDOWN — WHAT\'S DRIVING THE VERDICT</p>', unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Fundamentals", f"{pillars['fundamentals']}/25")
    p2.metric("CML + Risk-Adj", f"{pillars['cml_risk_adj']}/25")
    p3.metric("Growth", f"{pillars['growth']}/25")
    p4.metric("Risk Profile", f"{pillars['risk_profile']}/25")

    # Detailed verdict explanation
    st.markdown('<p class="section-label">WHY THIS SCORE</p>', unsafe_allow_html=True)

    # Build detailed explanation
    explanations = []

    # Fundamentals explanation
    if pillars['fundamentals'] >= 18:
        explanations.append(f"**Fundamentals ({pillars['fundamentals']}/25):** Strong. {grade_counts.get('A',0)} ratios graded A. Profitability and financial health are solid.")
    elif pillars['fundamentals'] >= 12:
        explanations.append(f"**Fundamentals ({pillars['fundamentals']}/25):** Decent but not exceptional. Mix of grades — some strengths, some concerns.")
    else:
        explanations.append(f"**Fundamentals ({pillars['fundamentals']}/25):** Weak. {grade_counts.get('F',0)} F-grades and {grade_counts.get('D',0)} D-grades. Profitability or solvency issues.")

    # CML explanation
    if cml_distance > 0.02:
        explanations.append(f"**CML Position ({pillars['cml_risk_adj']}/25):** {cml_distance*100:.1f}% above the Capital Market Line. You're earning more than the risk warrants. Sharpe of {sharpe:.2f} vs market's {market_sharpe:.2f}.")
    elif cml_distance > -0.02:
        explanations.append(f"**CML Position ({pillars['cml_risk_adj']}/25):** Roughly on the CML ({cml_distance*100:+.1f}%). Fair risk-return tradeoff — not overpaying for risk, but no bargain either. Sharpe of {sharpe:.2f}.")
    else:
        explanations.append(f"**CML Position ({pillars['cml_risk_adj']}/25):** {abs(cml_distance)*100:.1f}% below the CML. At {annual_vol*100:.0f}% volatility, a simple index+treasury mix would yield {cml_expected*100:.1f}% vs the {annual_return*100:.1f}% you're getting. Sharpe of {sharpe:.2f} trails market's {market_sharpe:.2f}.")

    # Growth explanation
    if revenue_cagr > 0.10:
        explanations.append(f"**Growth ({pillars['growth']}/25):** Revenue growing {revenue_cagr*100:.1f}% per year. {'Accelerating — latest year grew faster than the prior.' if accelerating else 'Decelerating — growth is slowing down.'}")
    elif revenue_cagr > 0:
        explanations.append(f"**Growth ({pillars['growth']}/25):** Modest {revenue_cagr*100:.1f}% CAGR. Not a high-flier. {'Trend is improving.' if accelerating else 'Trend is weakening.'}")
    else:
        explanations.append(f"**Growth ({pillars['growth']}/25):** Revenue declining at {revenue_cagr*100:.1f}% per year. This is a red flag regardless of other metrics.")

    # Risk explanation
    if annual_vol < 0.20:
        explanations.append(f"**Risk ({pillars['risk_profile']}/25):** Low volatility ({annual_vol*100:.0f}%) and max drawdown of {max_drawdown*100:.0f}%. Beta of {beta:.2f}. Relatively calm ride.")
    elif annual_vol < 0.35:
        explanations.append(f"**Risk ({pillars['risk_profile']}/25):** Moderate volatility ({annual_vol*100:.0f}%). Max drawdown hit {max_drawdown*100:.0f}%. Beta of {beta:.2f}. Some bumps but manageable.")
    else:
        explanations.append(f"**Risk ({pillars['risk_profile']}/25):** High volatility ({annual_vol*100:.0f}%) with {max_drawdown*100:.0f}% max drawdown. Beta of {beta:.2f}. This stock swings hard — you need the stomach for it.")

    for exp in explanations:
        st.markdown(f'<div class="insight-box">{exp}</div>', unsafe_allow_html=True)

    # Key metrics
    st.markdown("---")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("CML GAP", f"{cml_distance*100:+.1f}%")
    m2.metric("SHARPE", f"{sharpe:.2f}")
    m3.metric("CAGR", f"{revenue_cagr*100:.1f}%")
    m4.metric("RETURN", f"{annual_return*100:.1f}%")
    m5.metric("VOL", f"{annual_vol*100:.1f}%")

    st.markdown("---")

    # Charts
    left, right = st.columns(2)
    with left:
        st.markdown("### Revenue & Growth")
        if years_list and revenues_list:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            x = np.arange(len(years_list)); w = 0.35
            ax1.bar(x-w/2, revenues_list, w, color=BLUE, alpha=0.85, label='Revenue ($B)')
            ax1.bar(x+w/2, net_incomes_list, w, color=ACCENT, alpha=0.85, label='Net Income ($B)')
            mr = max(revenues_list) if revenues_list else 1
            for bar in ax1.patches[:len(years_list)]: ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+mr*0.01, f'{bar.get_height():.0f}', ha='center', fontsize=9, color=BLUE)
            for bar in ax1.patches[len(years_list):]: ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+mr*0.01, f'{bar.get_height():.0f}', ha='center', fontsize=9, color=ACCENT)
            ax1.set_xticks(x); ax1.set_xticklabels([str(y) for y in years_list])
            ax1.set_title('Revenue & Income ($B)', fontsize=12, fontweight='bold', color='white'); ax1.legend(fontsize=8, framealpha=0.3)
            if yoy_growth:
                gx = np.arange(len(yoy_growth)); gc = [ACCENT if g>=0 else RED for g in yoy_growth]
                ax2.bar(gx, [g*100 for g in yoy_growth], color=gc, alpha=0.85)
                ax2.axhline(y=0, color='white', linewidth=0.5, alpha=0.3)
                ax2.set_xticks(gx); ax2.set_xticklabels([str(y) for y in years_list[1:]])
                for i,g in enumerate(yoy_growth): ax2.text(i, g*100+(0.3 if g>=0 else -1.2), f'{g*100:.1f}%', ha='center', fontsize=10, fontweight='bold', color=gc[i])
            ax2.set_title('YoY Growth', fontsize=12, fontweight='bold', color='white')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with right:
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
        ax.scatter([market_vol*100], [market_return*100], color=BLUE, s=140, zorder=5, edgecolors='white', linewidth=1.5)
        ax.annotate('S&P 500', (market_vol*100,market_return*100), xytext=(12,-8), textcoords='offset points', fontsize=9, color=BLUE, fontweight='bold')
        ax.scatter([mvp_vol*100], [mvp_return*100], color=AMBER, s=100, zorder=5, marker='D', edgecolors='white', linewidth=1.5)
        sc = ACCENT if cml_distance > 0 else RED
        ax.scatter([annual_vol*100], [annual_return*100], color=sc, s=220, zorder=6, edgecolors='white', linewidth=2)
        pt = f'+{cml_distance*100:.1f}%' if cml_distance > 0 else f'{cml_distance*100:.1f}%'
        ax.annotate(f'{ticker}\n{pt}', (annual_vol*100,annual_return*100), xytext=(14,6), textcoords='offset points', fontsize=10, color=sc, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=BG, edgecolor=sc, alpha=0.9))
        ax.plot([annual_vol*100]*2, [annual_return*100,cml_expected*100], color=sc, linestyle='--', linewidth=1.5, alpha=0.6)
        ax.set_xlabel('Risk (%)'); ax.set_ylabel('Return (%)')
        ax.legend(loc='upper left', framealpha=0.3, fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")

    # Ratios
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

    # Risk metrics
    rm1, rm2, rm3, rm4 = st.columns(4)
    rm1.metric("Treynor", f"{treynor:.2f}")
    rm2.metric("Sortino", f"{sortino:.2f}")
    rm3.metric("Jensen's Alpha", f"{jensens_alpha*100:+.2f}%")
    rm4.metric("Max Drawdown", f"{max_drawdown*100:.1f}%")

    st.markdown("---")

    # Investor personas
    st.markdown("### Who should buy this?")
    st.markdown("How this stock fits different investor personalities.")
    for persona_name, persona_desc, persona_verdict, persona_color, persona_reason in personas:
        st.markdown(f"""<div class="persona-card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <span style="font-size:15px;font-weight:700;color:#f0f4f8;">{persona_name}</span>
            <span style="padding:4px 12px;border-radius:6px;background:{persona_color}18;color:{persona_color};font-size:12px;font-weight:700;font-family:'IBM Plex Mono',monospace;">{persona_verdict}</span>
        </div>
        <p style="font-size:12px;color:#4e6380;margin:0 0 8px;">{persona_desc}</p>
        <p style="font-size:13px;color:#c0ccd8;margin:0;">{persona_reason}</p>
        </div>""", unsafe_allow_html=True)

    # Bottom line
    if verdict in ('STRONG BUY', 'BUY'): bl = f"The math favors {ticker}. Strong fundamentals, good risk-return tradeoff."
    elif verdict == 'HOLD': bl = f"{ticker} is fairly priced. Not a bad stock, but not a clear winner at this price."
    elif verdict == 'UNDERPERFORM': bl = f"{ticker} isn't earning its risk premium. You can do better at this volatility level."
    else: bl = f"The numbers don't support {ticker}. An index fund would likely outperform at lower risk."

    st.markdown(f"""<div style="text-align:center;padding:20px;background:{vc}0a;border:1px solid {vc}18;border-radius:10px;margin-top:16px;">
    <p class="section-label">BOTTOM LINE</p>
    <p style="font-size:16px;font-weight:600;color:{vc};margin:8px 0 0;">{bl}</p>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# TAB 2: RISK ASSESSMENT (AI)
# ══════════════════════════════════════════════════
with tab2:
    st.markdown("### ⚠️ Risk Assessment")
    st.markdown(f"AI-generated bear case for **{company_name}**. What could go wrong?")

    if not CLAUDE_KEY:
        st.info("Add ANTHROPIC_API_KEY to your Streamlit secrets to enable this feature.\n\nGet a key at console.anthropic.com")
    else:
        if st.button("Generate Risk Assessment", key="risk_btn", type="primary"):
            with st.spinner("Researching risks... (30-45 seconds)"):
                risk_prompt = f"""Search the web for recent bear cases, risks, and concerns about {company_name} (ticker: {ticker}).

Then write a skeptical risk assessment covering exactly these 3 areas:

1. ACCOUNTING & FINANCIAL RISKS: Any red flags in how they report numbers? Aggressive revenue recognition? Off-balance-sheet liabilities? Goodwill impairment risk? Be specific with numbers if possible.

2. CUSTOMER & REVENUE CONCENTRATION: How dependent are they on a small number of customers, products, or geographies? What happens if their biggest revenue source shrinks 20%?

3. COMPETITIVE THREATS: Who is actively trying to take their market share? Any disruptive technologies that could make their business model obsolete? Name specific competitors and what they're doing.

End with a BEAR CASE SUMMARY: If everything goes wrong in the next 2 years, what does the downside look like? Give a rough downside price target or percentage.

Be specific. Use real numbers and real competitor names. No generic platitudes."""

                result = call_claude(risk_prompt)
                if result:
                    st.markdown(result)
                else:
                    st.error("Could not generate risk assessment. Check your API key.")

# ══════════════════════════════════════════════════
# TAB 3: DEEP RESEARCH (AI)
# ══════════════════════════════════════════════════
with tab3:
    st.markdown("### 🔬 Deep Research Report")
    st.markdown(f"AI-generated comprehensive analysis of **{company_name}**.")

    if not CLAUDE_KEY:
        st.info("Add ANTHROPIC_API_KEY to your Streamlit secrets to enable this feature.\n\nGet a key at console.anthropic.com")
    else:
        if st.button("Generate Deep Research", key="research_btn", type="primary"):
            with st.spinner("Deep research in progress... (45-60 seconds)"):
                research_prompt = f"""Search the web for comprehensive information about {company_name} (ticker: {ticker}, sector: {sector}).

Write a thorough research report covering these 4 areas:

1. BUSINESS MODEL: How exactly does {company_name} make money? Explain their core product/service in plain English. What's their revenue mix? Subscription vs one-time? Hardware vs software vs services? Who pays them and why?

2. MOAT & COMPETITION:
   - Name the top 3 direct competitors
   - Does {company_name} have a technological advantage, patent portfolio, or network effect that competitors can't easily replicate?
   - How defensible is their market position? Rate it: Strong Moat / Moderate Moat / Weak Moat and explain why.

3. CATALYSTS (Next 12 Months):
   - Any upcoming product launches, regulatory approvals, or major partnerships?
   - Earnings expectations — is the street bullish or bearish?
   - Any macro trends (AI, EVs, rate cuts, etc.) that specifically help or hurt this company?

4. ASYMMETRY CHECK:
   - What's the realistic downside floor? (If everything goes wrong, where does the stock land?)
   - What's the realistic upside ceiling? (If catalysts hit, where could it go?)
   - Is the risk/reward asymmetric? Meaning: is there more to gain than to lose from here?
   - Final take: Is this stock currently priced for perfection, priced for disaster, or somewhere reasonable?

Use real numbers, real analyst targets, and real competitive dynamics. No vague generalities."""

                result = call_claude(research_prompt)
                if result:
                    st.markdown(result)
                else:
                    st.error("Could not generate research report. Check your API key.")

st.markdown("---")
st.markdown('<p style="text-align:center;font-size:11px;color:#4e6380;">STOCKSIGHT — not financial advice, just math and AI</p>', unsafe_allow_html=True)
