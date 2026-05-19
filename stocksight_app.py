import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
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

try:
    CLAUDE_KEY = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    CLAUDE_KEY = None

@st.cache_data(ttl=3600, show_spinner="Pulling financial data...")
def pull_data(ticker_symbol):
    for attempt in range(3):
        try:
            stock = yf.Ticker(ticker_symbol)
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
            sp = yf.Ticker('^GSPC')
            sp_hist = sp.history(period='5y', interval='1mo')
            rf = 0.045
            try:
                time.sleep(2)
                tnx = yf.Ticker('^TNX')
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

@st.cache_data(ttl=3600, show_spinner=False)
def call_claude(prompt):
    if not CLAUDE_KEY:
        return "**Error:** No API key found. Add ANTHROPIC_API_KEY to Streamlit secrets."
    try:
        r = requests.post("https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json", "x-api-key": CLAUDE_KEY, "anthropic-version": "2023-06-01"},
            json={"model": "claude-sonnet-4-6", "max_tokens": 3000,
                  "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=60)
        if r.status_code == 200:
            data = r.json()
            text = "\n".join([c.get('text', '') for c in data.get('content', []) if c.get('type') == 'text'])
            return text if text else "API returned empty response."
        else:
            return f"**API Error {r.status_code}:** {r.text[:500]}"
    except Exception as e:
        return f"**Connection Error:** {str(e)[:300]}"

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

def compute_verdict_v2(all_ratios, cml_distance, sharpe, market_sharpe, revenue_cagr, accelerating, annual_return, annual_vol, beta, max_drawdown):
    grade_pts = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0, 'N/A': 1.5}
    total_grade_pts = sum(grade_pts.get(r[2], 0) for r in all_ratios)
    max_possible = len(all_ratios) * 4
    fundamental_pct = total_grade_pts / max_possible if max_possible > 0 else 0.5
    fundamental_score = fundamental_pct * 25
    cml_score = 0
    if cml_distance > 0.05: cml_score += 10
    elif cml_distance > 0.02: cml_score += 7
    elif cml_distance > 0: cml_score += 4
    elif cml_distance > -0.02: cml_score += 2
    if sharpe > 1.0: cml_score += 8
    elif sharpe > 0.7: cml_score += 6
    elif sharpe > 0.4: cml_score += 4
    elif sharpe > 0: cml_score += 2
    sv = sharpe - market_sharpe
    if sv > 0.3: cml_score += 7
    elif sv > 0.1: cml_score += 5
    elif sv > -0.1: cml_score += 3
    cml_score = min(cml_score, 25)
    growth_score = 0
    if revenue_cagr > 0.20: growth_score += 12
    elif revenue_cagr > 0.10: growth_score += 9
    elif revenue_cagr > 0.05: growth_score += 6
    elif revenue_cagr > 0.02: growth_score += 3
    elif revenue_cagr > 0: growth_score += 1
    if accelerating: growth_score += 6
    else: growth_score += 1
    if revenue_cagr < -0.05: growth_score = max(growth_score - 5, 0)
    growth_score = min(growth_score, 25)
    risk_score = 0
    if annual_vol < 0.15: risk_score += 8
    elif annual_vol < 0.25: risk_score += 6
    elif annual_vol < 0.35: risk_score += 4
    elif annual_vol < 0.50: risk_score += 2
    dd = abs(max_drawdown) if max_drawdown else 0
    if dd < 0.15: risk_score += 8
    elif dd < 0.25: risk_score += 6
    elif dd < 0.35: risk_score += 4
    elif dd < 0.50: risk_score += 2
    if beta is not None:
        if 0.5 <= beta <= 1.2: risk_score += 5
        elif 0.3 <= beta <= 1.5: risk_score += 3
        else: risk_score += 1
    if annual_return > 0.15: risk_score += 4
    elif annual_return > 0.05: risk_score += 2
    elif annual_return < 0: risk_score = max(risk_score - 3, 0)
    risk_score = min(risk_score, 25)
    total = round(fundamental_score + cml_score + growth_score + risk_score)
    total = max(0, min(100, total))
    if total >= 78: verdict, vc = 'STRONG BUY', ACCENT
    elif total >= 62: verdict, vc = 'BUY', BLUE
    elif total >= 45: verdict, vc = 'HOLD', AMBER
    elif total >= 30: verdict, vc = 'UNDERPERFORM', '#ff9f43'
    else: verdict, vc = 'AVOID', RED
    return {'score': total, 'verdict': verdict, 'color': vc, 'pillars': {
        'fundamentals': round(fundamental_score, 1), 'cml_risk_adj': round(cml_score, 1),
        'growth': round(growth_score, 1), 'risk_profile': round(risk_score, 1)}}

def investor_personas(annual_return, annual_vol, beta, sharpe, cml_distance, revenue_cagr, max_drawdown, de_ratio):
    personas = []
    dd = abs(max_drawdown) if max_drawdown else 0
    if annual_vol < 0.20 and dd < 0.25 and (de_ratio is None or de_ratio < 1.0):
        cv,cc,cr = "GOOD FIT",ACCENT,f"Low volatility ({annual_vol*100:.0f}%), manageable drawdown ({dd*100:.0f}%), conservative debt levels."
    elif annual_vol < 0.30 and dd < 0.35:
        cv,cc,cr = "MODERATE FIT",AMBER,f"Acceptable volatility ({annual_vol*100:.0f}%) but drawdown of {dd*100:.0f}% may test patience."
    else:
        cv,cc,cr = "POOR FIT",RED,f"Too volatile ({annual_vol*100:.0f}%) with {dd*100:.0f}% max drawdown."
    personas.append(("Conservative Investor","Prioritizes capital preservation. Wants steady returns, low drawdowns.",cv,cc,cr))
    if sharpe > 0.4 and annual_return > 0.05:
        cv,cc,cr = "GOOD FIT",ACCENT,f"Decent risk-adjusted returns (Sharpe {sharpe:.2f}). Balanced between growth and stability."
    elif sharpe > 0.2 and annual_return > 0:
        cv,cc,cr = "MODERATE FIT",AMBER,f"Acceptable Sharpe ({sharpe:.2f}) but returns of {annual_return*100:.1f}% aren't spectacular."
    else:
        cv,cc,cr = "POOR FIT",RED,f"Sharpe of {sharpe:.2f} means risk isn't being compensated well."
    personas.append(("Balanced Investor","Wants reasonable returns without stomach-churning volatility.",cv,cc,cr))
    if revenue_cagr > 0.10 and annual_return > 0.15:
        cv,cc,cr = "GOOD FIT",ACCENT,f"Strong revenue growth ({revenue_cagr*100:.1f}% CAGR) with {annual_return*100:.1f}% returns. High risk, high reward."
    elif revenue_cagr > 0.05 or annual_return > 0.10:
        cv,cc,cr = "MODERATE FIT",AMBER,f"Some growth ({revenue_cagr*100:.1f}% CAGR) but may not be explosive enough."
    else:
        cv,cc,cr = "POOR FIT",RED,f"Revenue CAGR of {revenue_cagr*100:.1f}% won't satisfy aggressive growth seekers."
    personas.append(("Aggressive / Growth Investor","Chases high returns. Comfortable with big drawdowns if upside is there.",cv,cc,cr))
    if annual_vol < 0.25 and (de_ratio is None or de_ratio < 2.0):
        cv,cc,cr = "CHECK DIVIDEND",BLUE,"Stable enough for income investors. Check the dividend yield separately."
    else:
        cv,cc,cr = "NOT IDEAL",AMBER,f"Volatility of {annual_vol*100:.0f}% makes this less suitable for income portfolios."
    personas.append(("Income / Dividend Investor","Wants reliable cash flow. Cares about dividend safety and predictability.",cv,cc,cr))
    return personas

st.markdown("## ◈ StockSight")
st.markdown("# Type a ticker. Get the truth.")
st.markdown("Pulls real financials, runs CML + portfolio math, gives you a straight answer.")
st.markdown("")

if 'active_ticker' not in st.session_state:
    st.session_state.active_ticker = ""

col_in, col_btn = st.columns([5, 1])
with col_in:
    ticker_input = st.text_input("Stock ticker", value="", placeholder="AAPL", label_visibility="collapsed")
with col_btn:
    if st.button("**Analyze**", type="primary", use_container_width=True):
        st.session_state.active_ticker = ticker_input.strip().upper()

qcols = st.columns(8)
for i, t in enumerate(["AAPL","TSLA","MSFT","AMZN","NVDA","GOOGL","META","JPM"]):
    with qcols[i]:
        if st.button(t, key=f"q{t}", use_container_width=True):
            st.session_state.active_ticker = t

ticker = st.session_state.active_ticker

if not ticker:
    st.markdown("---")
    st.markdown("### What you get")
    c1,c2,c3 = st.columns(3)
    c1.markdown("**📊 Analysis**\n\nRatios, CML, growth, realistic scoring, investor profiles")
    c2.markdown("**⚠️ Risk Assessment**\n\nAI-powered bear case: accounting, competition, concentration")
    c3.markdown("**🔬 Deep Research**\n\nBusiness model, moat, catalysts, asymmetric upside/downside")
    st.stop()

data, error = pull_data(ticker)
if error:
    st.error(error)
    st.stop()

info = data['info']; income_stmt = data['income']; balance_sheet = data['balance']
cash_flow = data['cashflow']; hist = data['hist']; sp_hist = data['sp_hist']; rf_rate = data['rf']
if rf_rate is None or rf_rate <= 0 or rf_rate > 0.20: rf_rate = 0.045

company_name = info.get('longName', info.get('shortName', ticker))
sector = info.get('sector', 'N/A'); industry = info.get('industry', 'N/A')
price = info.get('currentPrice', info.get('regularMarketPrice', info.get('previousClose', 'N/A')))
mc = info.get('marketCap', 0)
if mc and mc > 1e12: mcs = f'${mc/1e12:.2f}T'
elif mc and mc > 1e9: mcs = f'${mc/1e9:.1f}B'
else: mcs = 'N/A'

monthly_returns = hist['Close'].pct_change().dropna()
sp500_returns = sp_hist['Close'].pct_change().dropna() if sp_hist is not None and not sp_hist.empty else pd.Series(dtype=float)

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
grade_counts = {'A':0,'B':0,'C':0,'D':0,'F':0}
for _,_,g,_ in all_ratios:
    if g in grade_counts: grade_counts[g] += 1

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

v = compute_verdict_v2(all_ratios, cml_distance, sharpe, market_sharpe, revenue_cagr, accelerating, annual_return, annual_vol, beta, max_drawdown)
score = v['score']; verdict = v['verdict']; vc = v['color']; pillars = v['pillars']
personas = investor_personas(annual_return, annual_vol, beta, sharpe, cml_distance, revenue_cagr, max_drawdown, de_ratio)

st.markdown("---")
st.markdown(f"## {company_name}")
st.markdown(f"**${ticker}** · {sector} · {industry} · ${price} · {mcs}")

tab1, tab2, tab3 = st.tabs(["📊 Analysis", "⚠️ Risk Assessment", "🔬 Deep Research"])

with tab1:
    st.markdown(f"""<div style="background:{vc}10;border:1px solid {vc}30;border-radius:12px;padding:24px;margin:16px 0;position:relative;overflow:hidden;">
    <div style="position:absolute;top:0;left:0;right:0;height:3px;background:{vc};opacity:0.6;"></div>
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;">
    <div><p class="section-label">THE VERDICT</p>
    <h2 style="color:{vc} !important;font-size:28px;margin:0 0 8px;">{verdict}</h2></div>
    <div style="text-align:center;"><p class="section-label">SCORE</p>
    <p style="font-size:40px;font-weight:800;color:{vc};font-family:'IBM Plex Mono',monospace;margin:0;">{score}</p></div></div></div>""", unsafe_allow_html=True)

    # ── PLAIN ENGLISH VERDICT EXPLANATION ──
    risk_label = "Low" if annual_vol < 0.20 else "Moderate" if annual_vol < 0.35 else "High"
    risk_emoji = "🟢" if annual_vol < 0.20 else "🟡" if annual_vol < 0.35 else "🔴"
    ret_label = "Strong" if annual_return > 0.15 else "Decent" if annual_return > 0.05 else "Weak" if annual_return > 0 else "Negative"

    # Build the reason string
    if score >= 62:
        score_reason = f"The score is high because "
        reasons = []
        if pillars['fundamentals'] >= 16: reasons.append("the company's financials are healthy")
        if pillars['cml_risk_adj'] >= 14: reasons.append("it's earning more return than its risk level warrants")
        if pillars['growth'] >= 10: reasons.append(f"revenue is growing at {revenue_cagr*100:.1f}% per year")
        if pillars['risk_profile'] >= 14: reasons.append("the risk level is manageable")
        score_reason += ", ".join(reasons) + "." if reasons else "multiple factors are working in its favor."
    elif score >= 45:
        score_reason = f"The score is in the middle because "
        strengths = []
        drags = []
        if pillars['fundamentals'] >= 16: strengths.append("financials are solid")
        else: drags.append("financials are mixed")
        if pillars['cml_risk_adj'] >= 12: strengths.append("risk-return tradeoff is fair")
        else: drags.append("you're not being paid enough for the risk")
        if pillars['growth'] >= 8: strengths.append("there's some growth")
        else: drags.append("growth is slow")
        score_reason += " and ".join(strengths) + ", but " + " and ".join(drags) + "." if strengths and drags else "it's a mixed picture overall."
    else:
        score_reason = f"The score is low because "
        drags = []
        if pillars['fundamentals'] < 12: drags.append("financial health is poor")
        if pillars['cml_risk_adj'] < 8: drags.append(f"at {annual_vol*100:.0f}% volatility, a simple index fund would give you better returns")
        if pillars['growth'] < 5: drags.append("revenue growth is weak or declining")
        if pillars['risk_profile'] < 10: drags.append(f"the stock dropped {abs(max_drawdown)*100:.0f}% at its worst")
        score_reason += ", ".join(drags) + "." if drags else "multiple factors are working against it."

    # Investor guidance
    if score >= 62:
        investor_line = "Suitable for most investors. Growth and balanced investors will find this attractive."
    elif score >= 45:
        investor_line = "Best for patient, balanced investors who can handle some uncertainty. Not ideal if you need high growth or low risk."
    elif score >= 30:
        investor_line = "Only for aggressive investors who understand the downside. Conservative and income investors should avoid."
    else:
        investor_line = "Not recommended for any investor profile based on current numbers. Consider an index fund instead."

    st.markdown(f"""<div style="background:#0a1020;border:1px solid #1a2744;border-radius:10px;padding:20px;margin:10px 0;">
    <p style="font-size:15px;color:#f0f4f8;margin:0 0 12px;font-weight:600;">What this means for you</p>
    <p style="font-size:14px;color:#c0ccd8;margin:0 0 8px;">
    <b style="color:#f0f4f8;">{company_name}</b> trades at <b style="color:{vc};">${price}</b> with a market cap of <b>{mcs}</b>.
    Over the past 5 years, it returned roughly <b style="color:{'#00e5a0' if annual_return > 0 else '#ff6b6b'};">{annual_return*100:.1f}% per year</b> ({ret_label}).
    </p>
    <p style="font-size:14px;color:#c0ccd8;margin:0 0 8px;">
    {risk_emoji} <b>Risk Level: {risk_label}</b> — Volatility is {annual_vol*100:.0f}%, meaning in a bad year the stock could swing {annual_vol*100:.0f}% in either direction. The worst drawdown in the last 5 years was <b style="color:#ff6b6b;">{max_drawdown*100:.0f}%</b>. Beta of {beta:.2f} means it moves {'more' if beta > 1.1 else 'less' if beta < 0.9 else 'roughly the same as'} than the overall market.
    </p>
    <p style="font-size:14px;color:#c0ccd8;margin:0 0 8px;">
    {score_reason}
    </p>
    <p style="font-size:13px;color:#4e6380;margin:0;font-style:italic;">
    {investor_line}
    </p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-label">SCORE BREAKDOWN</p>', unsafe_allow_html=True)
    p1,p2,p3,p4 = st.columns(4)
    p1.metric("Fundamentals", f"{pillars['fundamentals']}/25")
    p2.metric("CML + Risk-Adj", f"{pillars['cml_risk_adj']}/25")
    p3.metric("Growth", f"{pillars['growth']}/25")
    p4.metric("Risk Profile", f"{pillars['risk_profile']}/25")

    st.markdown('<p class="section-label">WHY THIS SCORE</p>', unsafe_allow_html=True)
    if pillars['fundamentals'] >= 18: st.markdown(f"**Fundamentals ({pillars['fundamentals']}/25):** Strong. {grade_counts.get('A',0)} ratios graded A.")
    elif pillars['fundamentals'] >= 12: st.markdown(f"**Fundamentals ({pillars['fundamentals']}/25):** Decent but not exceptional. Mix of grades.")
    else: st.markdown(f"**Fundamentals ({pillars['fundamentals']}/25):** Weak. {grade_counts.get('F',0)} F-grades. Profitability or solvency issues.")
    if cml_distance > 0.02: st.markdown(f"**CML ({pillars['cml_risk_adj']}/25):** {cml_distance*100:.1f}% above CML. Earning more than risk warrants. Sharpe {sharpe:.2f} vs market {market_sharpe:.2f}.")
    elif cml_distance > -0.02: st.markdown(f"**CML ({pillars['cml_risk_adj']}/25):** On the CML ({cml_distance*100:+.1f}%). Fair tradeoff. Sharpe {sharpe:.2f}.")
    else: st.markdown(f"**CML ({pillars['cml_risk_adj']}/25):** {abs(cml_distance)*100:.1f}% below CML. Index+treasury would yield {cml_expected*100:.1f}% vs {annual_return*100:.1f}%. Sharpe {sharpe:.2f} trails market {market_sharpe:.2f}.")
    if revenue_cagr > 0.10: st.markdown(f"**Growth ({pillars['growth']}/25):** Revenue growing {revenue_cagr*100:.1f}%/yr. {'Accelerating.' if accelerating else 'Decelerating.'}")
    elif revenue_cagr > 0: st.markdown(f"**Growth ({pillars['growth']}/25):** Modest {revenue_cagr*100:.1f}% CAGR. {'Improving.' if accelerating else 'Weakening.'}")
    else: st.markdown(f"**Growth ({pillars['growth']}/25):** Revenue declining {revenue_cagr*100:.1f}%/yr. Red flag.")
    if annual_vol < 0.20: st.markdown(f"**Risk ({pillars['risk_profile']}/25):** Low vol ({annual_vol*100:.0f}%), drawdown {max_drawdown*100:.0f}%, beta {beta:.2f}. Calm ride.")
    elif annual_vol < 0.35: st.markdown(f"**Risk ({pillars['risk_profile']}/25):** Moderate vol ({annual_vol*100:.0f}%), drawdown {max_drawdown*100:.0f}%, beta {beta:.2f}. Some bumps.")
    else: st.markdown(f"**Risk ({pillars['risk_profile']}/25):** High vol ({annual_vol*100:.0f}%), drawdown {max_drawdown*100:.0f}%, beta {beta:.2f}. Swings hard.")

    st.markdown("---")
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("CML GAP", f"{cml_distance*100:+.1f}%"); m2.metric("SHARPE", f"{sharpe:.2f}")
    m3.metric("CAGR", f"{revenue_cagr*100:.1f}%"); m4.metric("RETURN", f"{annual_return*100:.1f}%"); m5.metric("VOL", f"{annual_vol*100:.1f}%")
    st.markdown("---")

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
    rm1,rm2,rm3,rm4 = st.columns(4)
    rm1.metric("Treynor", f"{treynor:.2f}"); rm2.metric("Sortino", f"{sortino:.2f}")
    rm3.metric("Jensen's Alpha", f"{jensens_alpha*100:+.2f}%"); rm4.metric("Max Drawdown", f"{max_drawdown*100:.1f}%")

    st.markdown("---")
    st.markdown("### Who should buy this?")
    for pname, pdesc, pverdict, pcolor, preason in personas:
        st.markdown(f"""<div style="background:#0f1628;border:1px solid #1a2744;border-radius:10px;padding:18px;margin:6px 0;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <span style="font-size:15px;font-weight:700;color:#f0f4f8;">{pname}</span>
        <span style="padding:4px 12px;border-radius:6px;background:{pcolor}18;color:{pcolor};font-size:12px;font-weight:700;font-family:'IBM Plex Mono',monospace;">{pverdict}</span></div>
        <p style="font-size:12px;color:#4e6380;margin:0 0 8px;">{pdesc}</p>
        <p style="font-size:13px;color:#c0ccd8;margin:0;">{preason}</p></div>""", unsafe_allow_html=True)

    if verdict in ('STRONG BUY','BUY'): bl = f"The math favors {ticker}. Strong fundamentals, good risk-return."
    elif verdict == 'HOLD': bl = f"{ticker} is fairly priced. Not bad, not a clear winner."
    elif verdict == 'UNDERPERFORM': bl = f"{ticker} isn't earning its risk premium. Better options exist."
    else: bl = f"Numbers don't support {ticker}. Index fund would likely outperform."
    st.markdown(f"""<div style="text-align:center;padding:20px;background:{vc}0a;border:1px solid {vc}18;border-radius:10px;margin-top:16px;">
    <p class="section-label">BOTTOM LINE</p>
    <p style="font-size:16px;font-weight:600;color:{vc};margin:8px 0 0;">{bl}</p></div>""", unsafe_allow_html=True)

with tab2:
    st.markdown("### ⚠️ Risk Assessment")
    st.markdown(f"AI-generated bear case for **{company_name}**.")
    if st.button("Generate Risk Assessment", key="risk_btn", type="primary"):
        with st.spinner("Researching risks... (30-45 seconds)"):
            result = call_claude(f"""Search the web for recent bear cases and risks about {company_name} ({ticker}).
Write a skeptical risk assessment covering:
1. ACCOUNTING & FINANCIAL RISKS: Red flags in reporting? Aggressive revenue recognition? Off-balance-sheet liabilities?
2. CUSTOMER & REVENUE CONCENTRATION: Dependent on few customers/products/geographies? What if biggest source shrinks 20%?
3. COMPETITIVE THREATS: Who's taking market share? Disruptive technologies? Name specific competitors.
End with BEAR CASE SUMMARY: If everything goes wrong in 2 years, what's the downside? Be specific with numbers.""")
            st.markdown(result)

with tab3:
    st.markdown("### 🔬 Deep Research Report")
    st.markdown(f"AI-generated analysis of **{company_name}**.")
    if st.button("Generate Deep Research", key="research_btn", type="primary"):
        with st.spinner("Deep research... (45-60 seconds)"):
            result = call_claude(f"""Search the web for comprehensive info about {company_name} ({ticker}, {sector}).
Write a research report:
1. BUSINESS MODEL: How do they make money? Revenue mix? Who pays them?
2. MOAT & COMPETITION: Top 3 competitors. Technological advantage? Rate: Strong/Moderate/Weak Moat.
3. CATALYSTS (12 months): Product launches, regulatory, partnerships? Street bullish or bearish?
4. ASYMMETRY CHECK: Realistic downside floor vs upside ceiling. Is risk/reward asymmetric? Priced for perfection or disaster?
Use real numbers and analyst targets.""")
            st.markdown(result)

st.markdown("---")
st.markdown('<p style="text-align:center;font-size:11px;color:#4e6380;">STOCKSIGHT — not financial advice, just math and AI</p>', unsafe_allow_html=True)
