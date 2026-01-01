import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from fastapi.responses import HTMLResponse
from strategy_quantx import run_simple_analysis, StrategyParams
from strategy_ma import run_ma_analysis
from strategy_core import BatchStrategy 
from backtesting import Backtest
import uvicorn
from datetime import datetime, timedelta
import sqlite3
import time

app = FastAPI()
DB_FILE = os.path.join(BASE_DIR, 'stocks.db')
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
RANKING_FILE = os.path.join(BASE_DIR, 'strategy_ranking.csv')
TRADES_FILE = os.path.join(BASE_DIR, 'all_trades.csv')

# 首頁預設參數
CURRENT_CONFIG = {
    "n_breakout": 20, "vol_mult": 1.2, "close_pct": 85.0, "vol_min": 200,
    "sl_pct": 15.0, "tp_pct": 30.0, "hold_days": 30, "discount": 10.0,
    "last_run_time": 0
}

NAME_MAP = {}
if os.path.exists(RANKING_FILE):
    try:
        df = pd.read_csv(RANKING_FILE, encoding='utf-8-sig')
        if 'Code' in df.columns:
            NAME_MAP = dict(zip(df['Code'].astype(str), df['Name']))
    except: pass

def get_latest_date_from_db():
    if not os.path.exists(DB_FILE): return datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_FILE)
    try:
        cursor = conn.execute("SELECT MAX(date) FROM stock_data")
        latest = cursor.fetchone()[0]
        if latest: return latest
        return datetime.now().strftime("%Y-%m-%d")
    except: return datetime.now().strftime("%Y-%m-%d")
    finally: conn.close()

def run_backtest_engine(params):
    start_t = time.time()
    conn = sqlite3.connect(DB_FILE)
    try:
        cursor = conn.execute("SELECT DISTINCT code, name FROM stock_data")
        codes = cursor.fetchall()
        cursor = conn.execute("SELECT MAX(date) FROM stock_data")
        max_date = cursor.fetchone()[0]
    except: return
    finally: conn.close()

    db_max_dt = datetime.strptime(max_date, '%Y-%m-%d')
    report_start_dt = db_max_dt - timedelta(days=365)
    fetch_start_dt = report_start_dt - timedelta(days=365)
    fetch_start_str = fetch_start_dt.strftime('%Y-%m-%d')

    BatchStrategy.n_breakout = params['n_breakout']
    BatchStrategy.vol_mult = params['vol_mult']
    BatchStrategy.close_pct = params['close_pct'] / 100.0
    BatchStrategy.vol_min = params['vol_min'] * 1000
    BatchStrategy.sl_pct = params['sl_pct'] / 100.0
    BatchStrategy.tp_pct = params['tp_pct'] / 100.0
    BatchStrategy.hold_days = params['hold_days']

    fee_base = 0.001425 * (params['discount'] / 10.0)
    final_commission = fee_base + 0.0015
    initial_cash = 100000

    all_single_trades = []
    
    for i, (code, name) in enumerate(codes):
        try:
            conn = sqlite3.connect(DB_FILE)
            query = f"SELECT date, open, high, low, close, volume FROM stock_data WHERE code='{code}' AND date >= '{fetch_start_str}'"
            df = pd.read_sql(query, conn)
            conn.close()
            if df.empty or len(df) < 120: continue
            cols = ['open', 'high', 'low', 'close', 'volume']
            for col in cols: df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
            df.dropna(inplace=True)
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values('date', inplace=True)
            df.set_index('date', inplace=True)
            df = df[~df.index.duplicated(keep='first')]
            df.columns = [c.capitalize() for c in df.columns]

            bt = Backtest(df, BatchStrategy, cash=initial_cash, commission=final_commission, trade_on_close=True)
            stats = bt.run()
            
            trades = stats['_trades']
            if not trades.empty:
                trades = trades.sort_values('EntryTime')
                trades['CumPnL'] = trades['PnL'].cumsum()
                trades['Equity'] = initial_cash + trades['CumPnL']

                valid_trades = trades[trades['EntryTime'] >= report_start_dt].copy()
                if not valid_trades.empty:
                    for idx, row in valid_trades.iterrows():
                        duration = (row['ExitTime'] - row['EntryTime']).days
                        ep = row['EntryPrice']
                        xp = row['ExitPrice']
                        sz = row['Size']
                        cost = (ep * sz + xp * sz) * final_commission 
                        pnl = (xp - ep) * sz - cost
                        ret_pct = row['ReturnPct'] * 100
                        
                        all_single_trades.append({
                            'Code': code, 'Name': name,
                            'EntryTime': row['EntryTime'].strftime('%Y-%m-%d'),
                            'ExitTime': row['ExitTime'].strftime('%Y-%m-%d'),
                            'EntryPrice': round(row['EntryPrice'], 2),
                            'ExitPrice': round(row['ExitPrice'], 2),
                            'Duration': duration,
                            'PnL': int(pnl),
                            'ReturnPct': round(ret_pct, 2),
                            'Cost': int(cost),
                            'Equity': int(row['Equity'])
                        })
        except: continue

    if all_single_trades:
        df_rank = pd.DataFrame(all_single_trades)
        df_rank = df_rank.sort_values(by='EntryTime', ascending=False)
        df_rank.to_csv(RANKING_FILE, index=False, encoding='utf-8-sig')
        df_rank.to_csv(TRADES_FILE, index=False, encoding='utf-8-sig')
    else:
        cols = ['Code','Name','EntryTime','ExitTime','EntryPrice','ExitPrice','Duration','PnL','ReturnPct','Cost','Equity']
        pd.DataFrame(columns=cols).to_csv(RANKING_FILE, index=False)
    
    CURRENT_CONFIG['last_run_time'] = round(time.time() - start_t, 2)

@app.get("/run_batch")
async def run_batch(
    n_breakout: int = 20, vol_mult: float = 1.2, close_pct: float = 85.0, vol_min: int = 200,
    sl_pct: float = 15.0, tp_pct: float = 30.0, hold_days: int = 30, discount: float = 10.0
):
    global CURRENT_CONFIG
    CURRENT_CONFIG = {
        "n_breakout": n_breakout, "vol_mult": vol_mult, "close_pct": close_pct, "vol_min": vol_min,
        "sl_pct": sl_pct, "tp_pct": tp_pct, "hold_days": hold_days, "discount": discount,
        "last_run_time": 0
    }
    run_backtest_engine(CURRENT_CONFIG)
    return RedirectResponse(url="/", status_code=303)

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    rankings = []
    summary = {
        'initial_capital': 100000, 'total_return_pct': 0, 'total_profit': 0, 'win_rate': 0,
        'avg_return': 0, 'win_loss_ratio': 0, 'sharpe': 0, 'total_trades': 0, 'max_drawdown': 0,
        'equity_curve': [], 'dates': [], 'date_range_str': '無資料', 'start_date': '', 'end_date': ''
    }

    # ★★★ 修改點：強制將結束日期預設為 2025-12-24 ★★★
    fixed_end_date = datetime(2025, 12, 24)
    end_date_str = fixed_end_date.strftime("%Y-%m-%d")
    start_date_str = (fixed_end_date - timedelta(days=365)).strftime("%Y-%m-%d")
    
    summary['date_range_str'] = f"{start_date_str} ~ {end_date_str}"
    summary['start_date'] = start_date_str
    summary['end_date'] = end_date_str

    if os.path.exists(RANKING_FILE):
        try:
            df_rank = pd.read_csv(RANKING_FILE, encoding='utf-8-sig').fillna(0)
            if not df_rank.empty:
                rankings = df_rank.to_dict(orient='records')
                summary['total_trades'] = len(df_rank)
                summary['total_profit'] = int(df_rank['PnL'].sum())
                summary['total_return_pct'] = round((summary['total_profit'] / 100000) * 100, 2)
                wins = df_rank[df_rank['PnL'] > 0]
                losses = df_rank[df_rank['PnL'] <= 0]
                if len(df_rank) > 0: summary['win_rate'] = round((len(wins) / len(df_rank)) * 100, 2)
                summary['avg_return'] = round(df_rank['ReturnPct'].mean(), 2)
                avg_win = wins['PnL'].mean() if not wins.empty else 0
                avg_loss = losses['PnL'].abs().mean() if not losses.empty else 0
                if avg_loss > 0: summary['win_loss_ratio'] = round(avg_win / avg_loss, 2)
                elif avg_win > 0: summary['win_loss_ratio'] = 999
                
                df_sorted = df_rank.sort_values('ExitTime')
                df_sorted['CumPnL'] = df_sorted['PnL'].cumsum()
                equity_series = 100000 + df_sorted['CumPnL']
                
                peak = equity_series.cummax()
                dd = (equity_series - peak) / peak
                summary['max_drawdown'] = round(dd.min() * 100, 2) if not dd.empty else 0
                returns = equity_series.pct_change().dropna()
                if not returns.empty and returns.std() != 0:
                    summary['sharpe'] = round((returns.mean() / returns.std()) * np.sqrt(252), 2)
                
                summary['dates'] = df_sorted['ExitTime'].tolist()
                summary['equity_curve'] = df_sorted['CumPnL'].tolist()
        except Exception as e: print(f"Error: {e}")

    return templates.TemplateResponse("ranking.html", {
        "request": request, "rankings": rankings, "trades": rankings, "summary": summary,
        "conf": CURRENT_CONFIG
    })

@app.get("/analysis", response_class=HTMLResponse)
async def analysis(
    request: Request, ticker: str = "2330.TW", start: str = None, end: str = None, 
    n_breakout: int = 20, vol_mult: float = 1.2, close_pct: float = 0.85, 
    sl: float = 15.0, tp: float = 30.0, hold_days: int = 30,
    init_cash: int = 100000, discount: float = 10.0,
    focus_entry: str = None, focus_exit: str = None
):
    try:
        # ★★★ 修改點：分析頁預設日期強制鎖定為 12/24 ★★★
        if end is None: end = "2025-12-24"
        if start is None: 
            # 依據 end 往前推 365 天
            end_dt = datetime.strptime(end, "%Y-%m-%d")
            start = (end_dt - timedelta(days=365)).strftime("%Y-%m-%d")
        
        params = StrategyParams(
            ticker=ticker, start_date=start, end_date=end,
            n_breakout=n_breakout, vol_mult=vol_mult, close_pct=close_pct, 
            sl_pct=sl/100.0, tp_pct=tp/100.0, hold_days=hold_days,
            initial_cash=init_cash, commission_discount=discount
        )
        perf = run_simple_analysis(params)
        code_only = ticker.split('.')[0]
        stock_name = NAME_MAP.get(code_only, "")
        actual_ticker = perf.get('ticker', ticker)
        
        return templates.TemplateResponse("analysis.html", { 
            "request": request, **params.model_dump(), "perf": perf, 
            "stock_name": stock_name, "actual_ticker": actual_ticker,
            "sl": sl, "tp": tp,
            "focus_entry": focus_entry, "focus_exit": focus_exit 
        })
    except Exception as e: return HTMLResponse(content=f"Error: {str(e)}", status_code=500)

@app.get("/comparison", response_class=HTMLResponse)
async def comparison(
    request: Request, ticker: str = "2330.TW", start: str = None, end: str = None,
    n_breakout: int = 20, vol_mult: float = 1.2, close_pct: float = 0.85, 
    sl: float = 15.0, tp: float = 30.0, hold_days: int = 30,
    init_cash: int = 100000, discount: float = 10.0,
    ma_short: int = 5, ma_long: int = 20,
    sl_ma: float = 15.0, tp_ma: float = 30.0
):
    try:
        # ★★★ 修改點：比較頁預設日期強制鎖定為 12/24 ★★★
        if end is None: end = "2025-12-24"
        if start is None:
            end_dt = datetime.strptime(end, "%Y-%m-%d")
            start = (end_dt - timedelta(days=365)).strftime("%Y-%m-%d")

        params = StrategyParams(
            ticker=ticker, start_date=start, end_date=end,
            n_breakout=n_breakout, vol_mult=vol_mult, close_pct=close_pct, 
            sl_pct=sl/100.0, tp_pct=tp/100.0, hold_days=hold_days,
            initial_cash=init_cash, commission_discount=discount
        )
        
        perf_a = run_simple_analysis(params)
        perf_b = run_ma_analysis(params, n_short=ma_short, n_long=ma_long, sl_ma=sl_ma/100.0, tp_ma=tp_ma/100.0)
        
        code_only = ticker.split('.')[0]
        stock_name = NAME_MAP.get(code_only, "")
        
        return templates.TemplateResponse("comparison.html", {
            "request": request, "stock_name": stock_name, "ticker": ticker,
            "strat1": perf_a, "strat2": perf_b,
            "params": params.model_dump(),
            "ma_short": ma_short, "ma_long": ma_long,
            "sl_ma": sl_ma, "tp_ma": tp_ma
        })
    except Exception as e: return HTMLResponse(content=f"Error: {str(e)}", status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)