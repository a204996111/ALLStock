import pandas as pd
from backtesting import Backtest, Strategy
from pydantic import BaseModel
import numpy as np
import sqlite3
import os
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, 'stocks.db')

class StrategyParams(BaseModel):
    ticker: str = "2330.TW"
    # ★★★ 修改點：預設日期調整 ★★★
    start_date: str = "2024-12-24" 
    end_date: str = "2025-12-24"
    
    n_breakout: int = 20
    vol_mult: float = 1.2
    close_pct: float = 0.85
    sl_pct: float = 0.15
    tp_pct: float = 0.30
    hold_days: int = 30
    
    initial_cash: int = 100000
    commission_discount: float = 10.0

class MyBreakoutStrategy(Strategy):
    n_breakout = 20
    vol_mult = 1.2
    close_pct = 0.85
    sl_pct = 0.15
    tp_pct = 0.30
    hold_days = 30
    
    def init(self):
        high = pd.Series(self.data.High)
        vol = pd.Series(self.data.Volume)
        close = pd.Series(self.data.Close)
        
        self.high_n = self.I(lambda: high.rolling(self.n_breakout).max().shift(1), name='High_N')
        self.vol_ma20 = self.I(lambda: vol.rolling(20).mean(), name='Vol_MA20')
        
        self.ma5 = self.I(lambda: close.rolling(5).mean(), name='MA5')
        self.ma10 = self.I(lambda: close.rolling(10).mean(), name='MA10')
        self.ma20 = self.I(lambda: close.rolling(20).mean(), name='MA20')

    def next(self):
        if len(self.high_n) < self.n_breakout or np.isnan(self.high_n[-1]): return
        price = self.data.Close[-1]
        high = self.data.High[-1]
        vol = self.data.Volume[-1]

        if not self.position:
            breakout = price > self.high_n[-1]
            vol_explode = vol > (self.vol_mult * self.vol_ma20[-1])
            strong_close = price > (self.close_pct * high)
            liquidity = vol > 200000 
            
            if breakout and vol_explode and strong_close and liquidity:
                self.buy()

        elif self.position:
            trade = self.trades[-1]
            days_held = (self.data.index[-1] - pd.Timestamp(trade.entry_time)).days
            
            if days_held >= self.hold_days:
                self.position.close()
                return

            if price <= trade.entry_price * (1 - self.sl_pct):
                self.position.close()
            elif price >= trade.entry_price * (1 + self.tp_pct):
                self.position.close()
            elif len(self.data) > 3:
                v1, v2, v3 = self.data.Volume[-1], self.data.Volume[-2], self.data.Volume[-3]
                m1, m2, m3 = self.vol_ma20[-1], self.vol_ma20[-2], self.vol_ma20[-3]
                if (v1 < 0.5*m1) and (v2 < 0.5*m2) and (v3 < 0.5*m3):
                    self.position.close()

def get_data_robust(ticker):
    if not os.path.exists(DB_FILE): return pd.DataFrame()
    code = ticker.split('.')[0]
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql(f"SELECT date, open, high, low, close, volume FROM stock_data WHERE code='{code}'", conn)
        conn.close()
        if not df.empty:
            for c in ['open','high','low','close','volume']: 
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce')
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            df = df[~df.index.duplicated(keep='last')]
            df.columns = [c.capitalize() for c in df.columns]
            return df
    except: conn.close()
    return pd.DataFrame()

def run_simple_analysis(params: StrategyParams, data_df=None):
    req_start_dt = datetime.strptime(params.start_date, "%Y-%m-%d")
    req_end_dt = datetime.strptime(params.end_date, "%Y-%m-%d")
    fetch_start_dt = req_start_dt - timedelta(days=365)

    df = pd.DataFrame()
    if data_df is not None: df = data_df.copy()
    else:
        df_full = get_data_robust(params.ticker)
        if df_full.empty: return {"error": "No Data", "ticker": params.ticker, "stats": {}, "trade_list": []}
        mask = (df_full.index >= fetch_start_dt) & (df_full.index <= req_end_dt)
        df = df_full[mask].copy()

    if df.empty or len(df) < params.n_breakout: 
        return {"error": "Data too short", "ticker": params.ticker, "stats": {}, "trade_list": []}

    MyBreakoutStrategy.n_breakout = params.n_breakout
    MyBreakoutStrategy.vol_mult = params.vol_mult
    MyBreakoutStrategy.close_pct = params.close_pct
    MyBreakoutStrategy.sl_pct = params.sl_pct
    MyBreakoutStrategy.tp_pct = params.tp_pct
    MyBreakoutStrategy.hold_days = params.hold_days

    fee_rate_buy = 0.001425 * (params.commission_discount / 10)
    fee_rate_tax_sim = fee_rate_buy + 0.0015
    
    try:
        bt = Backtest(df, MyBreakoutStrategy, cash=params.initial_cash, commission=fee_rate_tax_sim, trade_on_close=True)
        stats = bt.run()
    except:
        bt = Backtest(df, MyBreakoutStrategy, cash=params.initial_cash, commission=fee_rate_tax_sim, trade_on_close=True)
        stats = bt.run()

    trades_df = stats['_trades']
    trade_list = []
    trade_markers = []
    win_loss_ratio = 0
    current_equity = params.initial_cash
    total_profit = 0
    avg_return_pct = 0.0

    if not trades_df.empty:
        visible = trades_df[trades_df['EntryTime'] >= req_start_dt]
        if not visible.empty:
            w = visible[visible['PnL'] > 0]['PnL']
            l = visible[visible['PnL'] <= 0]['PnL'].abs()
            win_loss_ratio = round(w.mean() / l.mean(), 2) if not l.empty and l.mean() > 0 else (999 if not w.empty else 0)
            avg_return_pct = visible['ReturnPct'].mean() * 100

        for i, row in trades_df.iterrows():
            if row['EntryTime'] < req_start_dt: continue
            ep, xp, sz = row['EntryPrice'], row['ExitPrice'], row['Size']
            cost = int(ep * sz * fee_rate_buy + xp * sz * (fee_rate_buy + 0.003))
            pnl = int((xp - ep) * sz - cost)
            current_equity += pnl
            total_profit += pnl
            
            trade_markers.append({"date": str(row['EntryTime'].date()), "price": ep, "type": "buy"})
            trade_markers.append({"date": str(row['ExitTime'].date()), "price": xp, "type": "sell"})
            
            trade_list.append({
                "entry_date": str(row['EntryTime'].date()), 
                "entry_price": round(ep, 2),
                "exit_date": str(row['ExitTime'].date()), 
                "exit_price": round(xp, 2),
                "duration": f"{row['Duration'].days} 天", 
                "duration_days": row['Duration'].days,
                "cost": cost, 
                "pl": pnl,
                "return": round(row['ReturnPct']*100, 2), 
                "equity": current_equity
            })

    plot_mask = (df.index >= req_start_dt)
    df_vis = df[plot_mask]
    
    high_n = df['High'].rolling(params.n_breakout).max().shift(1)
    ma5 = df['Close'].rolling(5).mean()
    ma10 = df['Close'].rolling(10).mean()
    ma20 = df['Close'].rolling(20).mean()
    
    equity_curve = stats['_equity_curve']['Equity']
    if len(equity_curve) > len(df): equity_curve = equity_curve.iloc[-len(df):]
    equity_points = equity_curve[plot_mask].astype(float).tolist()

    return {
        "ticker": params.ticker,
        "time": [str(d.date()) for d in df_vis.index],
        "open": df_vis['Open'].tolist(), 
        "high": df_vis['High'].tolist(), 
        "low": df_vis['Low'].tolist(), 
        "close": df_vis['Close'].tolist(),
        "vol": df_vis['Volume'].tolist(), 
        "ma5": ma5[plot_mask].fillna(0).tolist(), 
        "ma10": ma10[plot_mask].fillna(0).tolist(), 
        "ma20": ma20[plot_mask].fillna(0).tolist(), 
        "breakout_line": high_n[plot_mask].fillna(0).tolist(),
        "stats": {
            "return": round(float(stats['Return [%]']), 2), 
            "mdd": round(float(stats['Max. Drawdown [%]']), 2),
            "sharpe": round(float(stats['Sharpe Ratio']), 2) if not np.isnan(stats['Sharpe Ratio']) else 0,
            "win_rate": round(float(stats['Win Rate [%]']), 2), 
            "trades": len(trade_list),
            "win_loss_ratio": win_loss_ratio, 
            "equity": equity_points,
            "total_profit": int(total_profit),
            "avg_return": round(avg_return_pct, 2)
        },
        "trade_markers": trade_markers, "trade_list": trade_list
    }