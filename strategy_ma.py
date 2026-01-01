import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from strategy_quantx import get_data_robust, StrategyParams
import numpy as np
from datetime import datetime, timedelta

class MaCrossStrategy(Strategy):
    n_short = 5
    n_long = 20
    sl_pct = 0.15
    tp_pct = 0.30
    
    def init(self):
        close = pd.Series(self.data.Close)
        self.ma_short = self.I(lambda: close.rolling(self.n_short).mean(), name='MA_Short')
        self.ma_long = self.I(lambda: close.rolling(self.n_long).mean(), name='MA_Long')

    def next(self):
        price = self.data.Close[-1]

        # 進場：黃金交叉
        if not self.position:
            if crossover(self.ma_short, self.ma_long):
                self.buy()

        # 出場邏輯
        elif self.position:
            trade = self.trades[-1]
            
            # 1. 獨立停損
            if price <= trade.entry_price * (1 - self.sl_pct):
                self.position.close()
            # 2. 獨立停利
            elif price >= trade.entry_price * (1 + self.tp_pct):
                self.position.close()
            # 3. 死亡交叉
            elif crossover(self.ma_long, self.ma_short):
                self.position.close()

def run_ma_analysis(params: StrategyParams, n_short=5, n_long=20, sl_ma=0.15, tp_ma=0.30):
    req_start_dt = datetime.strptime(params.start_date, "%Y-%m-%d")
    req_end_dt = datetime.strptime(params.end_date, "%Y-%m-%d")
    fetch_start_dt = req_start_dt - timedelta(days=365)

    df_full = get_data_robust(params.ticker)
    if df_full.empty: return {}
    
    mask = (df_full.index >= fetch_start_dt) & (df_full.index <= req_end_dt)
    df = df_full[mask].copy()

    if df.empty or len(df) < 20: return {}

    # 設定參數
    MaCrossStrategy.n_short = n_short
    MaCrossStrategy.n_long = n_long
    MaCrossStrategy.sl_pct = sl_ma 
    MaCrossStrategy.tp_pct = tp_ma

    fee_rate = 0.001425 * (params.commission_discount / 10) + 0.0015
    
    try:
        bt = Backtest(df, MaCrossStrategy, cash=params.initial_cash, commission=fee_rate, trade_on_close=True)
        stats = bt.run()
    except:
        return {}

    equity_curve = stats['_equity_curve']['Equity']
    trades_df = stats['_trades']
    
    plot_mask = (equity_curve.index >= req_start_dt)
    df_vis = df[plot_mask]
    
    # 計算 MA 數據供前端繪圖
    ma5 = df['Close'].rolling(5).mean()
    ma10 = df['Close'].rolling(10).mean()
    ma20 = df['Close'].rolling(20).mean()

    equity_points = equity_curve[plot_mask].astype(float).tolist()
    time_points = [str(d.date()) for d in equity_curve[plot_mask].index]
    vol_points = df_vis['Volume'].tolist()

    trade_markers = []
    trade_list = []
    current_equity = params.initial_cash

    if not trades_df.empty:
        for i, row in trades_df.iterrows():
            if row['EntryTime'] < req_start_dt: continue
            
            ep, xp, sz = row['EntryPrice'], row['ExitPrice'], row['Size']
            cost = int(ep * sz * fee_rate + xp * sz * fee_rate) 
            pnl = int(row['PnL'])
            current_equity += pnl
            
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
                "equity": int(current_equity)
            })

    return {
        "stats": {
            "return": round(float(stats['Return [%]']), 2), 
            "mdd": round(float(stats['Max. Drawdown [%]']), 2),
            "win_rate": round(float(stats['Win Rate [%]']), 2), 
            "trades": int(stats['# Trades']),
            "sharpe": round(float(stats['Sharpe Ratio']), 2) if not np.isnan(stats['Sharpe Ratio']) else 0,
            "total_profit": int(equity_points[-1] - params.initial_cash) if equity_points else 0
        },
        "equity": equity_points,
        "time": time_points,
        "vol": vol_points,
        "ma5": ma5[plot_mask].fillna(0).tolist(),
        "ma10": ma10[plot_mask].fillna(0).tolist(),
        "ma20": ma20[plot_mask].fillna(0).tolist(),
        "trade_markers": trade_markers,
        "trade_list": trade_list
    }