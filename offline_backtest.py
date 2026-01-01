import sqlite3
import pandas as pd
from backtesting import Backtest
from strategy_core import BatchStrategy 
import time
import warnings
import os
from datetime import datetime, timedelta
import numpy as np

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, 'stocks.db')
RANKING_FILE = os.path.join(BASE_DIR, 'strategy_ranking.csv')
TRADES_FILE = os.path.join(BASE_DIR, 'all_trades.csv')

def get_db_info():
    if not os.path.exists(DB_FILE): return [], None
    conn = sqlite3.connect(DB_FILE)
    try:
        cursor = conn.execute("SELECT DISTINCT code, name FROM stock_data")
        codes = cursor.fetchall()
        cursor = conn.execute("SELECT MAX(date) FROM stock_data")
        max_date = cursor.fetchone()[0]
    except: return [], None
    finally: conn.close()
    return codes, max_date

def load_data(code, start_date):
    conn = sqlite3.connect(DB_FILE)
    query = f"SELECT date, open, high, low, close, volume FROM stock_data WHERE code='{code}' AND date >= '{start_date}'"
    df = pd.read_sql(query, conn)
    conn.close()
    if df.empty: return None
    cols = ['open', 'high', 'low', 'close', 'volume']
    for col in cols: df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    df.dropna(inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    df.columns = [c.capitalize() for c in df.columns]
    return df

def main():
    print("🚀 開始離線運算 (trade_on_close=True + 0.6% 成本)...")
    
    codes, db_max_date_str = get_db_info()
    if not codes: return

    db_max_dt = datetime.strptime(db_max_date_str, '%Y-%m-%d')
    report_start_dt = db_max_dt - timedelta(days=365)
    fetch_start_dt = report_start_dt - timedelta(days=365)
    fetch_start_str = fetch_start_dt.strftime('%Y-%m-%d')

    all_single_trades = []
    start_time = time.time()
    
    # 預設參數 (與 main.py 保持一致)
    commission_rate = 0.002925 # 約等於 10 折扣的手續費 + 稅
    initial_cash = 100000

    for i, (code, name) in enumerate(codes):
        if i % 50 == 0: print(f"[{i+1}/{len(codes)}] 進度...", end="\r")
        try:
            df = load_data(code, fetch_start_str)
            if df is None or len(df) < 120: continue
            
            # 啟用 trade_on_close=True
            bt = Backtest(df, BatchStrategy, cash=initial_cash, commission=commission_rate, trade_on_close=True)
            stats = bt.run()
            
            trades = stats['_trades']
            if not trades.empty:
                # 計算每筆交易的累積餘額 (針對該檔股票的回測)
                trades = trades.sort_values('EntryTime') # 確保按時間排序
                trades['CumPnL'] = trades['PnL'].cumsum()
                trades['Equity'] = initial_cash + trades['CumPnL']

                valid_trades = trades[trades['EntryTime'] >= report_start_dt].copy()
                if not valid_trades.empty:
                    for idx, row in valid_trades.iterrows():
                        duration = (row['ExitTime'] - row['EntryTime']).days
                        pnl = row['PnL']
                        ret_pct = row['ReturnPct'] * 100
                        
                        # 計算單筆成本 (概算)
                        ep = row['EntryPrice']
                        xp = row['ExitPrice']
                        sz = row['Size']
                        cost = (ep * sz + xp * sz) * commission_rate

                        all_single_trades.append({
                            'Code': code, 'Name': name,
                            'EntryTime': row['EntryTime'].strftime('%Y-%m-%d'),
                            'ExitTime': row['ExitTime'].strftime('%Y-%m-%d'),
                            'EntryPrice': round(row['EntryPrice'], 2),
                            'ExitPrice': round(row['ExitPrice'], 2),
                            'Duration': duration,
                            'PnL': int(pnl),
                            'ReturnPct': round(ret_pct, 2),
                            'Cost': int(cost),            # 新增: 成本
                            'Equity': int(row['Equity'])  # 新增: 餘額
                        })

        except Exception: continue

    if all_single_trades:
        df_rank = pd.DataFrame(all_single_trades)
        
        # 依照出場時間排序 (為了讓餘額看起來有時間順序，雖然這是多檔股票混在一起)
        # 或者依照使用者習慣的 PnL 排序
        df_rank = df_rank.sort_values(by='PnL', ascending=False)
        
        df_rank.to_csv(RANKING_FILE, index=False, encoding='utf-8-sig')
        df_rank.to_csv(TRADES_FILE, index=False, encoding='utf-8-sig')
    else:
        # 更新 Header 包含新欄位
        cols = ['Code','Name','EntryTime','ExitTime','EntryPrice','ExitPrice','Duration','PnL','ReturnPct','Cost','Equity']
        pd.DataFrame(columns=cols).to_csv(RANKING_FILE, index=False)
    
    print(f"\n✨ 運算完成！耗時 {time.time() - start_time:.1f} 秒")

if __name__ == "__main__":
    main()