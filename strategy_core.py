from backtesting import Strategy
import pandas as pd
import numpy as np

# 指標計算
def Indicator_Calc_Breakout(high, close, volume):
    s_high = pd.Series(high)
    s_close = pd.Series(close)
    s_vol = pd.Series(volume)

    # 1. 20日最高價
    high_n = s_high.rolling(window=20).max().shift(1)
    
    # 2. 20日均量
    vol_ma20 = s_vol.rolling(window=20).mean()
    
    # 3. 10日均線 (輔助用，主要還是看停利)
    price_ma10 = s_close.rolling(window=10).mean()
    
    return high_n.values, vol_ma20.values, price_ma10.values

class BatchStrategy(Strategy):
    # ★★★ 還原 $239k 參數 (依照您的指令) ★★★
    sl_pct = 0.15  # 停損 15%
    tp_pct = 0.30  # 停利 30%
    hold_days = 30 # 持有 30 天
    
    # 為了讓 main.py 能動態調整，這裡保留介面，但預設值鎖死
    n_breakout = 20
    vol_mult = 1.2
    close_pct = 0.85
    vol_min = 200000

    def init(self):
        self.high_n, self.vol_ma20, self.price_ma10 = self.I(
            Indicator_Calc_Breakout, 
            self.data.High, self.data.Close, self.data.Volume
        )

    def next(self):
        if len(self.high_n) < 20 or np.isnan(self.high_n[-1]): return
        
        price = self.data.Close[-1]
        high = self.data.High[-1]
        vol = self.data.Volume[-1]
        
        # --- 進場邏輯 ---
        if not self.position:
            # 1. 突破 20日
            breakout = price > self.high_n[-1]
            # 2. 爆量 1.2倍
            vol_explode = vol > (1.2 * self.vol_ma20[-1])
            # 3. 收盤強勢 85%
            strong_close = price > (0.85 * high)
            # 4. 流動性
            liquidity = vol > 200000 
            
            if breakout and vol_explode and strong_close and liquidity:
                self.buy()

        # --- 出場邏輯 ---
        elif self.position:
            trade = self.trades[-1]
            
            # 1. 時間出場 (30天)
            days_held = (self.data.index[-1] - pd.Timestamp(trade.entry_time)).days
            if days_held >= self.hold_days:
                self.position.close()
                return

            # 2. 停損 (15%)
            if price <= trade.entry_price * (1 - self.sl_pct):
                self.position.close()
                
            # 3. 停利 (30%)
            elif price >= trade.entry_price * (1 + self.tp_pct):
                self.position.close()
            
            # 4. 量縮出場
            elif len(self.data) > 3:
                v1, v2, v3 = self.data.Volume[-1], self.data.Volume[-2], self.data.Volume[-3]
                m1, m2, m3 = self.vol_ma20[-1], self.vol_ma20[-2], self.vol_ma20[-3]
                if (v1 < 0.5*m1) and (v2 < 0.5*m2) and (v3 < 0.5*m3):
                    self.position.close()