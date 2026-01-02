import sqlite3
import os

# 設定
OLD_DB = "stocks.db"
NEW_DB = "stocks_small.db"
START_DATE = "2015-01-01"  # ★設定只保留 2015 年以後的資料 (這樣就夠回測10年了)

def shrink():
    if not os.path.exists(OLD_DB):
        print(f"❌ 找不到 {OLD_DB}，請確認檔案位置！")
        return

    print(f"🚀 開始瘦身... 保留 {START_DATE} 之後的資料")
    
    # 連結舊資料庫
    conn_old = sqlite3.connect(OLD_DB)
    cursor_old = conn_old.cursor()
    
    # 連結新資料庫
    if os.path.exists(NEW_DB): os.remove(NEW_DB)
    conn_new = sqlite3.connect(NEW_DB)
    cursor_new = conn_new.cursor()
    
    # 1. 複製資料表結構 (schema)
    print("📦 複製資料表結構...")
    cursor_old.execute("SELECT sql FROM sqlite_master WHERE type='table'")
    for (sql,) in cursor_old.fetchall():
        if sql: cursor_new.execute(sql)
    
    # 2. 搬移資料 (只搬 2015 之後的)
    print("🚚 搬移資料中 (這可能需要幾秒鐘)...")
    # 假設您的資料表叫做 stock_data 且有 date 欄位
    try:
        cursor_new.execute(f"ATTACH DATABASE '{OLD_DB}' AS old_db")
        cursor_new.execute(f"INSERT INTO stock_data SELECT * FROM old_db.stock_data WHERE date >= '{START_DATE}'")
        conn_new.commit()
        print("✅ 資料搬移完成！")
    except Exception as e:
        print(f"⚠️ 錯誤: {e}")
        # 如果失敗，嘗試直接複製所有資料 (備案)
        # cursor_new.execute(f"INSERT INTO stock_data SELECT * FROM old_db.stock_data")
    
    # 3. 檢查大小
    old_size = os.path.getsize(OLD_DB) / (1024*1024)
    new_size = os.path.getsize(NEW_DB) / (1024*1024)
    
    print(f"\n✨ 瘦身成果：")
    print(f"原本: {old_size:.2f} MB")
    print(f"現在: {new_size:.2f} MB")
    print(f"縮小了: {old_size - new_size:.2f} MB 🎉")
    
    conn_old.close()
    conn_new.close()

if __name__ == "__main__":
    shrink()
