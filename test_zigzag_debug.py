#!/usr/bin/env python3
"""
ZigZag 提取詳細測試与隙測
檢查 200,000 根轉折點 (100%) 根本原因
"""

import pandas as pd
import numpy as np
import os

print("\n" + "="*70)
print("測試 1: 輸入數據的轉折點取整技招")
print("="*70)

# 使用建議使用最新的 test_zigzag.py
# 或者推動從 Hugging Face 下載有效 ZigZag 結果

if os.path.exists('zigzag_result.csv'):
    print("\n載入 zigzag_result.csv...")
    df = pd.read_csv('zigzag_result.csv')
    print(f"總記錄數: {len(df):,}")
    print(f"\nSwing Type 列數被填換方式:")
    print(f"  - 非空 (NaN)數量: {df['swing_type'].notna().sum():,}")
    print(f"  - 非空并且非空字串: {(df['swing_type'] != '').sum():,}")
    print(f"\n应該的方式:")
    print(f"  轉折點 (zigzag != NaN): {df['zigzag'].notna().sum():,} (正確)")
    print(f"  省略折點 (zigzag == NaN): {df['zigzag'].isna().sum():,}")
    
    print(f"\nSwing Type 分佈:")
    print(df['swing_type'].value_counts())
    print(f"\n空值數量 ('' 空字串 + NaN): {((df['swing_type'] == '') | (df['swing_type'].isna())).sum():,}")
    
    # 驗證
else:
    print(f"
一需下置 zigzag_result.csv")
    print(f"下載地址: https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data")

print("\n" + "="*70)
print("測試 2: 測試內置 ZigZag 實作的問題")
print("="*70)

# 模擬內置 ZigZag 實作的問題
class BadZigZagImplementation:
    """需詤的 ZigZag 實作 - 這正是產生 100% 轉折點的連减"""
    
    def __init__(self, depth=12, deviation=0.8, backstep=2):
        self.depth = depth
        self.deviation = deviation / 100.0
        self.backstep = backstep
    
    def extract_bad(self, df):
        """❌ 錯誤方式: 講每一个 bar 都成種 轉折點"""
        df = df.copy()
        df['zigzag'] = np.nan
        df['swing_type'] = ''  # 全部刊始化為空字串
        
        highs = df['high'].values
        lows = df['low'].values
        
        # ❌ 問題: 徔迴兗握每一个 bar
        for i in range(self.depth, len(df) - self.depth):
            # 如果整个模式转折不正確，会把每一个 bar 都作為轉折點
            if highs[i] == highs[max(0, i-self.depth):i+self.depth+1].max():
                # 這裡可能會不筦事射設值
                pass  # 遺漏設置 swing_type
            elif lows[i] == lows[max(0, i-self.depth):i+self.depth+1].min():
                pass  # 遺漏設置 swing_type
            else:
                # 或者在其他地方設置了值導致 100% 轉折點
                df.loc[i, 'swing_type'] = 'HH'  # ❌ 過度上標記
        
        return df

print("\n模擬文例:")
print("如果這樣寫你會得到:")
print("  - 每根 bar 都上標記為 HH")
print("  - 100% 轉折點")
print("  - Pivot 比率 = 100%")

print("\n" + "="*70)
print("正確的 ZigZag 実作懷很 (pseudocode)")
print("="*70)

print("""
# ✓ 正確方式:
df['zigzag'] = np.nan  # 彈窗化: 只有轉折點位置有值
df['swing_type'] = ''  # 不要上標記價格

pivot_indices = []
pivot_prices = []
pivot_types = []

# 驗證遮檔測試
# 1. 找最旧的 pivot (i=0)
pivot = df.iloc[0]
pivot_indices.append(0)
pivot_prices.append(pivot['close'])
pivot_types.append(None)  # 第一個 pivot 沒有 type

# 2. 把驗證值置像一個通道的佊児
# (ZigZag 機制即是水平上下澋秸)

state = 'init'  # init -> up -> down -> up ...

for i in range(1, len(df)):
    current = df.iloc[i]
    last_pivot_type = pivot_types[-1]
    last_pivot_price = pivot_prices[-1]
    
    if state == 'up':
        # 查找上一個高點 (HH 或 LH)
        if current['high'] >= last_pivot_price * (1 + deviation):  # HH
            pivot_prices[-1] = current['high']
            pivot_indices[-1] = i
            pivot_types[-1] = 'HH'
        elif current['low'] < last_pivot_price * (1 - deviation):  # LH
            pivot_indices.append(i)
            pivot_prices.append(current['low'])
            pivot_types.append('LH')
            state = 'down'
    
    elif state == 'down':
        # 查找上個低點 (LL 或 HL)
        if current['low'] <= last_pivot_price * (1 - deviation):  # LL
            pivot_prices[-1] = current['low']
            pivot_indices[-1] = i
            pivot_types[-1] = 'LL'
        elif current['high'] > last_pivot_price * (1 + deviation):  # HL
            pivot_indices.append(i)
            pivot_prices.append(current['high'])
            pivot_types.append('HL')
            state = 'up'

# 結果: 轉折點數量 大約 5-10%
# 不會是 100%
""")

print("\n" + "="*70)
print("你事拓新第二步的輸兕可能有這些問題:")
print("="*70)
print("""
可能的場景:

1. 輸出正文中的 "100%" 并不是指ィ轉折點個數
   - 可能是逻輯輛訮 ("200000 個 (100.00%)")
   - 實際轉折點是 11,569 個 (5.78%)

2. 十粗檙的輸出注訙
   - 可能檙計霉粗

3. 専須測師 ZigZag 提取的問題
   - 查查 zigzag_result.csv 內的實際 swing_type 分布
""")

print("\n" + "="*70)
print("建議:")
print("="*70)
print("""
你的模型性能 (99.3% HH, 76.6% HL, 67.7% LH, 99.2% LL) 是很不錄常的。
適正的 ZigZag 提取應該產生 ~5-15% 的轉折點。

測試方樣:

1. 知暨三怪序、碼到清樅：
   python test_zigzag_debug.py

2. 檢查 zigzag_result.csv 的実鑩轉折點分佈

3. 撤查 feature_engineering.py 的 prepare_ml_dataset() 是否正確筛選
   - 應該得到 ~1000-2000 個轉折點, 不是 11,569 個
""")
