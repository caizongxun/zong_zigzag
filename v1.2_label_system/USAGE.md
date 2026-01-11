# v1.2 標籤系統使用指南

## 第一步: 安裝依賴

```bash
pip install pandas numpy pyyaml huggingface-hub scikit-learn
```

## 第二步: 測試BTC 15m

### 方法1: 直接運行測試腳本

```bash
cd v1.2_label_system
python test_btc_15m.py
```

這會:
1. 從HuggingFace下載BTCUSDT 15m數據
2. 計算所有技術指標
3. 生成三層標籤
4. 打印詳細的統計報告
5. 保存報告到 `output/BTCUSDT_15m_report.json`

### 方法2: 在Python中運行

```python
import sys
sys.path.insert(0, 'v1.2_label_system')

from label_generator import LabelGenerator
from label_statistics import LabelStatistics

# 初始化
generator = LabelGenerator('v1.2_label_system/config.yaml')

# 生成標籤
df = generator.generate_labels(
    symbol='BTCUSDT',
    timeframe='15m',
    save_path=None  # 不保存到文件,直接返回DataFrame
)

# 查看數據
print(df.shape)
print(df.head())

# 生成統計報告
report = LabelStatistics.generate_full_report(df, 'BTCUSDT', '15m')
LabelStatistics.print_report(report)
```

## 理解輸出結果

### 數據概覽

```
Total candles: 3000
Date range: 2024-01-01 12:00:00 to 2024-12-31 23:45:00
Price range: 42130.50 - 98700.25
```

### 進場候選統計

```
Entry candidates: 156 (5.20%)

Entry Success Breakdown:
  Successful: 98 (62.82%)    <- 有利潤的進場點
  Failed: 58 (37.18%)        <- 虧損的進場點

Entry Quality Score:
  Mean: 45.32               <- 平均品質分數
  Median: 42.15
  Min: 10.20
  Max: 98.75
```

### 進場原因分布

```
Entry Reason Distribution:
  fib: 95 (60.90%)          <- 接近Fibonacci級別的進場
  bb: 61 (39.10%)           <- 接近Bollinger Band的進場
```

### 最優進場回報

```
Optimal Entry Return:
  Mean: 2.45%               <- 平均最優進場能獲利2.45%
  Median: 1.82%
  Min: -5.30%
  Max: 12.80%
  Profitable: 142 (91.03%)  <- 91%的進場點最後都有利潤
```

## 標籤說明

### 進場候選 (is_entry_candidate)

**值**: True/False

**含義**: 該根K棒是否符合進場條件

**觸發條件**:
1. 價格接近某個Fibonacci級別 (距離 < 1%)
2. 或者價格接近Bollinger Band的上下軌 (距離 < 1%)

**用途**: 找到潛在的進場點

### 進場成功標籤 (entry_success)

**值**: 0 或 1

**含義**:
- 1 = 進場後20根K棒內有利潤
- 0 = 進場後20根K棒內虧損

**利潤判定標準**: 最大獲利 > 最大虧損 × 1.5倍

**用途**: 訓練"這個進場點好嗎"的分類器 (XGBoost/RandomForest)

### 進場品質分數 (entry_quality_score)

**值**: 0-100連續值

**構成**:
- 40分: 最大獲利幅度 (%) 
- 30分: 最小回撤控制 (倒數,越小越好)
- 20分: 風險報酬比 (最大獲利 / 最大虧損)
- 10分: 勝率 (向上破價次數)

**解釋**:
- < 30: 低品質進場點,風險高
- 30-60: 中等品質,可以進場
- 60-80: 好進場點,優先考慮
- 80+: 最優進場點,當然選擇

**用途**: 排序和過濾進場點

### 最優進場價 (optimal_entry_price)

**值**: 浮點數 (價格)

**含義**: 在進場候選區間內,歷史上最好的進場價格

**對多單**: 該區間的最低價 (買得最便宜)
**對空單**: 該區間的最高價 (賣得最貴)

**用途**: 
1. 訓練"最優進場價在哪裡"的迴歸模型
2. 計算Stop Loss和Take Profit級別
3. 評估進場點的品質

### 最優進場回報 (optimal_entry_return)

**值**: 小數值 (e.g., 0.0245 = 2.45%)

**含義**: 使用optimal_entry_price進場後,向前20根K棒內的最大可達回報

**解釋**:
- 正值: 如果按最優價進場,可以獲利
- 負值: 即使按最優價進場,也可能虧損
- 越大越好: 回報潛力越大

**用途**: 評估進場點的獲利潛力

## 常見用途

### 1. 找出最佳進場點

```python
candidates = df[df['is_entry_candidate'] == True]

# 篩選高品質進場點
quality_entries = candidates[candidates['entry_quality_score'] > 70]

# 篩選成功的進場點
successful = candidates[candidates['entry_success'] == 1]

# 組合篩選
best_entries = candidates[
    (candidates['entry_quality_score'] > 70) &
    (candidates['entry_success'] == 1) &
    (candidates['optimal_entry_return'] > 0.02)  # 至少2%回報
]

print(f"Found {len(best_entries)} best entry points")
```

### 2. 分析進場原因

```python
candidates = df[df['is_entry_candidate'] == True]

# 哪種原因的進場更有效?
fib_entries = candidates[candidates['entry_reason'] == 'fib']
bb_entries = candidates[candidates['entry_reason'] == 'bb']

print(f"Fib success rate: {(fib_entries['entry_success']==1).sum() / len(fib_entries) * 100:.2f}%")
print(f"BB success rate: {(bb_entries['entry_success']==1).sum() / len(bb_entries) * 100:.2f}%")
```

### 3. 計算策略回報

```python
candidates = df[df['is_entry_candidate'] == True]

# 假設我們只在quality_score > 50時進場
traded = candidates[candidates['entry_quality_score'] > 50]

# 計算勝率
win_rate = (traded['entry_success'] == 1).sum() / len(traded) * 100

# 計算平均回報
avg_return = traded['optimal_entry_return'].mean() * 100

# 計算盈虧比
wins = traded[traded['entry_success'] == 1]['optimal_entry_return'].mean()
losses = traded[traded['entry_success'] == 0]['optimal_entry_return'].mean()
profit_factor = abs(wins / losses) if losses != 0 else float('inf')

print(f"Win Rate: {win_rate:.2f}%")
print(f"Avg Return: {avg_return:.2f}%")
print(f"Profit Factor: {profit_factor:.2f}")
```

### 4. 訓練進場驗證分類器

```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# 準備訓練數據
candidates = df[df['is_entry_candidate'] == True]

# 特徵選擇 (技術指標)
feature_cols = ['atr', 'bb_width', 'bb_position'] + [col for col in df.columns if col.startswith('fib_')]

X = candidates[feature_cols]
y = candidates['entry_success']

# 訓練模型
clf = RandomForestClassifier(n_estimators=100, max_depth=10)
clf.clf.fit(X, y)

# 評估
score = clf.score(X, y)
print(f"Classifier accuracy: {score:.2f}")
```

## 調整配置

編輯 `config.yaml` 來改變標籤生成的行為:

```yaml
entry_validation:
  lookahead_bars: 20        # 進場後看20根K棒 (改成30看更遠)
  profit_threshold: 1.5     # 獲利需要勝風險1.5倍 (改成2.0更嚴格)
  fib_proximity: 0.01       # 接近Fibonacci的距離容限 (改成0.005更嚴格)
  bb_proximity: 0.01        # 接近BB的距離容限

indicators:
  bollinger_period: 20      # SMA週期
  bollinger_std: 2          # 標準差倍數 (改成1.5更緊)
  fibonacci_levels: [0.236, 0.382, 0.5, 0.618, 0.705, 0.786]  # Fib級別
```

## 故障排除

### 下載速度慢

```python
import os
# 使用中國鏡像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

### 內存不足

分批處理,逐個symbol生成:

```python
for symbol in ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']:
    df = generator.generate_labels(symbol, '15m', f'./output/{symbol}_15m.parquet')
    del df  # 釋放內存
```

### 無進場候選點

調整接近度參數:

```yaml
entry_validation:
  fib_proximity: 0.05      # 改成5%更寬鬆
  bb_proximity: 0.05
```

## 下一步

1. **生成多個symbol** - 修改test腳本,為38個symbol生成標籤
2. **訓練分類器** - 用entry_success訓練"進場判定"模型
3. **訓練迴歸模型** - 用optimal_entry_price/return訓練"最優進場"模型
4. **回測策略** - 在測試集上評估模型性能
5. **實盤應用** - 集成到交易系統

## 下載我的v1.2資料夾

```bash
git clone -b v1.2-label-system https://github.com/caizongxun/zong_zigzag.git
cd zong_zigzag/v1.2_label_system
python test_btc_15m.py
```
