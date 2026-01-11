# 本地運行 BTC 15m 標籤生成

## 步驟1: 安裝依賴

```bash
pip install pandas numpy pyyaml huggingface-hub scikit-learn
```

## 步驟2: 下載代碼

```bash
git clone https://github.com/caizongxun/zong_zigzag.git
cd zong_zigzag
git checkout v1.2-label-system
cd v1.2_label_system
```

## 步驟3: 運行測試

### 方法 A: 直接運行腳本 (推薦)

```bash
python test_btc_15m.py
```

這會:
1. 從HuggingFace下載BTCUSDT 15m數據 (約 50-100MB)
2. 計算所有技術指標 (ATR, Bollinger Bands, Fibonacci, ZigZag)
3. 生成三層標籤
4. 打印詳細統計報告
5. 保存報告到 `./output/BTCUSDT_15m_report.json`

預計耗時: 5-15分鐘 (取決於網速)

### 方法 B: 在Jupyter/IPython中運行

```python
import sys
sys.path.insert(0, '.')

from label_generator import LabelGenerator
from label_statistics import LabelStatistics

# 初始化
generator = LabelGenerator('config.yaml')

# 生成標籤 (不保存文件)
df = generator.generate_labels(
    symbol='BTCUSDT',
    timeframe='15m',
    save_path=None
)

# 查看數據
print(f"Total candles: {len(df)}")
print(df[['open_time', 'close', 'is_entry_candidate', 'entry_success', 'entry_quality_score']].head(20))

# 生成報告
report = LabelStatistics.generate_full_report(df, 'BTCUSDT', '15m')
LabelStatistics.print_report(report)
```

### 方法 C: 分步執行 (如果遇到問題)

```python
import sys
sys.path.insert(0, '.')

from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from entry_validator import EntryValidator
import yaml

# 1. 加載配置
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# 2. 加載數據
loader = DataLoader(
    repo_id=config['huggingface']['repo_id'],
    hf_root=config['huggingface']['hf_root']
)
df = loader.load_klines('BTCUSDT', '15m')
print(f"Loaded {len(df)} candles")

# 3. 計算特徵
fe = FeatureEngineer(config)
df = fe.extract_all_features(df)
print(f"Extracted features: {len(df.columns)} columns")

# 4. 生成標籤
validator = EntryValidator(config)
df = validator.generate_all_labels(df)
print(f"Generated labels")

# 5. 分析結果
candidates = df[df['is_entry_candidate'] == True]
print(f"Entry candidates: {len(candidates)} ({len(candidates)/len(df)*100:.2f}%)")
print(f"Success rate: {(candidates['entry_success']==1).sum() / len(candidates) * 100:.2f}%")
```

## 預期輸出

```
======================================================================
v1.2 Label System Test - BTC 15m
======================================================================

Data Overview
----------------------------------------------------------------------
Total candles: 3000
Date range: 2024-01-01 12:00:00 to 2024-12-31 23:45:00
Price range: 42130.50 - 98700.25

Entry Candidate Statistics
----------------------------------------------------------------------
Entry candidates: 156 (5.20%)

Entry Success Breakdown:
  Successful: 98 (62.82%)
  Failed: 58 (37.18%)

Entry Quality Score:
  Mean: 45.32
  Median: 42.15
  Min: 10.20
  Max: 98.75
  Std Dev: 18.50

Entry Reason Distribution:
  fib: 95 (60.90%)
  bb: 61 (39.10%)

Optimal Entry Return:
  Mean: 2.45%
  Median: 1.82%
  Min: -5.30%
  Max: 12.80%
  Profitable: 142 (91.03%)

======================================================================
Test completed successfully!
======================================================================
```

## 文件輸出位置

```
v1.2_label_system/
├── output/
│   └── BTCUSDT_15m_report.json     <- 統計報告
```

## 查看結果

### 1. 打印報告 (已在test腳本中自動執行)

```bash
cat output/BTCUSDT_15m_report.json
```

### 2. 在Python中加載結果

```python
import json

with open('output/BTCUSDT_15m_report.json') as f:
    report = json.load(f)

print("Entry Candidates:", report['entry_candidates'])
print("Quality Scores:", report['quality_scores'])
print("Entry Reasons:", report['entry_reasons'])
print("Optimal Returns:", report['optimal_returns'])
```

### 3. 保存標籤數據為Parquet

```python
from label_generator import LabelGenerator

generator = LabelGenerator('config.yaml')
df = generator.generate_labels(
    symbol='BTCUSDT',
    timeframe='15m',
    save_path='./output/BTCUSDT_15m_labeled.parquet'  # 加上這行
)
```

## 常見問題

### Q: 下載太慢
A: 使用中國鏡像
```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

### Q: 內存不足
A: 減少進程數或逐個symbol處理:
```python
for symbol in ['BTCUSDT', 'ETHUSDT']:
    df = generator.generate_labels(symbol, '15m', f'./output/{symbol}_15m.parquet')
    del df  # 立即釋放內存
```

### Q: 沒有進場候選點
A: 調整接近度參數在config.yaml:
```yaml
entry_validation:
  fib_proximity: 0.05      # 改成5%
  bb_proximity: 0.05
```

### Q: 進場成功率太低
A: 調整lookahead_bars或profit_threshold:
```yaml
entry_validation:
  lookahead_bars: 30       # 看更多K棒
  profit_threshold: 1.0    # 降低利潤門檻
```

## 下一步

成功運行BTC 15m後:

1. **批量生成其他symbol** - 修改test腳本或使用generate_batch()
2. **分析指標分布** - 檢查entry_quality_score的中位數
3. **驗證标签有效性** - 用entry_success訓練分類器
4. **優化參數** - 根據統計結果調整config.yaml
5. **訓練ML模型** - 為下一階段做準備

有任何問題歡迎提問!
