# v1.2 標籤系統快速開始指南

## 安裝依賴

```bash
pip install pandas numpy pyyaml huggingface-hub
```

## 基本使用

### 1. 生成單個symbol的標籤

```python
from v1.2_label_system import LabelGenerator

config_path = "v1.2_label_system/config.yaml"
generator = LabelGenerator(config_path)

# 生成標籤
df_labeled = generator.generate_labels(
    symbol="BTCUSDT",
    timeframe="15m",
    save_path="./output/BTCUSDT_15m_labeled.parquet"
)

print(df_labeled.head())
```

### 2. 批量生成多個symbol的標籤

```python
from v1.2_label_system import LabelGenerator

config_path = "v1.2_label_system/config.yaml"
generator = LabelGenerator(config_path)

# 為3個symbol和2個timeframe生成標籤
results = generator.generate_batch(
    symbols=["BTCUSDT", "ETHUSDT", "ADAUSDT"],
    timeframes=["15m", "1h"],
    output_dir="./output/labeled_data"
)

print(f"Generated {len(results)} datasets")
for key in results.keys():
    print(f"  {key}: {len(results[key])} candles")
```

### 3. 分析標籤統計

```python
from v1.2_label_system import LabelGenerator, LabelStatistics

config_path = "v1.2_label_system/config.yaml"
generator = LabelGenerator(config_path)

df = generator.generate_labels("ETHUSDT", "1h")

# 生成報告
report = LabelStatistics.generate_full_report(df, "ETHUSDT", "1h")

# 打印報告
LabelStatistics.print_report(report)

# 保存報告
LabelStatistics.save_report(report, "./output/report.json")
```

## 標籤說明

生成的數據包含以下標籤列:

### 進場候選標籤
- `is_entry_candidate`: 布林值,表示該K棒是否為進場候選
- `entry_reason`: 進場原因,值為 "fib" (斐波那契) 或 "bb" (布林通道)

### 進場成功性
- `entry_success`: 二分類(0/1),表示進場後N根K棒內是否獲利
  - 1 = 成功 (獲利 > 風險 * 1.5倍)
  - 0 = 失敗

### 進場品質分數
- `entry_quality_score`: 連續值(0-100),衡量進場點的品質
  - 40分: 最大獲利幅度
  - 30分: 最小回撤控制
  - 20分: 風險報酬比
  - 10分: 勝率

### 最優進場點
- `optimal_entry_price`: 在候選進場區域內,歷史上最好的進場價格
- `optimal_entry_return`: 使用最優進場價的回報率(%)

## 配置參數

在 `config.yaml` 中調整:

```yaml
entry_validation:
  lookahead_bars: 20          # 進場後看多少根K棒
  profit_threshold: 1.5       # 利潤需要勝風險多少倍
  fib_proximity: 0.01         # 斐波那契接近度 (1%)
  bb_proximity: 0.01          # 布林通道接近度 (1%)

indicators:
  fibonacci_levels: [0.236, 0.382, 0.5, 0.618, 0.705, 0.786]
  bollinger_period: 20
  bollinger_std: 2
  atr_period: 14
  zigzag_threshold: 1.0
```

## 輸出目錄結構

```
./output/
  BTCUSDT_15m_labeled.parquet      # 單個symbol的標籤數據
  labeled_data/
    BTCUSDT_15m_labeled.parquet
    BTCUSDT_1h_labeled.parquet
    ETHUSDT_15m_labeled.parquet
    ...
  report.json                       # 統計報告
```

## 常見用途

### 獲取高品質的進場點

```python
candidates = df[df["is_entry_candidate"] == True]
high_quality = candidates[candidates["entry_quality_score"] > 70]

print(f"Found {len(high_quality)} high quality entry points")
print(high_quality[["open_time", "close", "entry_quality_score", "optimal_entry_price"]])
```

### 分析進場原因分佈

```python
candidates = df[df["is_entry_candidate"] == True]
print(candidates["entry_reason"].value_counts())
print(candidates["entry_reason"].value_counts(normalize=True) * 100)
```

### 計算成功率

```python
candidates = df[df["is_entry_candidate"] == True]
success_rate = (candidates["entry_success"] == 1).sum() / len(candidates)
print(f"Entry success rate: {success_rate*100:.2f}%")
```

## 下一步

生成的標籤數據可用於:

1. 訓練Entry Validity分類器 (XGBoost/RF)
2. 訓練Optimal Entry Regressor (預測最優進場價)
3. 訓練Stop Loss和Take Profit預測模型
4. 進行策略回測和優化

## 問題排除

### HuggingFace下載太慢

設置緩存目錄和使用本地鏡像:

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 中國鏡像

generator = LabelGenerator(config_path)
```

### 內存不足

使用批量處理,逐個symbol生成:

```python
for symbol in ["BTCUSDT", "ETHUSDT", "ADAUSDT"]:
    for tf in ["15m", "1h"]:
        df = generator.generate_labels(symbol, tf, f"./output/{symbol}_{tf}.parquet")
        del df  # 及時釋放內存
```

## 性能優化

- 使用多進程批量處理: `LabelGenerator.generate_batch()` 已內置並行支持
- 調整config.yaml中的 `batch_size` 和 `n_workers` 參數
- 使用parquet格式存儲以加快I/O

## 更多示例

詳見 `examples/example_label_generation.py`
