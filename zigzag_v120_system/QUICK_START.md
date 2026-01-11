# ZigZag v1.2.0 - 快速開始指南

## 简易設置

### Google Colab 第一次使用

```python
import sys
sys.path.append('/content/zong_zigzag')

# 克隆儲存库
!git clone https://github.com/caizongxun/zong_zigzag.git -b refactor/rename-package
%cd zong_zigzag

# 安裝依賴
!pip install -q -r zigzag_v120_system/requirements.txt
```

### 正確的導入方式

```python
# 正確 - 使用新的套件名稱
from zigzag_v120_system.data_loader import HuggingFaceDataLoader, DataProcessor
from zigzag_v120_system.feature_engineering import FeatureBuilder
from zigzag_v120_system.model_architecture import LSTMPredictor, DualLayerDefenseSystem
from zigzag_v120_system.strategy_executor import StrategyExecutor

print("OK - 模組已成功導入")
```

## 基本使用範例

### 1. 加載數據

```python
loader = HuggingFaceDataLoader()
df = loader.load_klines('BTCUSDT', timeframe='15m')
print(f"數據: {df.shape[0]} 敷")
```

### 2. 數據準備

```python
processor = DataProcessor()
df_clean = processor.clean_data(df)
print(f"清理後: {df_clean.shape[0]} 敷")
```

### 3. 特徵構建

```python
feature_builder = FeatureBuilder()
features = feature_builder.build_feature_matrix(df_clean)
print(f"特徵: {features.shape[1]} 的")
```

## 常見問題

**Q: SyntaxError: invalid decimal literal**

**A:** 這是旧的套件名稱問題。請確保幾使用：
- 新正確: `zigzag_v120_system` 
- 老錯誤: `v1.2_label_system` (按照後迷)

## 版本

- v1.2.0
- 日期: 2026-01-11
