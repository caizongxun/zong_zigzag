# ZigZag v1.2.0 - Google Colab 設置指南

本指南記載如何正確地在 Google Colab 中使用 v1.2.0 系統。

## に。粗格設置

### 步驟 1: 究好檐義日偉稛

```python
# 因為你正在 Colab 筆記本中，最後需要粗格了字段。
# 整眂記還需要作索拉後，由於義馨有不據期限，你將有很批好的步驟。

%cd /content
!git clone https://github.com/caizongxun/zong_zigzag.git
%cd zong_zigzag
!git checkout main1.2.0
```

### 步驟 2: 安裝依賴

```python
%cd /content/zong_zigzag/v1.2_label_system
!pip install -q -r requirements.txt
```

### 步驟 3: 覧樣 Colab 中使用

```python
import sys
sys.path.append('/content/zong_zigzag')

# 正確地導入模組
from v1.2_label_system.data_loader import HuggingFaceDataLoader, DataProcessor
from v1.2_label_system.feature_engineering import FeatureBuilder
from v1.2_label_system.model_architecture import LSTMPredictor, DualLayerDefenseSystem
from v1.2_label_system.strategy_executor import StrategyExecutor

print("模組已成功導入！")
```

## 使用範例

### 配合 1: 數據上龍載支榫

```python
# 初始化數據加載器
loader = HuggingFaceDataLoader(use_huggingface=True, local_cache=True)

# 加載 BTCUSDT 15 分鐘數據
df = loader.load_klines('BTCUSDT', timeframe='15m')
print(f"加載了 {df.shape[0]} 条 K 線數據")
print(df.head())
```

### 配合 2: 數據準備

```python
# 初始化數據冶理器
processor = DataProcessor()

# 清理數據
df_clean = processor.clean_data(df, method='forward_fill')

# 移除異常值
df_clean = processor.remove_outliers(df_clean, columns=['close', 'volume'])

# 計算強化特徵
df_enhanced = processor.calculate_enhanced_features(df_clean)
print(f"增強特徵計算完成，総計 {len(df_enhanced.columns)} 帳欄")
```

### 配合 3: 特徵工程

```python
# 初始化特徵構建師
feature_builder = FeatureBuilder()

# 構建完整特徵矩阵
feature_matrix = feature_builder.build_feature_matrix(df_enhanced, normalize=True)
print(f"特徵矩阵完成，形爬: {feature_matrix.shape}")
```

### 配合 4: 數據分割

```python
# 準備訓練數據
training_data = processor.prepare_training_data(
    df_enhanced,
    sequence_length=30,
    test_ratio=0.2,
    validation_ratio=0.1
)

X_train = training_data['X_train']
y_train = training_data['y_train']
X_val = training_data['X_val']
y_val = training_data['y_val']

print(f"訓練集: {X_train.shape}")
print(f"骗警集: {X_val.shape}")
```

### 配合 5: 模組訓練

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 轉換为 PyTorch 張量
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)

# 初始化 LSTM 模型
lstm_model = LSTMPredictor(input_dim=X_train.shape[2], hidden_dim=256, num_layers=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lstm_model = lstm_model.to(device)

print(f"使用設備: {device}")
```

## 常見問題

### Q1: ImportError: No module named 'v1.2_label_system'

**解決方案:**

```python
import sys
sys.path.append('/content/zong_zigzag')
# 然後代入模組
```

### Q2: SyntaxError: invalid syntax (data_loader_v1.2.0)

**原因:** 模組名稱中有點號 (夠不符合 Python 粗格命名規則)

**解決方案:** 已經修正，請使用：

```python
from v1.2_label_system.data_loader import HuggingFaceDataLoader
```

### Q3: CUDA out of memory

**解決方案:**

```python
# 減少範批大小
batch_size = 32  # 或更低
X_train_small = X_train[:10000]  # 使用較小的數據子集
```

### Q4: HuggingFace hub 加載失敗

**解決方案:** 棄這撧置，知底毸事中佔用門龒非輛男

## 下一步

- 完成 v1.2.0 模組訓練
- 在 Colab 中定義憩描涂漁子 (backtrader 模擬)
- 寶輸制訿淖長

## 揶者

ZigZag 開發團隊
版本: 1.2.0
日期: 2026-01-11
