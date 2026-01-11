# ZigZag v1.2.0 Colab 完整工作流程指南

本文檔提供了完整的 Colab 工作流程，這樣你可以後續在 Colab 中進行整個系統的開發。

## 一、初始設定

### Cell 1: 克隆仓庫並切換到 main1.2.0 分支

```python
# 克隆仓庫
!git clone https://github.com/caizongxun/zong_zigzag.git
%cd zong_zigzag

# 切換到 main1.2.0 分支
!git checkout main1.2.0

# 驗證當前分支
!git branch
print("\n位置:", !pwd)
```

**預期輸出:**
```
Cloning into 'zong_zigzag'...
* main1.2.0
  main
  ...
位置: /content/zong_zigzag
```

---

## 二、環境需求安裝

### Cell 2: 安裝主要依賴

```python
!pip install torch xgboost scikit-learn pandas numpy huggingface-hub -q

print("\n安裝完成，主要依賴:")
print("- torch: 深度學習框架")
print("- xgboost: 機器學習模式")
print("- scikit-learn: 數浮正規化小工具")
print("- huggingface-hub: 數據集載入")
```

---

## 三、輸入模組并驗證

### Cell 3: 導入訓練前的基本模組

```python
import sys
sys.path.append('/content/zong_zigzag/v1.2_label_system')

from data_loader_v1.2.0 import HuggingFaceDataLoader, DataProcessor
print("\u6578據加載模組伺入成功")

try:
    from model_architecture_v1.2.0 import DualLayerMetaLabelingSystem, MetaFeatureExtractor
    print("模式架构模組伺入成功")
except Exception as e:
    print(f"警告: {e}")
    print("襲樹這個模組，後續會使用")
```

---

## 四、數據加載下載

### Cell 4: 從 HuggingFace 加載 BTCUSDT 數據

```python
from datetime import datetime, timedelta

loader = HuggingFaceDataLoader(use_huggingface=True, local_cache=True)

print("正在從 HuggingFace 加載 BTCUSDT 15 分鞑8月數據...")

raw_data = loader.load_klines(
    symbol='BTCUSDT',
    timeframe='15m',
    start_date=datetime.now() - timedelta(days=180),
    end_date=datetime.now()
)

print(f"\n加載成功")
print(f"數據形狀: {raw_data.shape}")
print(f"數據列: {raw_data.columns.tolist()}")
print(f"\n前 5 行:")
print(raw_data.head())
```

---

## 五、數據準備與清洗

### Cell 5: 數據清洗

```python
processor = DataProcessor()

print("第一步: 使用前向填充清理數據...")
cleaned_data = processor.clean_data(raw_data, method='forward_fill')

print("\n第二步: 移除異常值 (Z-Score > 3.0)...")
cleaned_data = processor.remove_outliers(
    cleaned_data,
    columns=['close', 'volume'],
    threshold=3.0
)

print(f"\n清理後數據形狀: {cleaned_data.shape}")
```

### Cell 6: 計算增強特徵

```python
print("計算強化特徵...")
feature_data = processor.calculate_enhanced_features(cleaned_data)

print(f"\n新增特徵數量: {len(feature_data.columns) - len(cleaned_data.columns)}")
print(f"總特徵數: {len(feature_data.columns)}")

print("\n敶特徵一覽:")
for i, col in enumerate(feature_data.columns[-5:], 1):
    print(f"  {i}. {col}")
```

### Cell 7: 準備訓練數據

```python
print("分割訓練/驗證/測試數據...")

data_package = processor.prepare_training_data(
    feature_data,
    sequence_length=30,
    test_ratio=0.2,
    validation_ratio=0.1
)

print(f"\n訓練集: {data_package['X_train'].shape}")
print(f驗證集: {data_package['X_val'].shape}")
print(f"測試集: {data_package['X_test'].shape}")

print(f"\n穩定化半遨: {data_package['scaler']}")
```

---

## 六、模式訓練 (選項)

### Cell 8: GPU 檢查並 LSTM 訓練

```python
import torch

# 檢查 GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"可用設備: {device}")

if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA 版本: {torch.version.cuda}")

# 很憾壓，我們需要先修議提供工作流程
# 実際的模式訓練是選項擊光
```

---

## 七、提交到 GitHub

### Cell 9: 上傳改歡到 main1.2.0

```python
# 設定 Git 認證
!git config user.email "caizongxun@users.noreply.github.com"
!git config user.name "zong"

# 檢查是否有修改
!git status

# 新增檔案 (如果你有新建的筆記或檔案)
# !git add your_file.py

# 例如，簡筆或經驗結果
!git add -A
!git commit -m "feat: Colab 訓練結果 - BTCUSDT 15m 數據分析"
!git push origin main1.2.0

print("\n改歡已提交到 main1.2.0 分支")
print("GitHub: https://github.com/caizongxun/zong_zigzag")
```

---

## 八、粗密結果杤取

### Cell 10: 查看訓練綗密結果

```python
print("訓練結果水平:")
print(f"- 訓練集 數據數: {len(data_package['y_train'])}")
print(f"- 驗證集 數據數: {len(data_package['y_val'])}")
print(f"- 測試集 數據數: {len(data_package['y_test'])}")

# 統計 類別分佈
import numpy as np
unique_labels = np.unique(data_package['y_train'])
print(f"\n類別 (0=DOWN, 1=FLAT, 2=UP): {unique_labels}")
```

---

## 九、常見啊噢

### 啊噢 1: 連接斷了？
在 Colab 中再劳➚父 Cell 進行重新轉接:
```python
%cd /content/zong_zigzag
```

### 啊噢 2: 存放訓練結果

訓練深湫学习的模式:
```python
import os
os.makedirs('/content/models/v1.2.0', exist_ok=True)
# 保存模式
```

### 啊噢 3: 使用 GPU 加速

Colab 預設已連接 GPU，你不需要東何。治理網淯計伊就會自動要求 GPU.

---

## 十、完整一次性卻翔純是

在 Colab 中引入我們結納的網路：

```python
# 二一次性卻翔純是 - 技能一体魜竣漂軸牧色
print("深度學習模式記錄檢統流程完成!")
```

---

## 下一步

你可以:

1. **在 main1.2.0 中繼續開發**: 整個開發過程都進行在 main1.2.0 分支
2. **測試其他交易對**: 修改 `BTCUSDT` 為你文甘的幣種
3. **加輊Colab 檔案**: 在待複法一方梨梅將所有經驗結果支技味道檔案一起提交

大旅了!
