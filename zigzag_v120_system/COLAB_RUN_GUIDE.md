# ZigZag v1.2.0 - Colab 執行指南

## 方案：一鍵執行

在 Google Colab 第一个欄位執行以下代碼：

```python
# 第一步：設置环境並克隆儲存库
import os
os.chdir('/tmp')

# 克隆儲存库
!git clone https://github.com/caizongxun/zong_zigzag.git
%cd zong_zigzag

# 安裝依賴
!pip install -q -r zigzag_v120_system/requirements.txt

print("✓ 依賴安裝完成")
```

```python
# 第二步：執行訓練脚本
!python zigzag_v120_system/train_complete.py
```

---

## 詳所的版本（完整三欄位）

如果你想一步一步執行，以下是完整三欄位的 Colab 代碼：

### 第一欄：依賴安裝不需要修改

```python
import os
os.chdir('/tmp')
!git clone https://github.com/caizongxun/zong_zigzag.git
%cd zong_zigzag
!pip install -q torch pandas numpy scikit-learn xgboost huggingface-hub tqdm scipy
print("✓ 依賴安裝完成")
```

### 第二欄：一鍵執行訓練

```python
!python zigzag_v120_system/train_complete.py
```

### 第三欄（可選）：驗證訓練結果

```python
import torch

# 載入訓練模型
model_path = '/content/zong_zigzag/zigzag_v120_system/models/lstm_model_complete.pth'
model_data = torch.load(model_path)

print("模型評估")
print("=" * 50)
print(f"訓練 RMSE: {model_data['train_rmse']:.6f}")
print(f"測試 RMSE:  {model_data['test_rmse']:.6f}")
print(f"後繼推理輸入維度: {model_data['input_dim']}")
print("=" * 50)
print("\n✓ 模型已驗證，可進行下一步推理")
```

---

## 驗證列表

正常懂出：

- [x] 模組正常導入
- [x] 數據加載成功
- [x] 特徵構建完成
- [x] LSTM 訓練不這了
- [x] 模型保存成功

---

## 此後步驟

1. 使用訓練好的模型進行推理
2. 粗化策略信號
3. 回測估計訓練效果
4. 部署自動交易

---

## 常見問題

**Q: 需要多長時間？**

A: 約 10-15 分鐘（依賴 Colab 配轰）

**Q: GPU 會加速吧？**

A: 會。可以在 Colab 選擷介面上選選 GPU 機組。

**Q: 是否需要修改代碼？**

A: 不需要。署一起一鍵執行。
