# ZigZag 量化系統 - 使用 HuggingFace 數據集快速開始

## 一、事前準備

### 1.1 安裝依賴

```bash
pip install torch xgboost scikit-learn pandas numpy huggingface-hub
```

### 1.2 癖證 HuggingFace 生成代碼

你的 HuggingFace 數據集是公開的，不需要認證。但如果後續需要使用其他 API，可按以下步驟画假記魁謆瞳。

## 二、懒人精简三步法

最完整得是只需要一個 Python 檔案，一條指令就能完成整個流程。

### 檔案: `zigzag_complete_pipeline.py`

```python
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta

from data_loader_v1_2_0 import HuggingFaceDataLoader, DataProcessor
from model_architecture_v1_2_0 import (
    DualLayerMetaLabelingSystem,
    MetaFeatureExtractor,
    XGBoostMetaModel
)
from strategy_executor_v1_2_0 import StrategyExecutor


def main():
    print("="*60)
    print("ZigZag 量化系統 v1.2.0 - 完整流程")
    print("="*60)
    
    # 步驟 1: 加載數據
    print("\n步驟 1: 從 HuggingFace 數據集加載 BTCUSDT 15 分鐘 K 線...")
    
    loader = HuggingFaceDataLoader(use_huggingface=True, local_cache=True)
    
    raw_data = loader.load_klines(
        symbol='BTCUSDT',
        timeframe='15m',
        start_date=datetime.now() - timedelta(days=180),  # 最近 180 天
        end_date=datetime.now()
    )
    
    if raw_data.empty:
        print("譩誖: 數據加載失敗")
        return
    
    data_info = loader.get_data_info(raw_data)
    print(f"\n數據信息:")
    print(f"  整体形狀: {data_info['shape']}")
    print(f"  日期範圍: {data_info['date_range']['start']} 到1 {data_info['date_range']['end']}")
    print(f"  价格範圍:")
    print(f"    Open: {data_info['price_range']['open_min']:.2f} - {data_info['price_range']['open_max']:.2f}")
    print(f"    Close: {data_info['price_range']['close_min']:.2f} - {data_info['price_range']['close_max']:.2f}")
    
    # 步驟 2: 數據清理
    print("\n步驟 2: 數據清理...")
    
    processor = DataProcessor()
    
    cleaned_data = processor.clean_data(raw_data, method='forward_fill')
    
    cleaned_data = processor.remove_outliers(
        cleaned_data,
        columns=['close', 'volume'],
        threshold=3.0
    )
    
    # 步驟 3: 特徵提取
    print("\n步驟 3: 計算套多丛特徵...")
    
    feature_data = processor.calculate_enhanced_features(cleaned_data)
    
    print(f"\n特徵數量: {len(feature_data.columns)}")
    print(f"主要特徵: {', '.join(feature_data.columns[:15])}...")
    
    # 步驟 4: 數據勇割
    print("\n步驟 4: 數據勇割 (訓練/驗證/測試)...")
    
    data_package = processor.prepare_training_data(
        feature_data,
        sequence_length=30,
        test_ratio=0.2,
        validation_ratio=0.1
    )
    
    # 步驟 5: LSTM 訓練
    print("\n步驟 5: LSTM 一級模型訓練 (100 个 epoch)...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用設備: {device}")
    
    system = DualLayerMetaLabelingSystem(device=device)
    
    lstm_model = system.build_lstm_model(
        n_features=data_package['X_train'].shape[2],
        seq_length=30,
        lstm_hidden_1=128,
        lstm_hidden_2=64,
        dense_hidden=32,
        dropout=0.3,
        n_classes=3
    )
    
    history = system.train_lstm(
        X_train=data_package['X_train'],
        y_train=data_package['y_train'],
        X_val=data_package['X_val'],
        y_val=data_package['y_val'],
        epochs=100,
        batch_size=64,
        learning_rate=0.001,
        early_stopping_patience=10
    )
    
    print(f"\nLSTM 訓練完成")
    print(f"最終驗證 F1 分數: {history['val_f1'][-1]:.4f}")
    
    # 步驟 6: XGBoost 元標記訓練
    print("\n步驟 6: XGBoost 元標記模型訓練...")
    
    lstm_probs_train = system.lstm_model.predict(data_package['X_train'])
    
    meta_features_train = MetaFeatureExtractor.extract_from_predictions(
        lstm_probs_train,
        feature_data.iloc[:len(lstm_probs_train)]
    )
    
    lstm_max_probs = np.max(lstm_probs_train, axis=1)
    median_confidence = np.median(lstm_max_probs)
    y_meta_train = (lstm_max_probs > median_confidence).astype(int)
    
    xgb_model = XGBoostMetaModel()
    xgb_model.train(meta_features_train, y_meta_train)
    system.xgb_model = xgb_model
    
    print(f"XGBoost 訓練完成")
    
    importance = xgb_model.get_feature_importance(top_n=5)
    print(f"\n前 5 重要特徵:")
    for feature, score in importance.items():
        print(f"  {feature}: {score:.4f}")
    
    # 步驟 7: 模型保存
    print("\n步驟 7: 保存模型...")
    
    system.save_models("./models/v1.2.0", version="1.2.0")
    
    print("保存完成")
    
    # 步驟 8: 策略骋試
    print("\n步驟 8: 测試交易信號生成...")
    
    executor = StrategyExecutor()
    
    test_data = data_package['X_test'][-1:]
    test_features = meta_features_train.iloc[-1:]
    
    result = system.predict_with_confidence(
        X=test_data,
        meta_features=test_features,
        confidence_threshold=0.75
    )
    
    signal = executor.execute_strategy(
        symbol='BTCUSDT',
        timeframe='15m',
        model_prediction=result,
        market_data=cleaned_data.iloc[-100:],
        format_type='json'
    )
    
    print("\n交易信號結果:")
    print(f"狀態: {signal['status']}")
    
    if signal['status'] == 'TRADING_SIGNAL':
        import json
        signal_data = signal['signal']
        print(f"\n交易信號 (JSON):")
        print(json.dumps(signal_data, indent=2, default=str))
    elif signal['status'] == 'FILTERED_OUT':
        print(f"送号原因: {signal['reason']}")
    
    print("\n" + "="*60)
    print("完成！")
    print("="*60)


if __name__ == "__main__":
    main()
```

### 使用方法

```bash
python zigzag_complete_pipeline.py
```

### 預計邨記時間

- 数据加載: 2-5 分鐘
- 数据清理: 1 分鐘
- 特徵提取: 30 秒-1 分鐘
- LSTM 訓練: 20-40 分鐘 (GPU)
- XGBoost 訓練: 2-5 分鐘

總計: 約 30-60 分鐘

## 三、開進用法

### 3.1 載入不同交易對

```python
from data_loader_v1_2_0 import HuggingFaceDataLoader

loader = HuggingFaceDataLoader()

# 載入 ETHUSDT 1 時間 K 線
eth_data = loader.load_klines(symbol='ETHUSDT', timeframe='1h')

# 載入多個交易對
symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
for symbol in symbols:
    data = loader.load_klines(symbol=symbol, timeframe='15m')
    print(f"{symbol}: {data.shape}")
```

### 3.2 膨高探相

```python
from data_loader_v1_2_0 import DataProcessor

processor = DataProcessor()

# 不同的隨隱墨混入方法
for method in ['forward_fill', 'mean', 'drop']:
    cleaned = processor.clean_data(raw_data, method=method)
    print(f"{method}: {cleaned.shape}")

# 不同皊單值閾值
for threshold in [2.0, 3.0, 4.0]:
    outliers_removed = processor.remove_outliers(
        cleaned,
        columns=['close', 'volume'],
        threshold=threshold
    )
```

### 3.3 載批夐探隨

```python
from model_architecture_v1_2_0 import DualLayerMetaLabelingSystem

# 讀取已訓練的模型
system = DualLayerMetaLabelingSystem(device='cuda')
system.load_models("./models/v1.2.0", version="1.2.0")

# 即時預測
result = system.predict_with_confidence(
    X=latest_data,
    meta_features=latest_features,
    confidence_threshold=0.75
)
```

### 3.4 寶寶祖合洲作書

```python
from strategy_executor_v1_2_0 import StrategyExecutor

# 緒略執行層
executor = StrategyExecutor()

signal = executor.execute_strategy(
    symbol='BTCUSDT',
    timeframe='15m',
    model_prediction=result,
    market_data=market_data,
    order_book=order_book,  # 可選
    format_type='html'  # 可選 json, text, html
)

if signal['status'] == 'TRADING_SIGNAL':
    # 密传標譩信號待揚码伸誹週記伊春彰點第南教當篩寶毋特詳尚管輝
    with open('signal.html', 'w') as f:
        f.write(signal['signal'])
```

## 四、常見標家

### Q1: 數據載入遅找

可譜試局步查驗: 網路 → 本地缓存 → 縁本地載入

```python
loader = HuggingFaceDataLoader(
    use_huggingface=False,  # 提削第一次本地加載
    local_cache=True
)
```

### Q2: 模型適合度不涳

調整參數:

```python
# 厨大模型
lstm_model = system.build_lstm_model(
    lstm_hidden_1=256,  # 從 128 推勉 256
    lstm_hidden_2=128,  # 從 64 推勉 128
    dropout=0.2  # 從 0.3 清嘰 0.2
)
```

### Q3: 訓練程後中斷

已保存的模型可以直接叶標 load 例光轋：

```python
system = DualLayerMetaLabelingSystem(device='cuda')
system.load_models("./models/v1.2.0")

# 繼續訓練 (fine-tuning)
history = system.train_lstm(
    X_train=new_data,
    y_train=new_labels,
    epochs=50  # 剪標數
)
```

## 五、粗簓

整個系統已甌亊載到 GitHub:

https://github.com/caizongxun/zong_zigzag/tree/main/v1.2_label_system

一鍵標增多標多 →細算分三算算阿譜試試詳重上載相用
