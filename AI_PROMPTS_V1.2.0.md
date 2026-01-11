# ZigZag 量化系統 v1.2.0 - 三階段 AI 開發提示詞

這份文檔包含完整的專業提示詞，用於與 AI 助手（ChatGPT、Claude、GitHub Copilot）協作開發 ZigZag 反轉預測系統。

## 專案資訊

**GitHub 倉庫:** https://github.com/caizongxun/zong_zigzag

**分支:** `v1.2-label-system`

**當前版本:** v1.2.0

**核心目錄結構:**
```
v1.2_label_system/
├── config.yaml                 # 配置文件（已優化參數）
├── label_generator.py         # 標籤生成引擎
├── entry_validator.py         # 進場驗證邏輯
├── feature_engineering.py     # 特徵工程
├── label_statistics.py        # 統計分析
├── data_loader.py             # 數據加載
├── grid_search_params.py      # 網格搜索參數優化
├── fast_grid_search.py        # 快速適應性搜索
├── quick_test.py              # 快速驗證腳本
├── test_btc_15m.py            # BTC 15分鐘測試
└── examples/                  # 示例和教程
```

**當前推薦參數 (v1.2.0 最優組合):**
```yaml
entry_validation:
  fib_proximity: 0.002        # 斐波那契接近度 (精確度較高)
  bb_proximity: 0.002         # 布林帶接近度 (精確度較高)
  lookahead_bars: 20
  profit_threshold: 1.5

indicators:
  zigzag_threshold: 0.3       # ZigZag 波幅閾值 (高靈敏度)
  bollinger_period: 20
  bollinger_std: 2
  atr_period: 14
  fibonacci_levels: [0.236, 0.382, 0.5, 0.618, 0.705, 0.786]
```

**驗證指標目標:**
- 進場比例: 8-15%
- 成功率: 35-40%
- 平均回報: 0.5-1.0%
- 盈利比率: 80%+
- 品質分數: 45+

---

# 第一階段：數據預處理與特徵工程提示詞

## 適用場景
使用此提示詞時，要求 AI 生成或優化數據處理、特徵工程相關的 Python 代碼。適用於:
- 改進現有的 `feature_engineering.py` 模塊
- 開發新的特徵計算方法
- 優化數據加載和預處理流程

---

## 提示詞內容

### 背景信息

我正在開發一個加密貨幣反轉預測系統 (ZigZag v1.2.0)，目標是在 15 分鐘時間框架上預測 BTC、ETH 等品種的反轉點位。

專案倉庫: https://github.com/caizongxun/zong_zigzag (分支: v1.2-label-system)

當前使用的主要工具和庫:
- **數據源:** Binance OHLCV 數據 (通過 Hugging Face datasets 庫)
- **數據框架:** Pandas, NumPy
- **技術指標:** Fibonacci levels, Bollinger Bands, ZigZag (自定義實現)
- **驗證邏輯:** 三重障礙法 (Triple Barrier Method)
- **標籤系統:** 多層次標籤 (進場信號 + 品質評分 + 回報預測)

### 任務要求

請擔任資深量化數據工程師，根據以下需求編寫或優化 Python 代碼:

#### 需求 1: 分數階差分特徵 (Fractional Differentiation)

**背景說明:**
標準的一階差分 (Delta) 會導致大量信息丟失，但完全不做差分會導致數據非平穩 (non-stationary)。我們需要「分數階差分」來平衡這兩個問題。

**具體需求:**
1. 實現一個函數 `fractional_differentiation(series, d=0.4)` 其中:
   - `series`: Pandas Series (時間序列數據，如收盤價)
   - `d`: 差分階數 (通常 0 < d < 1，推薦 d=0.3 到 0.5)
   - 返回: 經過分數階差分後的平穩時間序列

2. 數學原理:
   - 使用 Yule-Walker 方法計算自相關 (ACF)
   - 或使用展開級數: $\Delta^d_t = \sum_{k=1}^{t} \binom{d}{k} (-1)^k X_{t-k}$
   - 其中 $\binom{d}{k} = \frac{d(d-1)...(d-k+1)}{k!}$ (廣義二項式係數)

3. 應驗證輸出數據是否為平穩 (Augmented Dickey-Fuller 測試，p-value < 0.05)

4. 代碼應包含:
   - 完整的 ADF 檢驗 (使用 `statsmodels.tsa.stattools.adfuller`)
   - 可視化: 繪製原始序列、一階差分、分數階差分的比較圖
   - 在 `feature_engineering.py` 中集成該函數

#### 需求 2: 衍生品特徵構建 (Derivatives Features)

**背景說明:**
我們的數據中包含資金費率 (Funding Rate) 和未平倉合約量 (Open Interest)，這些衍生品指標可以揭示市場情緒。

**具體需求:**
1. 實現以下特徵:

   a) **價格-OI 背離指標:**
   ```
   oi_divergence = (price_change % 5d) - (oi_change % 5d)
   ```
   - 當價格創新高但 OI 下降時，通常預示反轉
   - 計算 5 天的變化率差值

   b) **資金費率異常偵測:**
   ```
   funding_rate_anomaly = (current_funding_rate - MA(funding_rate, 30d)) / std(funding_rate, 30d)
   ```
   - 當異常值 > 2 時，表示極端情緒

   c) **OI 與價格的領先-滯後關係:**
   - 計算過去 7 天內 OI 變化與未來 1 天價格變化的相關性
   - 如果相關性 > 0.5，OI 可作為領先指標

2. 代碼應返回 DataFrame，包含:
   - 時間戳
   - `price`
   - `oi`
   - `funding_rate`
   - `oi_divergence`
   - `funding_rate_anomaly`
   - `oi_lead_correlation`

3. 處理缺失值和極端值的策略

#### 需求 3: 微結構特徵 (Market Microstructure Features)

**背景說明:**
訂單簿不平衡可以反映市場參與者的實時情緒。

**具體需求:**
1. 實現訂單簿不平衡指標:
   ```
   OBI = (BidQty - AskQty) / (BidQty + AskQty)
   ```
   - 範圍: [-1, 1]
   - 正值: 買盤優於賣盤
   - 負值: 賣盤優於買盤

2. 進階版本 (加權訂單簿不平衡):
   ```
   WOBI = (∑(BidPrice * BidQty) - ∑(AskPrice * AskQty)) / (∑(BidPrice * BidQty) + ∑(AskPrice * AskQty))
   ```

3. 滾動統計:
   - 5 分鐘內 OBI 的平均值
   - OBI 的標準差 (波動性)
   - OBI 的最大值/最小值

4. 假設我們可以從 Binance API 或本地數據中獲取訂單簿深度數據，代碼應支持:
   - 讀取 Level 2 訂單簿數據
   - 對齐時間戳
   - 計算每個時段的 OBI

#### 需求 4: 綜合特徵集成

請編寫一個主函數 `build_feature_matrix()`，該函數:
1. 輸入: OHLCV DataFrame (從 HuggingFace 加載的 Binance 數據)
2. 計算所有上述特徵
3. 輸出: 完整的特徵矩陣，包含:
   - 基礎 OHLCV 特徵
   - 分數階差分特徵
   - 衍生品特徵
   - 微結構特徵
   - 時間特徵 (小時、日期、星期幾)
4. 進行標準化 (Z-score normalization) 並保存到 Parquet 格式

#### 需求 5: 數據質量檢查

實現一個驗證函數 `validate_feature_quality()`:
1. 檢查缺失值比例 (應 < 2%)
2. 檢查異常值 (基於 IQR 方法)
3. 檢查特徵之間的相關性 (去除高度相關的特徵)
4. 生成特徵統計報告 (mean, std, min, max, skewness, kurtosis)

### 輸出要求

1. **完整的函數封裝:**
   - 每個函數都有詳細的 docstring
   - 包含類型提示 (Type Hints)
   - 異常處理和日誌記錄

2. **數學解釋:**
   - 每個特徵的公式寫成 LaTeX 格式
   - 解釋為什麼這些特徵對反轉預測有效
   - 附加學術參考文獻 (如果適用)

3. **使用示例:**
   ```python
   # 示例代碼
   df = load_binance_data('BTCUSDT', '15m')
   feature_matrix = build_feature_matrix(df)
   quality_report = validate_feature_quality(feature_matrix)
   ```

4. **集成到現有項目:**
   - 修改 `v1.2_label_system/feature_engineering.py` 添加新函數
   - 確保與現有的 `label_generator.py` 兼容
   - 更新 `config.yaml` (如需要)
   - 提供單元測試 (使用 pytest)

### 補充信息

目前系統的驗證指標如下 (使用推薦參數 fib=0.002, bb=0.002, zigzag=0.3):
- 進場比例: 12.5%
- 成功率: 38.5%
- 平均回報: 0.85%
- 盈利比率: 85.2%

希望新的特徵工程能進一步提升預測準確度。

---

# 第二階段：模型架構與元標記提示詞

## 適用場景
使用此提示詞時，要求 AI 設計或實現機器學習模型架構，特別是涉及:
- LSTM/GRU 時間序列模型
- 雙層元標記 (Meta-Labeling) 框架
- XGBoost 二層過濾
- 自定義損失函數

---

## 提示詞內容

### 背景信息

專案信息同上 (ZigZag v1.2.0)。

當前系統在數據處理層已完成:
- 特徵工程 (包含分數階差分、衍生品特徵等)
- 三重障礙法標籤生成
- 品質評分系統

現在需要構建預測模型層，採用「雙層元標記」架構:
1. **第一層 (Primary Model):** LSTM 預測價格方向 (多/空/無效)
2. **第二層 (Meta-Model):** XGBoost 判斷第一層的預測是否可信

### 任務要求

請擔任機器學習架構師，專精於金融時間序列預測。請基於 PyTorch 和 XGBoost 庫設計完整的雙層模型系統。

#### 需求 1: 三重障礙法標籤生成

**已有基礎:** `v1.2_label_system/entry_validator.py` 和 `label_generator.py`

**進一步要求:**
1. 優化標籤生成邏輯，確保:
   - **上漲止盈障礙:** 從進場點位上升 X% (建議 X=2-3%)
   - **下跌止損障礙:** 從進場點位下降 Y% (建議 Y=1-1.5%)
   - **時間垂直障礙:** 進場後 N 根 K 線 (建議 N=20 根，約 5 小時)

2. 標籤類別定義:
   - 類別 0: "無效" (未觸及任何障礙，時間到期)
   - 類別 1: "下跌" (觸及下跌障礙)
   - 類別 2: "上漲" (觸及上漲障礙)

3. 同時生成輔助標籤:
   - `触及障礙的時間步長` (用於時間預測)
   - `触及時的回報百分比` (用於回報預測)
   - `品質評分` (基於進場的信號強度)

#### 需求 2: LSTM 初級模型

**模型架構:**
```
Input (batch_size, seq_length, n_features)
  ↓
Embedding Layer (可選，用於時間特徵)
  ↓
LSTM Layer 1 (hidden_size=128, dropout=0.3)
  ↓
Attention Layer (計算時間步的權重)
  ↓
LSTM Layer 2 (hidden_size=64, dropout=0.3)
  ↓
Global Average Pooling
  ↓
Dense Layer 1 (32 neurons, ReLU)
  ↓
Dropout (0.3)
  ↓
Output Layer (3 neurons, Softmax)
  ↓
Output: [P(Down), P(Flat), P(Up)]
```

**詳細需求:**

1. **Attention 機制實現:**
   ```python
   # 計算 attention weights
   attention_scores = tanh(W * h_t + b)  # h_t 是 LSTM 隱狀態
   attention_weights = softmax(attention_scores)
   context_vector = sum(attention_weights * h_t)  # 加權求和
   ```

2. **數據準備:**
   - 序列長度 (seq_length): 50 根 K 線 (約 12.5 小時的 15m 數據)
   - 特徵數 (n_features): 根據第一階段的特徵工程結果
   - 訓練集/驗證集/測試集分割: 70% / 10% / 20%
   - 數據標準化: Z-score normalization (fit 在訓練集，apply 到全部)

3. **超參數:**
   - Learning Rate: 0.001 (Adam optimizer)
   - Batch Size: 64
   - Epochs: 100 (with early stopping, patience=10)
   - Loss Function: Categorical Cross-Entropy
   - Metric: Weighted F1-Score (考慮類別不平衡)

4. **類別不平衡處理:**
   - 計算每個類別的權重 (inverse frequency)
   - 在損失函數中使用 `class_weight` 參數
   - 輸出預測概率而非 one-hot 編碼

5. **模型評估指標:**
   - 精確度 (Precision)
   - 召回率 (Recall)
   - F1-Score (加權)
   - 混淆矩陣
   - ROC-AUC (一對多)

#### 需求 3: XGBoost 元模型 (Meta-Model)

**模型目的:**
XGBoost 的任務是判斷 LSTM 的預測是否可信，即判斷這筆交易是否應該執行。

**輸入特徵:**
1. LSTM 的預測結果:
   - `lstm_pred_class` (0, 1, 或 2)
   - `lstm_pred_prob_max` (最高概率值)
   - `lstm_pred_entropy` (預測的不確定性)

2. 市場特徵:
   - 當前波動率 (ATR)
   - 成交量與平均成交量的比值
   - 當前的資金費率
   - OI 的變化速率

3. 信號強度特徵:
   - 進場時斐波那契水平的接近度
   - 布林帶位置 (0-1 之間)
   - 最近 N 根 K 線的趨勢強度

**輸出:**
二分類: 0 = "不交易" (過濾掉)
        1 = "交易" (執行)

**超參數:**
```python
xgb_params = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
    'scale_pos_weight': weight_positive_class,  # 處理類別不平衡
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
}
```

#### 需求 4: 自定義損失函數

**設計原則:**
損失函數應該懲罰以下情況:
1. **方向錯誤:** 預測多但價格下跌
2. **高波動下的錯誤:** 波動率 > 某閾值時的錯誤應被加重
3. **錯過機會:** 標記為"無效"但實際上盈利很多

**自定義損失函數建議:**

```python
def custom_loss(y_true, y_pred, volatility, return_magnitude):
    """
    y_true: 真實標籤 (0=Down, 1=Flat, 2=Up)
    y_pred: 預測概率 (softmax 輸出)
    volatility: 當前波動率
    return_magnitude: 絕對回報幅度
    """
    base_ce_loss = categorical_cross_entropy(y_true, y_pred)
    
    # 高波動懲罰
    volatility_penalty = base_ce_loss * (1 + volatility / volatility_baseline)
    
    # 錯誤類型懲罰 (方向錯誤更嚴重)
    direction_penalty = base_ce_loss * (1 + return_magnitude) if wrong_direction else base_ce_loss
    
    return volatility_penalty + direction_penalty
```

#### 需求 5: 訓練流程

1. **數據準備階段:**
   ```python
   # 1. 加載特徵 + 標籤
   features = load_feature_matrix()  # 來自第一階段
   labels, auxiliary_labels = load_labels_with_quality()
   
   # 2. 移除低質量的樣本 (品質分數 < 30)
   valid_idx = auxiliary_labels['quality_score'] >= 30
   features_clean = features[valid_idx]
   labels_clean = labels[valid_idx]
   
   # 3. 分割數據
   X_train, X_test, y_train, y_test = train_test_split(
       features_clean, labels_clean, test_size=0.2, shuffle=False
   )
   ```

2. **LSTM 訓練:**
   ```python
   lstm_model = build_lstm_model()
   history = lstm_model.fit(
       X_train, y_train,
       validation_split=0.1,
       epochs=100,
       batch_size=64,
       callbacks=[early_stopping, lr_scheduler]
   )
   
   # 評估
   lstm_pred_train = lstm_model.predict(X_train)
   lstm_pred_test = lstm_model.predict(X_test)
   ```

3. **XGBoost 訓練:**
   ```python
   # 提取 LSTM 的預測特徵
   meta_features_train = extract_meta_features(lstm_pred_train, X_train)
   meta_features_test = extract_meta_features(lstm_pred_test, X_test)
   
   # 生成元標籤: 1 如果 LSTM 預測正確，0 否則
   lstm_pred_class = argmax(lstm_pred_train)
   meta_label_train = (lstm_pred_class == y_train).astype(int)
   
   # 訓練 XGBoost
   xgb_model = xgb.XGBClassifier(**xgb_params)
   xgb_model.fit(
       meta_features_train, meta_label_train,
       eval_set=[(meta_features_test, meta_label_test)],
       early_stopping_rounds=20
   )
   ```

#### 需求 6: 模型集成輸出

設計一個推理函數 `predict_with_confidence()`:

```python
def predict_with_confidence(x_new):
    """
    輸入: 新的特徵向量 (1, seq_length, n_features)
    輸出: 預測結果 + 置信度
    """
    # 第一層: LSTM 預測方向
    lstm_probs = lstm_model.predict(x_new)  # shape: (1, 3)
    lstm_class = argmax(lstm_probs)  # 0, 1, 或 2
    lstm_confidence = max(lstm_probs)
    
    # 第二層: XGBoost 判斷是否可信
    meta_features = extract_meta_features(lstm_probs, x_new)
    xgb_pred = xgb_model.predict(meta_features)  # 0 或 1
    xgb_prob = xgb_model.predict_proba(meta_features)[0, 1]
    
    # 綜合決策
    final_confidence = lstm_confidence * xgb_prob
    
    return {
        'direction': ['DOWN', 'FLAT', 'UP'][lstm_class],
        'lstm_confidence': float(lstm_confidence),
        'xgb_approval': bool(xgb_pred),
        'xgb_confidence': float(xgb_prob),
        'final_confidence': float(final_confidence),
        'should_trade': final_confidence > CONFIDENCE_THRESHOLD  # e.g., 0.75
    }
```

#### 需求 7: 模型持久化與版本控制

1. 保存模型:
   ```python
   # LSTM
   lstm_model.save('models/lstm_v1.2.0.h5')
   
   # XGBoost
   xgb_model.save_model('models/xgb_v1.2.0.json')
   
   # 元數據
   metadata = {
       'version': '1.2.0',
       'lstm_architecture': lstm_model.to_json(),
       'xgb_params': xgb_params,
       'feature_names': feature_names,
       'training_date': datetime.now(),
       'validation_metrics': {...}
   }
   save_json(metadata, 'models/metadata_v1.2.0.json')
   ```

### 輸出要求

1. **完整的模型代碼:**
   - PyTorch/TensorFlow 實現
   - 支持 GPU 和 CPU 運行
   - 完整的異常處理

2. **訓練流程:**
   - 數據加載和預處理
   - 超參數調整建議
   - 訓練監控 (TensorBoard 整合)

3. **評估報告:**
   - 混淆矩陣
   - ROC 曲線
   - 特徵重要性分析

4. **文檔:**
   - 模型架構圖
   - 超參數調整指南
   - 故障排除

---

# 第三階段：具體入場點位計算與輸出邏輯提示詞

## 適用場景
使用此提示詞時，要求 AI 實現策略執行層，將模型預測轉化為具體的交易指令。

---

## 提示詞內容

### 背景信息

前兩個階段已完成:
1. **數據層:** 完整的特徵工程
2. **模型層:** LSTM + XGBoost 雙層預測

現在需要實現「策略執行層」，將模型的概率輸出轉換為具體的交易建議，例如:
```
"ETH/USDT (15m): 強烈看漲。建議入場價: 3000 (基於強支撐)。止損: 2950。目標: 3050。模型置信度: 0.85。"
```

### 任務要求

請編寫一個完整的策略執行層模塊 `strategy_executor.py`，包含以下邏輯:

#### 需求 1: 入場點位計算

**輸入:**
1. 模型預測結果:
   - `direction`: 'UP' 或 'DOWN'
   - `final_confidence`: 0-1 置信度

2. 市場數據 (實時):
   - 當前價格
   - 訂單簿 (Bids 和 Asks)
   - 成交量分佈 (Volume Profile / VPVR)
   - 過去 4 小時的 K 線數據

3. 技術指標:
   - Fibonacci retracement levels (下跌幅度的 23.6%, 38.2%, 50%, 61.8%)
   - 布林帶 (上軌、中線、下軌)
   - 支撐和阻力位

**定價邏輯:**

**情況 1: 預測為 UP (看漲)**
```
1. 在訂單簿中尋找最大支撐位 (High Volume Node)
   - 掃描過去 4 小時的 VPVR
   - 找出成交量最多的價格區間
   - 將其設為 primary_support

2. 在 Fibonacci retracement 中尋找次要支撐
   - 過去 5 天的最高點 High
   - 過去 5 天的最低點 Low
   - 計算 Fib 水平: Low + (High - Low) * [0.236, 0.382, 0.5, 0.618]
   - 尋找最接近當前價格的 Fib 水平
   - 設為 secondary_support

3. 根據置信度決策:
   if confidence >= 0.85:
       entry_price = primary_support  # 使用主支撐
   elif confidence >= 0.75:
       entry_price = max(primary_support, secondary_support)  # 使用更強的支撐
   else:
       entry_price = (primary_support + current_price) / 2  # 折中策略

4. 確保 entry_price < current_price (只在價格回調時進場)
```

**情況 2: 預測為 DOWN (看跌)**
```
相同邏輯，但:
- 尋找阻力位而非支撐位
- entry_price > current_price
- 方向相反
```

#### 需求 2: 止損和止盈點位

**止損設置:**
```python
def calculate_stop_loss(direction, entry_price, volatility_atr):
    """
    基於 ATR (Average True Range) 設置動態止損
    """
    if direction == 'UP':
        # 看漲時，止損在進場下方 1.5 倍 ATR
        stop_loss = entry_price - 1.5 * volatility_atr
    else:
        # 看跌時，止損在進場上方 1.5 倍 ATR
        stop_loss = entry_price + 1.5 * volatility_atr
    
    return stop_loss
```

**止盈設置:**
```python
def calculate_take_profit(direction, entry_price, volatility_atr, risk_reward_ratio=2.0):
    """
    基於風險/收益比設置止盈
    standard: risk:reward = 1:2
    """
    risk = abs(entry_price - stop_loss)
    reward = risk * risk_reward_ratio
    
    if direction == 'UP':
        take_profit = entry_price + reward
    else:
        take_profit = entry_price - reward
    
    return take_profit
```

#### 需求 3: 過濾邏輯

**過濾規則 (滿足任一即不進場):**

1. **置信度過低:**
   ```python
   if final_confidence < 0.60:
       return "觀望，模型置信度不足 (< 0.60)"
   ```

2. **市場波動過大:**
   ```python
   if current_volatility > volatility_threshold:  # e.g., 95th percentile
       return "觀望，市場波動過大，暫不進場"
   ```

3. **成交量不足:**
   ```python
   if current_volume < average_volume * 0.5:
       return "觀望，成交量不足，信號不可靠"
   ```

4. **價格偏離過遠:**
   ```python
   # 對於看漲信號，如果當前價格已經很高，不進場
   if direction == 'UP' and current_price > entry_price * 1.05:
       return "觀望，價格已上漲，進場位不合理"
   ```

5. **時間限制:**
   ```python
   # 避免在特殊時間段進場 (例如: 新聞發佈前一小時)
   if is_high_impact_news_coming():
       return "觀望，待重大新聞公佈後"
   ```

#### 需求 4: 訂單簿深度分析

實現一個函數 `analyze_order_book()` 計算:

```python
def analyze_order_book(bids, asks, depth_levels=5):
    """
    bids: [{"price": 1.0, "quantity": 100}, ...]
    asks: [{"price": 1.1, "quantity": 100}, ...]
    
    返回:
    {
        'bid_ask_spread': ask[0] - bid[0],
        'bid_ask_spread_pct': (ask[0] - bid[0]) / bid[0],
        'bid_depth_volume': sum of bid quantities in top 5 levels,
        'ask_depth_volume': sum of ask quantities in top 5 levels,
        'imbalance_ratio': bid_volume / ask_volume,
        'liquidity_score': 1 - (imbalance_ratio).abs(),
    }
    """
```

#### 需求 5: 成交量分佈分析 (VPVR)

```python
def analyze_volume_profile(ohlcv_data, time_period='4h'):
    """
    計算過去 4 小時的成交量分佈
    
    返回:
    {
        'high_volume_nodes': [  # 成交量最集中的價格區間
            {'price_level': 3000, 'volume': 500000, 'percentage': 15.2},
            ...
        ],
        'poc': 3000,  # Point of Control (最多成交量的價格)
        'value_area': [2950, 3050],  # 70% 成交量的價格範圍
    }
    """
```

#### 需求 6: 輸出格式化

設計一個輸出函數 `format_trading_signal()`，返回人類可讀的信息:

```python
def format_trading_signal(signal_dict):
    """
    signal_dict 包含:
    - symbol: 'BTCUSDT'
    - timeframe: '15m'
    - direction: 'UP' 或 'DOWN'
    - entry_price: 1000.5
    - stop_loss: 990.2
    - take_profit: 1020.8
    - confidence: 0.85
    - reason: "基於 Fibonacci 支撐 + LSTM 上升信號"
    
    返回:
    """
    output = f"""
    =====================================================
    交易信號 - {signal_dict['symbol']} {signal_dict['timeframe']}
    =====================================================
    
    方向:        {signal_dict['direction']} (看漲 / 看跌)
    推薦入場價:  {signal_dict['entry_price']:.2f}
    止損價:      {signal_dict['stop_loss']:.2f}
    止盈價:      {signal_dict['take_profit']:.2f}
    
    風險:       {abs(signal_dict['entry_price'] - signal_dict['stop_loss']):.2f}
    收益:       {abs(signal_dict['take_profit'] - signal_dict['entry_price']):.2f}
    風險/收益:  1 : {abs(signal_dict['take_profit'] - signal_dict['entry_price']) / abs(signal_dict['entry_price'] - signal_dict['stop_loss']):.2f}
    
    模型置信度:  {signal_dict['confidence']:.1%}
    
    理由:        {signal_dict['reason']}
    
    =====================================================
    """
    
    return output
```

#### 需求 7: 實時監控和調整

實現一個監控函數 `monitor_open_positions()`:

```python
def monitor_open_positions(positions_list):
    """
    監控已開倉位:
    - 檢查是否觸及止損/止盈
    - 計算當前盈虧
    - 根據市場變化調整止損 (trailing stop)
    
    輸出:
    {
        'position_id': 'POS_001',
        'symbol': 'BTCUSDT',
        'entry_price': 1000.0,
        'current_price': 1010.0,
        'pnl': 10.0,
        'pnl_pct': 1.0,
        'stop_loss': 990.0,
        'take_profit': 1020.0,
        'status': 'MONITORING' / 'HIT_TP' / 'HIT_SL',
    }
    """
```

#### 需求 8: 完整的策略執行流程

```python
class StrategyExecutor:
    def __init__(self, model, config):
        self.lstm_model = model['lstm']
        self.xgb_model = model['xgb']
        self.config = config
        self.open_positions = []
    
    def execute_strategy(self, symbol, timeframe, current_market_data):
        """
        完整流程:
        1. 特徵提取
        2. 模型推理
        3. 信號生成
        4. 過濾
        5. 點位計算
        6. 輸出
        """
        # Step 1: 特徵提取
        features = self.extract_features(current_market_data)
        
        # Step 2: 模型推理
        prediction = self.predict_with_confidence(features)
        
        # Step 3: 過濾
        if not self.should_trade(prediction, current_market_data):
            return {"status": "FILTERED_OUT", "reason": "..."}
        
        # Step 4: 點位計算
        entry_price = self.calculate_entry_price(
            prediction['direction'],
            current_market_data['current_price'],
            current_market_data['order_book'],
            current_market_data['volume_profile']
        )
        
        stop_loss = self.calculate_stop_loss(
            prediction['direction'],
            entry_price,
            current_market_data['atr']
        )
        
        take_profit = self.calculate_take_profit(
            prediction['direction'],
            entry_price,
            stop_loss
        )
        
        # Step 5: 輸出
        signal = {
            'symbol': symbol,
            'timeframe': timeframe,
            'direction': prediction['direction'],
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence': prediction['final_confidence'],
            'timestamp': datetime.now(),
        }
        
        return format_trading_signal(signal)
    
    def monitor_positions(self):
        """
        監控所有開倉位
        """
        for position in self.open_positions:
            status = monitor_open_positions(position)
            if status['status'] == 'HIT_TP':
                self.close_position(position, 'TP')
            elif status['status'] == 'HIT_SL':
                self.close_position(position, 'SL')
```

### 輸出要求

1. **完整的執行層代碼:**
   - 模塊化設計
   - 易於集成到交易系統
   - 完整的日誌記錄

2. **計算函數:**
   - 支撐/阻力位計算
   - 止損/止盈計算
   - Fibonacci 重繪

3. **訊號模板:**
   - JSON 格式
   - 人類可讀文本
   - 實時推送格式

4. **集成示例:**
   ```python
   # 完整的端到端示例
   executor = StrategyExecutor(trained_models, config)
   signal = executor.execute_strategy('BTCUSDT', '15m', market_data)
   print(signal)
   
   # 輸出:
   # =====================================================
   # 交易信號 - BTCUSDT 15m
   # =====================================================
   # 方向:        UP (看漲)
   # 推薦入場價:  3000 (基於強支撐)
   # 止損價:      2950
   # 止盈價:      3050
   # 模型置信度:  0.85
   # =====================================================
   ```

---

# 使用指南

## 推薦使用順序

### 第一步: 數據層 (1-2 周)
- 使用「第一階段提示詞」
- 優化 `feature_engineering.py`
- 運行 `quick_test.py` 驗證特徵質量

### 第二步: 模型層 (2-3 周)
- 使用「第二階段提示詞"
- 實現 LSTM + XGBoost 模型
- 使用 `grid_search_params.py` 優化超參數

### 第三步: 策略層 (1 周)
- 使用「第三階段提示詞"
- 實現執行層
- 集成到交易系統

## 與 AI 的最佳實踐

1. **逐步提問:** 不要一次問整個系統，分解為小任務
2. **提供上下文:** 每次都附加項目信息和當前進度
3. **要求代碼範例:** 要求 AI 提供可運行的示例代碼
4. **驗證輸出:** 在使用前驗證 AI 生成的代碼
5. **版本控制:** 跟蹤所有模型版本和超參數

---

# 附錄

## 相關文件路徑

```
https://github.com/caizongxun/zong_zigzag
└── v1.2_label_system/
    ├── config.yaml                      # 配置 (推薦參數)
    ├── feature_engineering.py           # 特徵工程
    ├── label_generator.py               # 標籤生成
    ├── entry_validator.py               # 進場驗證
    ├── label_statistics.py              # 統計分析
    ├── fast_grid_search.py              # 快速搜索
    ├── quick_test.py                    # 快速驗證
    ├── test_btc_15m.py                  # BTC 測試
    └── examples/                        # 使用示例
```

## 推薦 AI 工具

1. **ChatGPT Plus (GPT-4)** - 最穩定，代碼生成能力強
2. **Claude 3** - 長上下文，適合複雜架構設計
3. **GitHub Copilot** - 與 IDE 集成，實時代碼補完
4. **Gemini Advanced** - 多模態，可提供流程圖

## 常見 Prompt 技巧

1. **角色扮演:** "請擔任資深量化工程師"
2. **提供約束:** "使用 PyTorch 和 XGBoost"
3. **要求格式:** "請提供可運行的代碼"
4. **版本控制:** "基於 v1.2.0 版本"
5. **錯誤修復:** "以上代碼有問題，請修改..."

---

**最後更新:** 2026-01-11
**版本:** 1.2.0
**作者:** ZigZag 開發團隊
