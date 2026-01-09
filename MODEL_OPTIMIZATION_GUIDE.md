# ZigZag 模型優化指南

## 一、模型驗證方法

### 1. 快速驗證（使用預設參數）
```bash
python model_validation_visualization.py
```

這會自動：
- 找到最新的模型
- 加載訓練特徵
- 計算所有性能指標
- 生成混淆矩陣和可視化圖表
- 輸出詳細報告

### 2. 指定特定模型驗證
```bash
python model_validation_visualization.py --model-dir models/BTCUSDT_15m_20260109_033108
```

### 3. 生成交易信號（預測結果）
```bash
python model_validation_visualization.py --signal --threshold 0.7
```

**threshold 參數說明：**
- 只有當模型預測的置信度 > threshold 時才生成信號
- 推薦值：0.6-0.8（更高的閾值 = 更保守但更可靠）
- 輸出：`trading_signals.csv` 包含：
  - predicted_class: 預測的 ZigZag 類別
  - confidence: 預測置信度
  - signal: 交易信號 (BUY/SELL/HOLD)

## 二、性能指標解讀

### 關鍵指標

| 指標 | 含義 | 理想值 |
|------|------|--------|
| **Accuracy (準確率)** | 正確預測的比例 | > 70% |
| **Precision (精確率)** | 預測為正類的正確率 | > 70% |
| **Recall (召回率)** | 實際正類被正確預測的比例 | > 70% |
| **F1 Score** | 精確率和召回率的調和平均 | > 70% |
| **Macro F1** | 不同類別 F1 的平均值 | > 70% |

### 混淆矩陣
- **對角線（預測正確）**：越多越好
- **非對角線（預測錯誤）**：越少越好

### 類別分布
- 檢查 HH, HL, LH, LL 四個類別是否均衡
- 不均衡可能導致模型偏向多數類

## 三、優化策略

### 1. 模型參數優化

#### 調整 ZigZag 參數
```python
# 在 train_complete_pipeline.py 中修改
pipeline = CompletePipeline(
    pair='BTCUSDT',
    interval='15m',
    depth=12,           # ← 增大捕捉更多轉折點
    deviation=0.8,      # ← 降低增加靈敏度
    backstep=2,         # ← 調整回退步數
    sample_size=200000
)
```

**參數調優建議：**
- `depth`: 12 → 16 (更多候選轉折點)
- `deviation`: 0.8% → 0.5% (更敏感) 或 1.0% (更保守)
- `backstep`: 2 → 3 (更嚴格的驗證)

#### 調整 XGBoost 參數
```python
params = {
    'max_depth': 6,              # ← 6-8（防止過度擬合）
    'learning_rate': 0.1,        # ← 0.05-0.1（越小越穩定）
    'n_estimators': 100,         # ← 100-200（越多越好）
    'subsample': 0.8,            # ← 0.7-0.9（防止過度擬合）
    'colsample_bytree': 0.8,     # ← 0.7-0.9（特徵採樣）
}
```

### 2. 特徵優化

#### 增加更多技術指標
```python
# 在 _basic_feature_engineering 中添加
df['atr'] = calculate_atr(df, 14)  # 平均真實範圍
df['bbands_upper'] = calculate_bbands(df['close'], 20)[0]  # 布林帶
df['stochastic'] = calculate_stochastic(df, 14)  # 隨機指標
df['adx'] = calculate_adx(df, 14)  # 平均動向指標
df['obv'] = calculate_obv(df)  # 能量潮指標
```

#### 特徵工程最佳實踐
```python
# 1. 規範化特徵
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 2. 特徵選擇（移除不相關特徵）
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=10)
X = selector.fit_transform(X, y)

# 3. 特徵工程
# - 交叉項：price * volume
# - 比率：high / low
# - 變化率：pct_change
```

### 3. 數據質量優化

#### 類別均衡
```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

xgb_model.fit(
    X_train, y_train,
    sample_weight=class_weights[y_train],  # 添加樣本權重
    eval_set=[(X_test, y_test)]
)
```

#### 數據清理
```python
# 1. 移除極端異常值
Q1 = df['close'].quantile(0.25)
Q3 = df['close'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['close'] >= Q1 - 1.5*IQR) & (df['close'] <= Q3 + 1.5*IQR)]

# 2. 前向填充缺失值（保持時間序列連續性）
df = df.fillna(method='ffill')
```

### 4. 模型集成優化

#### 使用集成方法
```python
from sklearn.ensemble import VotingClassifier

# 訓練多個基礎模型
model1 = xgb.XGBClassifier(**params)
model2 = LGBMClassifier(**params)
model3 = RandomForestClassifier(**params)

# 組合成投票分類器
voting_clf = VotingClassifier(
    estimators=[('xgb', model1), ('lgbm', model2), ('rf', model3)],
    voting='soft'  # 使用概率投票
)
voting_clf.fit(X_train, y_train)
```

### 5. 驗證策略優化

#### 使用交叉驗證
```python
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

# 時間序列交叉驗證（保持時序順序）
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(
    xgb_model, X, y,
    cv=tscv,
    scoring='f1_weighted'
)
print(f"CV Scores: {scores}")
print(f"Mean F1: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

## 四、常見優化場景

### 場景 1：準確率低（< 65%）

**診斷：** 模型泛化能力差

**解決方案：**
1. 檢查數據質量（是否有缺失或異常值）
2. 增加訓練數據量 (sample_size 增加)
3. 調整 ZigZag 參數（更敏感）
4. 簡化模型 (max_depth 降低)
5. 增加正則化 (subsample/colsample 降低)

### 場景 2：特定類別識別差

**例如：LL（更低低點）識別率低**

**解決方案：**
1. 檢查該類別的樣本數（類別不均衡）
2. 使用 class_weights 加權
3. 調整 ZigZag deviation 參數
4. 創建特定於該類別的特徵

### 場景 3：過度擬合（訓練精度 > 95%, 測試 < 70%）

**診斷：** 模型在訓練集表現好但測試集差

**解決方案：**
1. 增加 L1/L2 正則化
2. 降低 max_depth
3. 增加 subsample 和 colsample_bytree
4. 使用早停（early stopping）
5. 增加訓練數據

## 五、實驗跟蹤

### 記錄實驗結果
```python
# 創建實驗日誌
experiments = [
    {
        'name': 'baseline',
        'params': {'depth': 12, 'deviation': 0.8},
        'accuracy': 0.8721,
        'f1_score': 0.8298,
        'notes': '初始模型'
    },
    {
        'name': 'depth_16_deviation_0.5',
        'params': {'depth': 16, 'deviation': 0.5},
        'accuracy': 0.8850,
        'f1_score': 0.8420,
        'notes': 'ZigZag 參數調整'
    },
]

import json
with open('experiments.json', 'w') as f:
    json.dump(experiments, f, indent=2)
```

### 對比分析
```bash
python model_validation_visualization.py --model-dir models/BTCUSDT_15m_experiment1
python model_validation_visualization.py --model-dir models/BTCUSDT_15m_experiment2
# 比較生成的報告
```

## 六、推薦優化流程

1. **基線測試** → 記錄現有性能
2. **單參數調優** → 逐個調整參數，記錄結果
3. **多參數組合** → 找到最佳參數組合
4. **特徵優化** → 添加高質量特徵
5. **模型集成** → 組合多個模型
6. **交叉驗證** → 驗證泛化能力
7. **生產部署** → 使用最優模型

## 七、進階技巧

### 1. 特徵重要性分析
```python
import matplotlib.pyplot as plt

# XGBoost 特徵重要性
xgb.plot_importance(xgb_model, top_n=15)
plt.show()

# 保留前 N 個重要特徵
top_features = xgb_model.get_booster().get_score(importance_type='weight')
selected_features = sorted(top_features.items(), key=lambda x: x[1], reverse=True)[:15]
```

### 2. SHAP 值解釋
```python
import shap

# 創建 SHAP 解釋器
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# 繪製 SHAP 摘要圖
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

### 3. 超參數網格搜索
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.15],
    'n_estimators': [100, 150, 200],
}

grid_search = GridSearchCV(
    xgb.XGBClassifier(),
    param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"最優參數: {grid_search.best_params_}")
print(f"最優分數: {grid_search.best_score_}")
```

## 八、性能基準

根據不同市場狀況的期望性能：

| 場景 | 準確率 | F1 Score | 備註 |
|------|--------|----------|------|
| 強勢上漲/下跌 | 85-95% | 0.82-0.92 | 趨勢明確 |
| 震盪市場 | 55-75% | 0.50-0.70 | 轉折點不明確 |
| 整體平均 | 70-80% | 0.65-0.78 | 生產環境目標 |

## 九、故障排除

### 問題：模型無法加載
```
解決：檢查 model_dir 是否正確，文件是否完整
python model_validation_visualization.py --model-dir <correct-path>
```

### 問題：記憶體不足
```
解決：減少 sample_size 或使用較小的 batch_size
python train_complete_pipeline.py --sample 50000
```

### 問題：結果波動大
```
解決：設置隨機種子、增加數據、使用交叉驗證
np.random.seed(42)
tf.random.set_seed(42)
```

---

更多信息請參考 README.md 或聯繫技術支持。
