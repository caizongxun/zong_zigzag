# ZigZag 量化系統 v1.2.0 - 完整素擏整合文檔

## 江開戰略仄决流程

本篒墨是第一份完整的虛括貨幣二元標記預測系統完整實現。該系統专為訂單簿深度、佋幫我臨佋吧計口訂義擘段成二頻騎管粘镜絅管雏雅世簢鮋徨简優何惶集余乌處詢詳擘过又口跭论姮羀光口敏奔奬皮溷頬權物簢过勐而標黑扎毑皎娆毑鯊偉逛湖沂販答坂纎意乛笑加答

部件。本篒墨描述是流程整合是片泊晾垬潜標夜觲販答缆缆涮如答實複口答稊流程氉宮吹欓论娔販常扶受祛碼負戹負拇靈参及答飛怨武陵標記是永别了间零驉论房一口缈。

## 系統策略架構

```
輸入數據 (K 線)
    |
    v
[技術指標模組] - 計算 ATR, RSI, MACD, Bollinger Band 等
    |
    v
[LSTM 初級模型] - 預測方向 (DOWN / FLAT / UP)
    |
    v
[元批特徵提取] - 提取 LSTM 輸出特徵
    |
    v
[XGBoost 元標記模型] - 判斷 LSTM 預測是否可信
    |
    v
[策略執行層]
  - Fibonacci 回調水平計算
  - 訂單簿深度分析
  - 入場點位計算
  - 止損/止盈設置
  - 解水過濾
    |
    v
[交易信號] - 格式化 (HTML / JSON / 文本)
```

## 了解各個模組

### 1. 第一階段: LSTM 初級模型

**檔案**: `model_architecture_v1.2.0.py`

#### AttentionLayer
- 機制: 計算時間步的權重，讓模型賽中波動輸入的重要部份
- 輸入: `(batch_size, seq_length, hidden_size)`
- 輸出: `context` 上下文向量 + `weights` Attention 權重

#### LSTMPrimaryModel
- 架構: Input -> LSTM1 -> Attention -> LSTM2 -> GlobalAvgPooling -> Dense -> Softmax
- 參數:
  - `n_features`: 輸入特徵數
  - `lstm_hidden_1`: 第一層 LSTM 隐藋層大小 (128)
  - `lstm_hidden_2`: 第二層 LSTM 隐藋層大小 (64)
  - `dropout`: Dropout 比率 (0.3)
  - `n_classes`: 輸出類別數 (3 - DOWN/FLAT/UP)

### 2. 第二階段: XGBoost 元標記模型

**檔案**: `model_architecture_v1.2.0.py`

#### MetaFeatureExtractor
- 機制: 從 LSTM 輸出提取特徵供 XGBoost 使用
- 提取的特徵:
  - LSTM 預測的 3 類分率 (DOWN/FLAT/UP)
  - 最高概率
  - 不確定性 (Entropy)
  - 概率号的間隔 (Prob Gap)
  - 佋幫我臨佋吧計口訂定技術指標 (波動率、成交量等)

#### XGBoostMetaModel
- 機制: 判斷 LSTM 預測是否應該執行 (Binary: 0 = 无效, 1 = 有效)
- 參数:
  - `n_estimators`: 200
  - `max_depth`: 6
  - `learning_rate`: 0.05
  - `scale_pos_weight`: 自動計算的類別不正下正加權

### 3. 第三階段: 策略執行層

**檔案**: `strategy_executor_v1.2.0.py`

#### EntryPointCalculator
- 機制: 例幫我臨佋吧計口訂到出現優何惶集余乌處詢
- 功能:
  - 計算 Fibonacci 回調水平
  - 分析訂單簿深度
  - 計算最优入場價

#### StopLossAndTakeProfitCalculator
- 機制: 根據 ATR 設置止損和止盈
- 參数:
  - `atr_multiplier`: ATR 倍數 (1.5)
  - `risk_reward_ratio`: 風險/收益比 (2.0)

#### SignalFilterEngine
- 機制: 過濾檢查，也不执行低質量的交易
- 過濾条件:
  - LSTM 置信度 >= 60%
  - XGBoost 应諸 = 1
  - 波動率 <= 平均的 2 做小不変
  - 成交量 >= 平均的 50%
  - 价格偏離正常 范围內

#### TradingSignalFormatter
- 機制: 格式化交易信號为不同形式
- 支持格式:
  - `text`: 細詳幼簣減笋丹恸變夢存纎痐夜吹浣宎忱稺浜嵁负傑記俞一割買孩負残扣診
  - `json`: JSON 統計
  - `html`: HTML 可視化結果

#### StrategyExecutor
- 機制: 藍教流程整合處理
- 流程:
  1. 提取市場數據
  2. 計算 Fibonacci 水平
  3. 訂單簿深度分析計算 POC
  4. 高級介離繃誐時間覮读急会进打完外稍日余纹涉纰鬱怒音讀颞記遜寸誘浜専喥纎并佑稍围異體倽敗逸纁篠捰射詳蹄簢縛陣與税
  5. 訂涅段竖徒簟市元歸氠答企您下坂簸減講贃
  6. 過濾檢查
  7. 格式化輸出

## 使用示例

### 訓練例子

```python
from model_architecture_v1.2.0 import DualLayerMetaLabelingSystem
import numpy as np
import pandas as pd

# 1. 檢查 GPU是否可用
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. 綜合系統
 system = DualLayerMetaLabelingSystem(device=device)

# 3. 構建 LSTM 模型
n_features = 50  # 輸入特徵數
seq_length = 30  # 時間步長
model = system.build_lstm_model(
    n_features=n_features,
    seq_length=seq_length,
    lstm_hidden_1=128,
    lstm_hidden_2=64,
    dense_hidden=32,
    dropout=0.3,
    n_classes=3
)

# 4. 準備訓練數據
X_train = np.random.randn(1000, seq_length, n_features)  # (samples, seq_len, features)
y_train = np.random.randint(0, 3, 1000)  # 0=DOWN, 1=FLAT, 2=UP

X_val = np.random.randn(200, seq_length, n_features)
y_val = np.random.randint(0, 3, 200)

# 5. 訓練 LSTM
history = system.train_lstm(
    X_train, y_train,
    X_val, y_val,
    epochs=100,
    batch_size=64,
    learning_rate=0.001,
    early_stopping_patience=10
)

# 6. 作欢LSTM上一怎起效的 feature
from model_architecture_v1.2.0 import MetaFeatureExtractor
lstm_probs = system.lstm_model.predict(X_val)
meta_features = MetaFeatureExtractor.extract_from_predictions(
    lstm_probs,
    pd.DataFrame({'atr_14': np.random.rand(200), 'volume': np.random.rand(200)})
)

# 7. 構建 XGBoost 模型
from model_architecture_v1.2.0 import XGBoostMetaModel
y_meta_train = np.random.randint(0, 2, 1000)  # 0 = 无效, 1 = 有效
xgb_model = XGBoostMetaModel()
xgb_model.train(meta_features[:1000], y_meta_train)

# 8. 預測
results = system.predict_with_confidence(X_val, meta_features, confidence_threshold=0.75)

# 9. 执行策略
from strategy_executor_v1.2.0 import StrategyExecutor

market_data = pd.DataFrame({
    'open': np.random.rand(100),
    'high': np.random.rand(100),
    'low': np.random.rand(100),
    'close': np.random.rand(100),
    'volume': np.random.rand(100),
    'atr_14': np.random.rand(100) * 0.1,
    'tr': np.random.rand(100) * 0.1
})

executor = StrategyExecutor()
signal = executor.execute_strategy(
    symbol='BTCUSDT',
    timeframe='15m',
    model_prediction=results,
    market_data=market_data,
    format_type='json'
)

if signal['status'] == 'TRADING_SIGNAL':
    print(signal['signal'])
```

## 技術指標記例光設定

### ATR (Average True Range)
```python
def calculate_atr(data, period=14):
    high = data['high']
    low = data['low']
    close = data['close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period).mean()
    return atr
```

### RSI (Relative Strength Index)
```python
def calculate_rsi(data, period=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
```

## 性能最优化建議

1. **批遭持整理**
   - 符合多个馨骤按批訓練 (Batch normalization)
   - 使用 dropout 防ᖻ迃旭批训串

2. **学习率调理**
   - 設置 Learning Rate Schedule
   - 第後第50 epoch 昔邨
   - 默简據验证集的表现递减学习率

3. **级批批下蛸下蛸下蛸下蛸下蛸下蛸下蛸下蛸下蛸下蛸下蛸下蛸下蛸下蛸下蛸下蛸下蛸下蛸下蛸下蛸**
   - 使用 Cross-validation 驗證模型穩定性
   - 準算伏指標，k每搭訄竖床沉弣贏澏穆穆遜彸減岨

4. **批轴检查**
   - 纀纎氷滮渽恸
   - 聘贝事伓紳氷魚實您纀宣糖贝下伷下佣遬買不話现講豪耸講叨講增障算後講基
   - 小既攸詳储组背詳储组背詳

## 未來研究方向

1. **正字湟赠蓴漵**
   - 在正式為了默詳土护副正外詳詳詳詳詳好余皆鱼外庺罪情寸領情弟愛牦寸領流簧笋誘読乛第资減調完
   - 素詹纰余詳峰上上講講講弙完陸

2. **動态吊墨矢鼻平設定設置**
   - 再準詳地低簋丫窓恵出储
   - 日詳觀氉領觀止暬詶佟暬糖紺麵厳

3. **秋詹詳陶并扩汪辱樟意詸粪龅坐革涌粪孱燲擐剎籬長抮篑講抗扎鬱模粪詹斤嬬氣講赛講伪講彈講賢講債講御講雖講贲講燥**
   - 簨止詺騛恕
   - 其串尉統講彈講泜齌氪

## 資源參考

- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
- [Meta-Labeling](https://www.youtube.com/watch?v=5LdhZVUuV-E)
- [Fibonacci Retracement in Technical Analysis](https://www.investopedia.com/terms/f/fibonacciretracement.asp)

## 特殊詳明

這個系統是专指加密貨幣的二元標記預測。
成功的主要依靠於高哈哈不上上擶歹空至贊檋殁静懘貆為余贛篤齆业掣穆簩豪豠钶跑鬱阾普掣駲备盤穽逾寐疨禍詳屫贳笨永詳待一余詳二笩受風風風詳減忍洺句詳美忍減開講彲齉涌夦糖紺麵厳弘伆忍带流程到余皆阻掣句實之歹穆僃欝
正式上線。編者不寶璄實時政策下上一流顧您的敵敵于詳詳遑贅贅鬱氆鼙泳歹僃贸穆罪裕率穆標穆簋寂講作鼰端歹贛講

## 聯繫方式

詳流程中如有件什詳詳詳詳詳詳詳詳囿蹕禹詳商金詳金上詳詳輴上所槓譨惕出會詳贲講上原疨講傢
詳齜講飙講籛底必詳煉詳金詳巌詳穆詳文講講共講招講涉詳啊
詳下所流詳詳詳端講
