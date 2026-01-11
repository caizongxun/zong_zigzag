# v1.2 標籤系統

這個資料夾用於實現v1.2系統的標籤層(Label System)。

## 目錄結構

```
v1.2_label_system/
├── README.md                              # 本檔案
├── config.yaml                            # 全局配置
├── data_loader.py                         # HuggingFace數據加載
├── feature_engineering.py                 # 特徵工程
├── label_generator.py                     # 標籤生成核心邏輯
├── entry_validator.py                     # 進場驗證與標籤化
├── label_statistics.py                    # 標籤統計與分析
└── examples/
    └── example_label_generation.py        # 使用示例
```

## 核心概念

### 標籤層的三層標籤體系

#### 標籤1: 進場成功性 (Binary Classification)
標記一個進場點是否為「好的進場」
- 1: 好進場（進場後N根K棒內獲利 > 風險*1.5）
- 0: 壞進場（不滿足條件）

#### 標籤2: 進場品質分數 (Regression 0-100)
對進場點進行量化評分
- 最大獲利幅度 (40分)
- 最小回撤 (30分)
- 風險報酬比 (20分)
- 勝率 (10分)

#### 標籤3: 最優進場價位 (Regression)
在候選進場區域內,找出歷史上的最優進場價
- 多單：區間內最低價
- 空單：區間內最高價

### 進場候選點的定義

一個「進場候選點」由以下因素定義：

1. **Fibonacci 等級接近**
   - 計算最近的ZigZag swing
   - 生成Fib等級 (0.236, 0.382, 0.5, 0.618, 0.705, 0.786)
   - 檢測價格距離Fib等級 < 1%

2. **Bollinger Band 接近**
   - 計算動態BB (20期SMA, 2個標準差)
   - 檢測價格距離上/下軌 < 1%

3. **ZigZag 樞紐點**
   - HH (高點)、HL (高點後低點)、LH (低點後高點)、LL (低點)
   - 使用v1.0模型預測結果

### 訓練流程

```
1. 加載原始K線數據 (HuggingFace)
   ↓
2. 計算技術指標 (ZigZag, Fib, BB, ATR等)
   ↓
3. 識別進場候選點
   ↓
4. 為每個候選點生成三層標籤
   ↓
5. 統計與驗證
   ↓
6. 保存為訓練數據集
```

## 使用說明

### 快速開始

```python
from v1.2_label_system.data_loader import DataLoader
from v1.2_label_system.label_generator import LabelGenerator

# 1. 加載數據
loader = DataLoader(repo_id="zongowo111/v2-crypto-ohlcv-data")
df = loader.load_klines(symbol="BTCUSDT", timeframe="15m")

# 2. 生成標籤
generator = LabelGenerator(config_path="v1.2_label_system/config.yaml")
labeled_data = generator.generate_labels(
    df=df,
    symbol="BTCUSDT",
    timeframe="15m"
)

# 3. 查看結果
print(labeled_data.head())
print(f"好進場率: {(labeled_data['entry_success']==1).mean():.2%}")
```

### 配置文件 (config.yaml)

```yaml
# HuggingFace配置
huggingface:
  repo_id: "zongowo111/v2-crypto-ohlcv-data"
  hf_root: "klines"

# 進場驗證參數
entry_validation:
  lookahead_bars: 20          # 進場後看多少根K棒
  profit_threshold: 1.5       # 利潤需要勝風險多少倍
  fib_proximity: 0.01         # 距離Fib等級多近算接近 (1%)
  bb_proximity: 0.01          # 距離BB上/下軌多近算接近 (1%)

# 品質評分參數
quality_scoring:
  max_profit_weight: 40       # 最大獲利權重
  max_drawdown_weight: 30     # 最大回撤權重
  rrr_weight: 20              # 風險報酬比權重
  win_rate_weight: 10         # 勝率權重

# 技術指標參數
indicators:
  fibonacci_levels: [0.236, 0.382, 0.5, 0.618, 0.705, 0.786]
  bollinger_period: 20
  bollinger_std: 2
  atr_period: 14
```

## 輸出數據格式

生成的標籤數據包含以下欄位：

```python
# 原始OHLCV欄位
open_time, open, high, low, close, volume, ...

# 技術指標
fib_0_236, fib_0_382, fib_0_5, fib_0_618, fib_0_705, fib_0_786
bb_upper, bb_middle, bb_lower, bb_width
atr, zigzag_phase

# 進場候選標記
is_entry_candidate           # 是否為進場候選點
entry_reason                 # 進場原因 (fib/bb/zigzag)

# 三層標籤
entry_success               # 標籤1: 進場成功性 (0/1)
entry_quality_score         # 標籤2: 進場品質分數 (0-100)
optimal_entry_price         # 標籤3: 最優進場價位
optimal_entry_return        # 最優進場的回報率
```

## 下一步

1. **調整config.yaml參數** - 根據你的回測結果優化
2. **生成訓練數據** - 為所有38個symbol生成標籤
3. **訓練Entry Validity模型** - 使用生成的標籤訓練XGBoost/RF
4. **驗證標籤質量** - 檢查標籤的一致性和有效性

## 參考資料

- HuggingFace Dataset: zongowo111/v2-crypto-ohlcv-data
- Fibonacci研究: 支撑阻力預測 with LSTM
- Bollinger Band動態預測: 技術指標+ML整合
