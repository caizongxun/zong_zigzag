# v1.2 標籤系統優化說明

## 修復內容

### 1. 修復 optimal_entry_return 計算邏輯 (Critical Bug)

**問題:**
原本的代碼在計算 `optimal_entry_return` 時使用了錯誤的邏輯:
```python
# 錯誤的邏輯
optimal_price = future_prices.min()  # 取最低點
return_pct = (optimal_price - entry_price) / entry_price  # 最低點相比進場的return
```

這導致所有的 return 都是負數或零,因為最低點永遠低於進場點!

**修復:**
現在改為取最高點,代表進場後能達到的最大盈利:
```python
# 正確的邏輯
max_price = future_prices.max()  # 取最高點
optimal_return = (max_price - entry_price) / entry_price  # 最高點的潛在回報
```

**含義:**
- `optimal_entry_return` 現在表示: "如果我在這個進場點買入,接下來20根K棒內能賺到的最大收益"
- 正值 = 有盈利潛力
- 負值 = 即使等最優時機也虧損(不應該發生)
- 0 = 沒有向上空間

### 2. 調整進場觸發條件

**之前的問題:**
- 91.10%的K棒都被標記為進場點 (太多了!)
- 說明 Bollinger Band 的接近度設定太寬鬆

**修改:**
```yaml
# 之前
fib_proximity: 0.01    # 1%
bb_proximity: 0.01     # 1%

# 現在
fib_proximity: 0.005   # 0.5% (更嚴格!)
bb_proximity: 0.005    # 0.5% (更嚴格!)
```

**效果:**
- 進場候選點會從 91% 降到約 8-15%
- 只有真正接近重要級別的K棒才算進場點
- 標籤更有區分度

### 3. 調整 ZigZag 閾值

**之前的問題:**
```
Latest swing high: 10282.67, low: 8290.0
```
這說明 ZigZag 閾值太高,只捕捉到6年的全局高低點,沒有本地oscillation

**修改:**
```yaml
# 之前
zigzag_threshold: 1.0

# 現在  
zigzag_threshold: 0.5  # 改小,捕捉更多本地swing
```

**效果:**
- Fibonacci 級別計算會基於更多的本地高低點
- 提高 Fibonacci 標籤的準確性

---

## 預期改進效果

修復前的結果:
```
Entry candidates: 201,140 (91.10%)  <- 太多
Success rate: 30.83%
Mean return: -0.70%                 <- 負數!
Profitable: 0 (0.00%)               <- 沒有盈利!
Max return: 0.00%
```

修復後的預期結果:
```
Entry candidates: ~20,000-30,000 (8-15%)  <- 正常比例
Success rate: 40-60%                       <- 更合理
Mean return: 1-3%                         <- 正數!
Profitable: 50-70%                        <- 大部分有利潤
Max return: 5-15%                         <- 真實潛力
```

---

## 重新運行測試

### 步驟1: 拉取最新代碼
```bash
cd C:\Users\zong\PycharmProjects\zong_zigzag
git pull origin v1.2-label-system
```

### 步驟2: 清除快取 (重要!)
```bash
cd v1.2_label_system
rm -r .cache  # Linux/Mac
rmdir /s .cache  # Windows
```

理由: 之前下載的數據可能被緩存了,需要清除以確保使用最新參數

### 步驟3: 運行測試
```bash
python test_btc_15m.py
```

### 步驟4: 驗證結果

查看這些關鍵指標是否改善:

```python
# 快速檢查腳本
import pandas as pd

# 讀取報告
import json
with open('./output/BTCUSDT_15m_report.json') as f:
    report = json.load(f)

print("修復後的結果檢查:")
print(f"進場比例: {report['entry_candidates']['candidate_pct']:.2f}%")
if report['entry_candidates']['candidate_pct'] < 20:
    print("  ✓ 通過 (目標 < 20%)")
else:
    print("  ✗ 失敗")

print(f"成功率: {report['entry_candidates']['success_rate']:.2f}%")
if 30 < report['entry_candidates']['success_rate'] < 70:
    print("  ✓ 通過 (目標 30-70%)")
else:
    print("  ✗ 失敗")

print(f"平均回報: {report['optimal_returns']['mean_optimal_return']:.2f}%")
if report['optimal_returns']['mean_optimal_return'] > 0:
    print("  ✓ 通過 (應該是正數)")
else:
    print("  ✗ 失敗")

print(f"盈利率: {report['optimal_returns']['profitable_pct']:.2f}%")
if report['optimal_returns']['profitable_pct'] > 30:
    print("  ✓ 通過 (目標 > 30%)")
else:
    print("  ✗ 失敗")

print(f"最大回報: {report['optimal_returns']['max_optimal_return']:.2f}%")
if report['optimal_returns']['max_optimal_return'] > 0:
    print("  ✓ 通過 (應該是正數)")
else:
    print("  ✗ 失敗")
```

---

## 如果結果還是不理想

### 情況1: 進場點還是太多 (>20%)

再進一步減小接近度:
```yaml
entry_validation:
  fib_proximity: 0.002   # 0.2%
  bb_proximity: 0.002    # 0.2%
```

### 情況2: 進場點太少 (<5%)

增加接近度:
```yaml
entry_validation:
  fib_proximity: 0.01    # 1%
  bb_proximity: 0.01     # 1%
```

### 情況3: 成功率太低 (<30%)

降低profit_threshold:
```yaml
entry_validation:
  profit_threshold: 1.0  # 改成只要利潤>風險就算成功
```

### 情況4: 還有負的return

檢查 feature_engineering.py 中的 Fibonacci 計算邏輯是否正確

---

## 代碼變更詳細說明

### entry_validator.py 中的修改

#### 修改1: label_optimal_entry 方法

```python
def label_optimal_entry(self, df, entry_idx, direction="long"):
    # ... 前面代碼不變 ...
    
    entry_price = df.iloc[entry_idx]["close"]
    future_prices = df.iloc[entry_idx:entry_idx + self.lookahead_bars]["close"]
    
    if direction == "long":
        # 修改: 從 min() 改為 max() - 取最大收益而不是最小價格
        max_price = future_prices.max()
        optimal_return = (max_price - entry_price) / entry_price
    else:
        min_price = future_prices.min()
        optimal_return = (entry_price - min_price) / entry_price
    
    return max_price if direction == "long" else min_price, optimal_return
```

#### 修改2: generate_all_labels 方法

沒有改變,只是現在調用修復後的 label_optimal_entry

### config.yaml 中的修改

```yaml
entry_validation:
  fib_proximity: 0.005   # 0.01 -> 0.005
  bb_proximity: 0.005    # 0.01 -> 0.005

indicators:
  zigzag_threshold: 0.5  # 1.0 -> 0.5
```

---

## 驗證修復效果的詳細指標

重新運行後檢查這些指標:

| 指標 | 修復前 | 修復後目標 | 檢查方法 |
|------|--------|-----------|----------|
| 進場比例 | 91.10% | 8-15% | `report['entry_candidates']['candidate_pct']` |
| 成功率 | 30.83% | 40-60% | `report['entry_candidates']['success_rate']` |
| 平均回報 | -0.70% | 1-3% | `report['optimal_returns']['mean_optimal_return']` |
| 最大回報 | 0.00% | 5-15% | `report['optimal_returns']['max_optimal_return']` |
| 盈利率 | 0.00% | 50-70% | `report['optimal_returns']['profitable_pct']` |
| 平均質量分 | 40.42 | 50-65 | `report['quality_scores']['mean_quality_score']` |

所有指標都改善則標籤系統正常工作。

---

## 後續改進方向

修復這兩個bug後,標籤系統應該能生成有效的標籤。接下來可以:

1. **為所有38個symbol生成標籤** - 使用 `generate_batch()` 或修改test腳本
2. **分析標籤分布** - 檢查不同symbol的標籤特性是否合理
3. **訓練分類器** - 用 entry_success 訓練 XGBoost 預測"這個進場點好嗎"
4. **訓練回歸模型** - 用 optimal_entry_price/return 預測最佳進場點
5. **回測策略** - 驗證標籤在實際交易中的有效性

---

## 快速檢查清單

修復後運行 test_btc_15m.py:

- [ ] 進場比例 8-15%
- [ ] 成功率 40-60%
- [ ] 平均回報 > 0%
- [ ] 最大回報 > 0%
- [ ] 盈利率 > 40%
- [ ] 平均質量分數 > 45
- [ ] 沒有負的 max_optimal_return
- [ ] Entry reason 分布合理 (Fib 和 BB 都有)

通過以上檢查後,標籤系統可以用於模型訓練。
