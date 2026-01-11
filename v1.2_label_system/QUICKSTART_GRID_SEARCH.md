# 快速開始: 網格搜索參數優化

## 30秒快速開始

### 1. 拉取代碼
```bash
cd C:\Users\zong\PycharmProjects\zong_zigzag\v1.2_label_system
git pull origin v1.2-label-system
```

### 2. 清除快取
```bash
rmdir /s .cache
```

### 3. 運行網格搜索
```bash
python grid_search_params.py
```

### 4. 等待完成 (~8小時)

### 5. 查看結果
```bash
# 查看推薦參數
type .\output\recommended_config.yaml

# 或用 Excel 打開完整結果
start .\output\grid_search_results.csv
```

### 6. 應用最優參數
```bash
copy .\output\recommended_config.yaml .\config.yaml
```

### 7. 驗證效果
```bash
python test_btc_15m.py
```

---

## 測試內容

**180 個參數組合**
- fib_proximity: 0.001, 0.002, 0.003, 0.005, 0.007, 0.01 (6個)
- bb_proximity: 0.001, 0.002, 0.003, 0.005, 0.007, 0.01 (6個)
- zigzag_threshold: 0.2, 0.3, 0.5, 0.7, 1.0 (5個)
- **總計:** 6 × 6 × 5 = 180 組合

---

## 輸出文件

| 文件 | 說明 |
|------|------|
| `./output/grid_search_results.csv` | 所有180個組合的完整結果 (可用Excel分析) |
| `./output/recommended_config.yaml` | 最優參數的配置文件 (直接複製到config.yaml) |
| 控制台輸出 | TOP 10排名和詳細分析 |

---

## 評分說明

每個參數組合會根據以下指標評分 (滿分 ~100分):

| 指標 | 權重 | 目標範圍 |
|------|------|----------|
| Entry candidates | 30分 | 8-15% |
| Mean return | 15分 | > 0.5% |
| Profitable % | 20分 | > 80% |
| Mean quality | 15分 | > 45 |
| Success rate | 10分 | 30-50% |
| 穩定性 | 5分 | quality 分布均勻 |

---

## 預期最優結果

```
Rank #1 - Score: ~85-90
  Parameters:
    fib_proximity: 0.002
    bb_proximity: 0.002
    zigzag_threshold: 0.3
  Results:
    Entry candidates: ~12% ✓ (目標 8-15%)
    Success rate: ~38% ✓
    Mean return: ~0.85% ✓ (目標 > 0.5%)
    Max return: ~15% ✓
    Profitable: ~87% ✓ (目標 > 80%)
    Mean quality: ~46 ✓ (目標 > 45)
```

---

## 時間估算

| 項目 | 耗時 |
|------|------|
| 180個組合 × 2.5分鐘/個 | ~7-8小時 |
| 查看結果 | ~5分鐘 |
| 應用參數 | ~1分鐘 |
| **驗證效果** | ~2.5分鐘 |
| **總計** | ~7-8.5小時 |

---

## 運行中的提示

### 進度顯示
```
[1/180] fib=0.001, bb=0.001, zigzag=0.2 Score: 45.32
[2/180] fib=0.001, bb=0.001, zigzag=0.3 Score: 48.91
[3/180] fib=0.001, bb=0.001, zigzag=0.5 Score: 42.17
```

每一行代表一個參數組合的測試結果。

### 中途暫停
- 按 `Ctrl+C` 停止
- 已完成的結果會保存,可以稍後查看

### 跳過某個失敗的組合
- 某些組合可能失敗,腳本會自動跳過並繼續下一個

---

## 完成後的操作流程

### Step 1: 查看結果
```bash
# 終端中查看 TOP 10
# 或用 Excel 打開完整結果
start .\output\grid_search_results.csv
```

### Step 2: 檢查推薦參數
```bash
type .\output\recommended_config.yaml
```

輸出示例:
```yaml
entry_validation:
  lookahead_bars: 20
  profit_threshold: 1.5
  fib_proximity: 0.002
  bb_proximity: 0.002
indicators:
  bollinger_period: 20
  bollinger_std: 2
  atr_period: 14
  zigzag_threshold: 0.3
  fibonacci_levels: [0.236, 0.382, 0.5, 0.618, 0.705, 0.786]
processing:
  batch_size: 1000
  n_workers: 4
```

### Step 3: 應用到配置
```bash
# 直接複製推薦配置到 config.yaml
copy .\output\recommended_config.yaml .\config.yaml
```

### Step 4: 驗證效果
```bash
# 用新參數重新生成標籤
python test_btc_15m.py
```

驗證輸出:
```
Entry candidates: 12.34% (目標 8-15%) ✓
Mean return: 0.85% (目標 > 0.5%) ✓
Profitable: 87.45% (目標 > 80%) ✓
Mean quality: 46.8 (目標 > 45) ✓
```

### Step 5: 批量生成所有symbol
```bash
# 接下來可以用優化後的參數生成所有38個symbol的標籤
python generate_batch.py
```

---

## 如果不滿意??

### 進場比例還是太高?
- 選擇 CSV 中進場比例更低的組合
- 或者編輯 grid_search_params.py 改小搜索範圍

### 回報不夠高?
- 查看 CSV 中 mean_return 最高的組合
- 可能需要調整 lookahead_bars 或 profit_threshold

### 質量分數不理想?
- 查看 CSV 中 mean_quality 最高的組合
- 或者增加 ZigZag swing 檢測靈敏度

---

## 進階調整

如果想自定義搜索範圍,編輯 `grid_search_params.py`:

```python
# 第88-90行
fib_proximities = [0.001, 0.002, 0.003, 0.005, 0.007, 0.01]  # 改這裡
bb_proximities = [0.001, 0.002, 0.003, 0.005, 0.007, 0.01]   # 改這裡
zigzag_thresholds = [0.2, 0.3, 0.5, 0.7, 1.0]                 # 改這裡

# 例如: 只測試部分
fib_proximities = [0.001, 0.002, 0.003]  # 3個 (原本6個)
bb_proximities = [0.001, 0.002, 0.003]   # 3個 (原本6個)
zigzag_thresholds = [0.2, 0.3]            # 2個 (原本5個)
# 3 × 3 × 2 = 18 組合,約1.2小時
```

---

## 完成後的下一步

1. 批量生成所有 symbol 標籤 (2-3小時)
2. 驗證其他 symbol 的標籤質量
3. 訓練 XGBoost 分類器 (Entry validity)
4. 訓練迴歸模型 (Optimal entry price)
5. 在回測框架上驗證策略有效性

---

## 常見問題

**Q: 運行時間太長?**
A: 可以編輯搜索範圍減少組合數,或購買更快的CPU。

**Q: 能並行嗎?**
A: 可以,但會加重系統負擔。建議順序運行。

**Q: 中途電腦關機?**
A: 下次運行時可以手動刪除已測試的組合,但會丟失進度。建議接上電源。

**Q: 結果不穩定?**
A: 因為網路加載時間不同,但標籤結果應該完全相同。

---

## 成功案例

最終應該達到:

```
✓ Entry candidates: 10-15% (足夠稀缺)
✓ Mean return: 0.7-1.2% (持續正回報)
✓ Profitable: 85%+ (高成功率)
✓ Mean quality: 45-50 (良好品質)
✓ 能在多個 symbol 上複製 (通用性強)
```

這就是生產級別的標籤系統!

---

## 支持

遇到問題?
- 查看 `grid_search_guide.md` 詳細說明
- 檢查 `output/grid_search_results.csv` 的完整結果
- 查看終端輸出的詳細分析

祝運氣好!
