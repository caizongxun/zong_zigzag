# Colab 完整指南 (Google Drive 方案 - 已修復)

## 問題解決

Colab 直接克隆 GitHub 會失敗 (網路限制)。

**解決方案:** 使用 Google Drive 上傳文件,完全迴避此問題。

---

## 準備工作 (5分鐘) - 本地電腦

### Step 1: 打開 Google Drive

https://drive.google.com

### Step 2: 新建文件夾

1. 右鍵點擊空白區域
2. 選擇「新建文件夾」
3. 命名為: `zong_zigzag_v1.2`

### Step 3: 上傳文件

從你的本地電腦上傳以下文件到該文件夾:

```
C:\Users\zong\PycharmProjects\zong_zigzag\v1.2_label_system\
  ├── grid_search_params.py      ← 必需
  ├── config.yaml                ← 必需
  ├── label_generator.py         ← 必需
  ├── data_loader.py             ← 必需
  ├── feature_engineering.py     ← 必需
  ├── entry_validator.py         ← 必需
  └── label_statistics.py        ← 必需
```

上傳方法:
- 打開 Google Drive 中的 `zong_zigzag_v1.2` 文件夾
- 點擊「新增」→ 「上傳文件"
- 選擇上述文件
- 等待上傳完成 (~2分鐘)

**確認:** 刷新頁面,看到所有文件都在 Drive 中

---

## Colab 操作 (按順序運行)

### Cell 1: 連接 Google Drive

```python
from google.colab import drive
import os

print("連接 Google Drive...")
drive.mount('/content/drive')

# 進入你上傳的目錄
work_dir = '/content/drive/My Drive/zong_zigzag_v1.2'
os.chdir(work_dir)

print(f"\n✓ 當前目錄: {os.getcwd()}")
print("\n✓ 目錄中的文件:")
!ls -la
```

**預期輸出:**
```
連接 Google Drive...
Mounted at /content/drive

✓ 當前目錄: /content/drive/My Drive/zong_zigzag_v1.2

✓ 目錄中的文件:
total 120
drwxr-xr-x 2 root root  4096 Jan 11 12:00 .
drwxr-xr-x 3 root root  4096 Jan 11 12:00 ..
-rw-r--r-- 1 root root 50000 Jan 11 12:00 config.yaml
-rw-r--r-- 1 root root 12000 Jan 11 12:00 grid_search_params.py
-rw-r--r-- 1 root root 15000 Jan 11 12:00 label_generator.py
...
```

---

### Cell 2: 安裝依賴

```python
print("安裝依賴...")
!pip install pyyaml pandas huggingface-hub datasets -q
print("✓ 依賴安裝完成")

# 驗證
import yaml
import pandas as pd
print("✓ pyyaml 已安裝")
print("✓ pandas 已安裝")
```

**預期輸出:**
```
安裝依賴...
✓ 依賴安裝完成
✓ pyyaml 已安裝
✓ pandas 已安裝
```

---

### Cell 3: 運行網格搜索 (主要步驟 - 會運行 8 小時)

```python
import subprocess
import os
from datetime import datetime

os.chdir('/content/drive/My Drive/zong_zigzag_v1.2')

print("="*70)
print("開始網格搜索")
print("="*70)
print(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"預計完成: {datetime.now().strftime('%Y-%m-%d')} 晚上 ~21:30")
print("="*70)
print("參數組合數: 180")
print("預計耗時: 6-8 小時")
print("="*70)
print()

# 運行網格搜索
print("運行 grid_search_params.py...\n")
result = subprocess.run(['python', 'grid_search_params.py'])

print()
print("="*70)
print(f"網格搜索完成! 結束時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)
```

**預期輸出:**
```
======================================================================
開始網格搜索
======================================================================
開始時間: 2026-01-11 13:50:00
預計完成: 2026-01-11 晚上 ~21:30
======================================================================
參數組合數: 180
預計耗時: 6-8 小時
======================================================================

運行 grid_search_params.py...

[1/180] fib=0.001, bb=0.001, zigzag=0.2 Score: 45.32
[2/180] fib=0.001, bb=0.001, zigzag=0.3 Score: 48.91
[3/180] fib=0.001, bb=0.001, zigzag=0.5 Score: 42.17
...
[180/180] fib=0.01, bb=0.01, zigzag=1.0 Score: 35.20

======================================================================
網格搜索完成! 結束時間: 2026-01-11 21:50:00
======================================================================
```

**注意:** 這個 Cell 會運行 8 小時。期間你可以:
- 關閉瀏覽器
- 關閉電腦
- 做其他事情

Colab 後台會繼續運行。

---

### Cell 4: 檢查是否完成 (8小時後運行)

```python
import os

work_dir = '/content/drive/My Drive/zong_zigzag_v1.2'
output_dir = f'{work_dir}/output'

print("檢查輸出文件...")
print()

if os.path.exists(output_dir):
    files = os.listdir(output_dir)
    if files:
        print(f"✓ 找到 {len(files)} 個文件:")
        for f in files:
            size = os.path.getsize(os.path.join(output_dir, f))
            print(f"  - {f} ({size/1024/1024:.2f} MB)")
        print("\n✓ 搜索已完成!")
    else:
        print("✗ output 目錄為空")
        print("搜索可能還在運行中...")
else:
    print("✗ output 目錄不存在")
    print("搜索可能還在運行中...")
```

---

### Cell 5: 查看推薦配置

```python
import os

work_dir = '/content/drive/My Drive/zong_zigzag_v1.2'
config_file = f'{work_dir}/output/recommended_config.yaml'

if os.path.exists(config_file):
    print("推薦配置文件:")
    print("="*70)
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
        print(content)
    print("="*70)
    print()
    print("✓ 推薦配置已生成!")
    print("接下來將下載到本地")
else:
    print("✗ 推薦配置還未生成")
    print("搜索可能還在運行中,請稍候...")
```

---

### Cell 6: 查看 TOP 10 結果

```python
import pandas as pd
import os

work_dir = '/content/drive/My Drive/zong_zigzag_v1.2'
csv_file = f'{work_dir}/output/grid_search_results.csv'

if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    
    # 按分數排序,取 TOP 10
    top10 = df.nlargest(10, 'score')
    
    print("TOP 10 最優參數組合:")
    print("="*120)
    print(top10[['fib_proximity', 'bb_proximity', 'zigzag_threshold', 
                 'entry_candidates_pct', 'success_rate', 'mean_return', 
                 'profitable_pct', 'mean_quality', 'score']].to_string(index=False))
    print("="*120)
    
    print()
    print(f"總共測試了 {len(df)} 個組合")
    print(f"最高分: {df['score'].max():.2f}")
    print(f"平均分: {df['score'].mean():.2f}")
    print(f"最低分: {df['score'].min():.2f}")
else:
    print("✗ 結果 CSV 還未生成")
    print("搜索可能還在運行中...")
```

---

### Cell 7: 下載文件到本地

```python
import pandas as pd
from google.colab import files
import os

work_dir = '/content/drive/My Drive/zong_zigzag_v1.2'
csv_file = f'{work_dir}/output/grid_search_results.csv'

if os.path.exists(csv_file):
    # 統計分析
    df = pd.read_csv(csv_file)
    
    print("📊 統計分析:")
    print("-"*70)
    print(f"測試組合總數: {len(df)}")
    print(f"平均分數: {df['score'].mean():.2f}")
    print(f"最高分數: {df['score'].max():.2f}")
    print(f"最低分數: {df['score'].min():.2f}")
    print()
    
    # 統計進場比例
    print("進場比例分布:")
    print(f"  < 10%:   {len(df[df['entry_candidates_pct'] < 10])} 個")
    print(f"  10-15%:  {len(df[(df['entry_candidates_pct'] >= 10) & (df['entry_candidates_pct'] <= 15)])} 個")
    print(f"  15-20%:  {len(df[(df['entry_candidates_pct'] > 15) & (df['entry_candidates_pct'] <= 20)])} 個")
    print(f"  > 20%:   {len(df[df['entry_candidates_pct'] > 20])} 個")
    print()
    
    print("下載文件到本地...")
    print("-"*70)
    print()
    
    # 下載推薦配置
    print("1️⃣ 下載 recommended_config.yaml")
    files.download(f'{work_dir}/output/recommended_config.yaml')
    
    # 下載完整結果
    print("2️⃣ 下載 grid_search_results.csv")
    files.download(f'{work_dir}/output/grid_search_results.csv')
    
    print()
    print("✓ 下載完成!")
    print()
    print("接下來的步驟:")
    print("1. 將 recommended_config.yaml 複製到本地的 config.yaml")
    print("2. 運行 python test_btc_15m.py 驗證效果")
    
else:
    print("✗ 結果還未生成")
    print("請等待搜索完成...")
```

---

## 完整流程總結

### 本地 (今天 13:40)
1. 打開 Google Drive
2. 新建文件夾 `zong_zigzag_v1.2`
3. 上傳 7 個 Python 文件
(5分鐘)

### Colab (今天 13:50)
1. ✅ 運行 Cell 1 (連接 Drive)
2. ✅ 運行 Cell 2 (安裝依賴)
3. ✅ 運行 Cell 3 (開始搜索 - 8小時)
4. ⏳ 等待完成

### Colab (明天 21:50)
5. ✅ 運行 Cell 4 (確認完成)
6. ✅ 運行 Cell 5 (查看推薦配置)
7. ✅ 運行 Cell 6 (查看 TOP 10)
8. ✅ 運行 Cell 7 (下載文件)

### 本地 (明天 22:00)
9. 將 recommended_config.yaml 複製到 config.yaml
10. 運行 python test_btc_15m.py

---

## 常見問題

**Q: 為什麼用 Google Drive?**
A: Colab 有網路限制,無法直接克隆 GitHub。Drive 是最穩定的方案。

**Q: 上傳文件要多久?**
A: 只有 7 個文件,總計幾 MB,通常 1-2 分鐘完成。

**Q: Cell 3 會運行 8 小時嗎?**
A: 是的,180 個參數組合 × 2.5分鐘/個 ≈ 7.5 小時。

**Q: 中途可以關閉瀏覽器嗎?**
A: 可以!Colab 後台會繼續運行。只需定期檢查 Drive 是否有 output 文件夾。

**Q: 結果會保存在哪?**
A: 既保存在 Google Drive (`/output` 文件夾),也會下載到本地 Downloads。

**Q: 如果 Cell 3 還沒完成怎麼辦?**
A: 等待即可。可以運行 Cell 4 檢查是否完成。

**Q: recommended_config.yaml 怎麼用?**
A: 將內容複製到本地的 `config.yaml`,然後運行 `test_btc_15m.py`。

---

## 預期時間

| 步驟 | 時間 | 說明 |
|------|------|------|
| 本地上傳文件 | 5分鐘 | Google Drive |
| Colab 連接 Drive | 2分鐘 | Cell 1 |
| 安裝依賴 | 3分鐘 | Cell 2 |
| 網格搜索 | 7.5小時 | Cell 3 |
| 檢查完成 | 1分鐘 | Cell 4 |
| 查看結果 | 2分鐘 | Cell 5, 6 |
| 下載文件 | 2分鐘 | Cell 7 |
| **總計** | **~7.5小時** | |

---

## 成功的標誌

✓ Cell 3 運行完成  
✓ output 文件夾出現在 Google Drive 中  
✓ 包含 `recommended_config.yaml` 和 `grid_search_results.csv`  
✓ Cell 7 成功下載文件到本地  
✓ 本地應用新配置後 test_btc_15m.py 的指標改善  

---

## 開始行動

1. 現在就上傳文件到 Google Drive (5分鐘)
2. 打開 Colab,粘貼代碼
3. 運行 Cell 1-3 (10分鐘)
4. 等待 8 小時
5. 下載結果並應用

祝運氣好!
