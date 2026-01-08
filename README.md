# ZigZag 深度學習預測模型 (ZZZ)

本專案旨在解決 ZigZag 指標的滯後性問題，透過深度機器學習模型預測 HH (Higher High)、HL (Higher Low)、LL (Lower Low)、LH (Lower High) 信號。

## 問題描述

ZigZag 指標在技術分析中廣泛使用，但存在嚴重的滯後性問題：

- 必須等待價格反轉達到 Deviation 閾值
- 需要經過 Backstep 周期後才確認信號
- 當信號產生時，行情可能已經快結束

本專案使用深度學習模型提前預測 ZigZag 轉折點，減少滯後性對交易策略的影響。

## 專案架構

```
zong_zigzag/
├── README.md                    # 專案說明文件
├── test_zigzag.py              # 根目錄測試檔案(驗證標記邏輯)
├── 01_data_fetching/           # 資料獲取模組
│   ├── fetch_hf_data.py
│   └── data_validator.py
├── 02_feature_engineering/     # 特徵工程
│   ├── zigzag_labeler.py       # ZigZag標記器
│   └── feature_generator.py
├── 03_model_training/          # 模型訓練
│   ├── dataset_builder.py
│   ├── model_architecture.py
│   └── train.py
├── 04_inference/               # 預測推論
│   └── predict.py
└── utils/                      # 通用工具
    └── config.py
```

## ZigZag 標記邏輯

本專案實現 MT4 風格的 ZigZag 算法，核心參數如下：

### 參數說明

- **Depth (預設 12)**: 定義回溯多少根 K 棒來判定高低點
- **Deviation (預設 5%)**: 價格變動的最小百分比閾值
- **Backstep (預設 2)**: 連續極值點之間的最小 K 棒間隔

### HH/HL/LL/LH 判斷邏輯

1. **當方向為向下 (direction < 0) 時形成頂部**:
   - 若當前頂部價格 < 前一個頂部價格 → **LL** (Lower Low, 較低低點)
   - 若當前頂部價格 > 前一個頂部價格 → **HL** (Higher Low, 較高低點)

2. **當方向為向上 (direction > 0) 時形成底部**:
   - 若當前底部價格 > 前一個底部價格 → **HH** (Higher High, 較高高點)
   - 若當前底部價格 < 前一個底部價格 → **LH** (Lower High, 較低高點)

## 快速開始

### 安裝依賴

```bash
pip install pandas numpy requests pyarrow
```

### 執行測試

```bash
python test_zigzag.py
```

此指令將：
1. 從 Hugging Face 下載 BTC 15分鐘數據
2. 應用 ZigZag 標記算法
3. 統計 HH/HL/LL/LH 信號分佈
4. 儲存結果至 `zigzag_result.csv`

## 資料來源

本專案使用來自 Hugging Face 的加密貨幣 OHLCV 數據：

- 根目錄: https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data
- BTC 15m 數據: https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data/blob/main/klines/BTCUSDT/BTC_15m.parquet

## 開發進度

- [x] 創建 GitHub 倉庫
- [x] 實現 MT4 風格 ZigZag 標記算法
- [x] 建立測試檔案驗證標記邏輯
- [ ] 建立資料處理流程
- [ ] 設計特徵工程
- [ ] 實現深度學習模型
- [ ] 訓練與驗證

## 授權

本專案基於 Mozilla Public License 2.0 授權，ZigZag++ Pine Script 原始代碼來自 Dev Lucem。

## 貢獻

歡迎提交 Issue 和 Pull Request。
