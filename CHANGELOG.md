# ZigZag反轉預測系統 - 版本更新日誌

## [v1.0.0] - 2026-01-10 (穩定版本)

### 完成功能

#### 核心指標模組
- 實現MT4風格的ZigZag算法
- 支援自訂參數: Depth、Deviation、Backstep
- 準確標記HH/HL/LL/LH轉折點
- 處理全量資料(219,643根K棒)
- 進度顯示與效能優化(20,600筆/秒)

#### 特徵工程(100+特徵)
- 技術指標(40+): SMA, EMA, MACD, RSI, Stochastic, Bollinger Bands, ATR等
- 價格行為(15+): 價格變動、K線實體、影線統計、Gap分析
- 滾動統計(60+): 多時間框架(5, 10, 20, 50)的統計特徵
- ZigZag歷史(10+): 轉折點距離、價格變動、歷史模式

#### 機器學習模型
- 混合架構: XGBoost(60%) + LSTM(40%)
- 時間序列驗證防止數據洩漏
- 預期準確率: 65-72%
- 預期F1 Score: 0.65-0.70

#### 工具與可視化
- 預測腳本: 批量預測與信心分數
- 圖表工具: 顯示最後N根K棒的標記狀況
- Web應用(Flask): 實時預測界面
- 驗證工具: 數據洩漏檢測與修復

#### 命令行支援
```bash
# ZigZag計算
python test_zigzag.py --depth 12 --deviation 1.0 --backstep 3 --all-data

# 模型訓練
python train_model.py

# 預測
python predict.py

# 視覺化
python visualize_zigzag.py --bars 300

# Web應用
python app.py
```

### 已解決的問題

1. **參數調優**: 從5% deviation調整至0.8-1.5%
   - 結果: 從1個轉折點增加至7,229個轉折點
   - 百分比: 2.6%的合理比例

2. **數據洩漏修復**
   - 原因: 使用未來數據進行特徵計算
   - 解決: 實現時間序列驗證
   - 驗證指標正常化(100%準確率改為65-72%)

3. **語法錯誤修正**
   - f-string中文字符轉義問題
   - 依賴套件檢查機制

### 技術規格

**資料集**
- 來源: Hugging Face (BTC 15分鐘K線)
- 規模: 219,643筆記錄 (~6年數據)
- 時間範圍: 2019-09-23 至 2025-12-30

**轉折點分布**
- 總計: 7,229個
- HH (更高高點): 1,743個 (24.1%)
- HL (更高低點): 1,904個 (26.3%)
- LH (更低高點): 1,850個 (25.6%)
- LL (更低低點): 1,730個 (23.9%)

**效能指標**
- 處理速度: 20,605筆/秒
- 訓練時間: 10-30分鐘
- 預測速度: 毫秒級

### 專案結構

```
zong_zigzag/
├── test_zigzag.py              # ZigZag指標計算
├── feature_engineering.py      # 特徵工程
├── train_model.py              # 模型訓練
├── train_complete_pipeline.py  # 完整Pipeline
├── predict.py                  # 預測腳本
├── visualize_zigzag.py         # 圖表工具
├── app.py                      # Web應用
├── verify_fix.py               # 驗證工具
├── CHANGELOG.md                # 本文件
├── README.md                   # 主文檔
├── README_ML.md                # ML文檔
├── DATA_LEAKAGE_FIX.md         # 修復說明
├── MODEL_OPTIMIZATION_GUIDE.md # 優化指南
├── TRAINING_REPORT.md          # 訓練報告
└── requirements.txt            # 依賴套件
```

### 下一步改進方向

1. **多幣種支援**: 擴展至其他加密貨幣(ETH、BNB等)
2. **不同時間框架**: 1H、4H、1D級別測試
3. **模型優化**: 
   - 測試Transformer架構
   - 集成更多技術指標
   - 參數超調優化
4. **實時交易**: 集成交易API進行回測
5. **部署優化**: Docker容器化、模型量化

### 安裝與使用

```bash
# 安裝依賴
pip install -r requirements.txt

# 快速開始
python test_zigzag.py --all-data --depth 12 --deviation 1.0 --backstep 3
```

### 貢獻者

zong (caizongxun)

### 許可證

MIT License

---

## 版本號說明

遵循語義化版本控制(Semantic Versioning):
- **主版本**: 重大更新或不兼容更改
- **次版本**: 新增功能,向後兼容
- **修訂版**: 修復bug,向後兼容
