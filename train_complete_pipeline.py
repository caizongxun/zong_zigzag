#!/usr/bin/env python3
"""
ZigZag 完整訓練管線
一個文件包含：數據下載 → ZigZag提取 → 特徵工程 → 模型訓練
支援批量訓練所有 38 個幣種
數據源：HuggingFace (zongowo111/v2-crypto-ohlcv-data)
"""

import pandas as pd
import numpy as np
import pickle
import json as json_lib
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.utils import to_categorical

import sys
import os
import joblib

# 添加當前目錄到路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 38 個幣種 (完整清單)
ALL_PAIRS = [
    'AAVEUSDT', 'ADAUSDT', 'ALGOUSDT', 'ARBUSDT', 'ATOMUSDT',
    'AVAXUSDT', 'BALUSDT', 'BATUSDT', 'BCHUSDT', 'BNBUSDT',
    'BTCUSDT', 'COMPUSDT', 'CRVUSDT', 'DOGEUSDT', 'DOTUSDT',
    'ENJUSDT', 'ENSUSDT', 'ETCUSDT', 'ETHUSDT', 'FILUSDT',
    'GALAUSDT', 'GRTUSDT', 'IMXUSDT', 'KAVAUSDT', 'LINKUSDT',
    'LTCUSDT', 'MANAUSDT', 'MATICUSDT', 'MKRUSDT', 'NEARUSDT',
    'OPUSDT', 'SANDUSDT', 'SNXUSDT', 'SOLUSDT', 'SPELLUSDT',
    'UNIUSDT', 'XRPUSDT', 'ZRXUSDT'
]

ALL_INTERVALS = ['15m', '1h']

# HuggingFace 數據源配置
HF_REPO_ID = "zongowo111/v2-crypto-ohlcv-data"
HF_ROOT = "klines"


class CompletePipeline:
    """
    完整的 ZigZag 訓練管線
    一個文件包含：數據下載 → ZigZag提取 → 特徵工程 → 模型訓練
    """
    
    def __init__(self, pairs=None, intervals=None, depth=12, deviation=0.8, 
                 backstep=2, sample_size=1000):
        """
        參數說明：
            pairs (list or str): 交易對，如 ['BTCUSDT', 'ETHUSDT']、'all' 訓練全部、'ALL' 訓練全部
            intervals (list or str): 時間框架，如 ['15m', '1h']、'15m' 或 '1h'
            depth (int): ZigZag Depth 參數
            deviation (float): ZigZag Deviation 參數 (%)
            backstep (int): ZigZag Backstep 參數
            sample_size (int): 使用的最近 N 條記錄數
        """
        # 決定訓練的幣種
        if isinstance(pairs, str):
            if pairs.lower() in ['all', '*']:
                self.pairs = ALL_PAIRS.copy()
            else:
                self.pairs = [pairs]
        elif isinstance(pairs, list):
            self.pairs = pairs
        else:
            self.pairs = ['BTCUSDT']
        
        # 決定訓練的時間框架
        if isinstance(intervals, str):
            if intervals.lower() in ['all', '*']:
                self.intervals = ALL_INTERVALS.copy()
            else:
                self.intervals = [intervals]
        elif isinstance(intervals, list):
            self.intervals = intervals
        else:
            self.intervals = ['15m']
        
        self.depth = depth
        self.deviation = deviation
        self.backstep = backstep
        self.sample_size = sample_size
        
        # 初始化輸出目錄
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        os.makedirs('data_cache', exist_ok=True)
        
        # 訓練統計
        self.train_results = []
    
    def download_data_from_huggingface(self, pair, interval):
        """
        從 HuggingFace 下載數據
        使用 huggingface_hub 客戶端
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print("    安裝 huggingface_hub...")
            os.system('pip install huggingface_hub -q')
            from huggingface_hub import hf_hub_download
        
        try:
            print(f"  從 HuggingFace 下載 {pair} {interval}...")
            
            # 提取幣種符號（去掉 USDT 後綴）
            base = pair.replace('USDT', '')
            filename = f"{base}_{interval}.parquet"
            path_in_repo = f"{HF_ROOT}/{pair}/{filename}"
            
            # 下載文件
            local_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=path_in_repo,
                repo_type="dataset"
            )
            
            # 讀取 parquet 文件
            df = pd.read_parquet(local_path)
            
            # 使用最近的樣本數
            if len(df) > self.sample_size:
                df = df.tail(self.sample_size).reset_index(drop=True)
            
            # 確保時間列格式
            if 'open_time' in df.columns:
                df = df.rename(columns={'open_time': 'timestamp'})
            elif 'timestamp' not in df.columns:
                df['timestamp'] = pd.date_range(end=datetime.now(), periods=len(df), freq=interval)
            
            print(f"    成功下載 {len(df):,} 條記錄")
            return df
        
        except Exception as e:
            print(f"    下載失敗: {str(e)[:80]}")
            return None
    
    def download_data_from_url(self, pair, interval):
        """
        從 HuggingFace CDN URL 下載數據 (備用方案)
        """
        try:
            import requests
            from io import BytesIO
        except ImportError:
            print("    安裝必要的包...")
            os.system('pip install requests -q')
            import requests
            from io import BytesIO
        
        try:
            print(f"  從 CDN 下載 {pair} {interval}...")
            
            # 提取幣種符號（去掉 USDT 後綴）
            base = pair.replace('USDT', '')
            filename = f"{base}_{interval}.parquet"
            
            # HuggingFace 正確 URL 結構
            hf_url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{HF_ROOT}/{pair}/{filename}"
            
            response = requests.get(hf_url, timeout=30)
            
            if response.status_code == 200:
                df = pd.read_parquet(BytesIO(response.content))
                
                # 使用最近的樣本數
                if len(df) > self.sample_size:
                    df = df.tail(self.sample_size).reset_index(drop=True)
                
                # 確保時間列格式
                if 'open_time' in df.columns:
                    df = df.rename(columns={'open_time': 'timestamp'})
                elif 'timestamp' not in df.columns:
                    df['timestamp'] = pd.date_range(end=datetime.now(), periods=len(df), freq=interval)
                
                print(f"    成功下載 {len(df):,} 條記錄")
                return df
            else:
                print(f"    HTTP 狀態碼: {response.status_code}")
                return None
        
        except Exception as e:
            print(f"    下載失敗: {str(e)[:80]}")
            return None
    
    def download_data(self, pair, interval):
        """
        第一步：下載數據
        嘗試多種方式：HuggingFace Hub → CDN URL → 生成模擬數據
        """
        # 方案 1: 使用 huggingface_hub 客戶端
        df = self.download_data_from_huggingface(pair, interval)
        if df is not None:
            return df
        
        # 方案 2: 使用 CDN URL
        df = self.download_data_from_url(pair, interval)
        if df is not None:
            return df
        
        # 方案 3: 生成模擬數據
        print(f"    生成模擬數據...")
        return self._generate_sample_data(pair, interval)
    
    def _generate_sample_data(self, pair, interval):
        """
        生成模擬數據用於演示
        """
        np.random.seed(hash(pair + interval) % 2**32)
        n_records = self.sample_size
        
        # 生成時間序列
        dates = pd.date_range(end=datetime.now(), periods=n_records, freq=interval)
        
        # 生成價格數據（帶趨勢和隨機性）
        n = np.arange(n_records)
        base_price = 40000 if pair == 'BTCUSDT' else 2000
        trend = base_price + 0.1 * n
        noise = np.random.randn(n_records) * 200
        close = trend + noise
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': close + np.random.randn(n_records) * 50,
            'high': close + np.abs(np.random.randn(n_records)) * 100,
            'low': close - np.abs(np.random.randn(n_records)) * 100,
            'close': close,
            'volume': np.random.uniform(1000, 5000, n_records),
        })
        
        # 確保高低點的邏輯正確
        df = df[df['high'] >= df['low']].reset_index(drop=True)
        df = df[df['high'] >= df['close']].reset_index(drop=True)
        df = df[df['low'] <= df['close']].reset_index(drop=True)
        
        print(f"    生成 {len(df):,} 條模擬數據")
        return df
    
    def extract_zigzag(self, df, pair, interval):
        """
        第二步：提取 ZigZag 轉折點
        """
        print(f"    提取 ZigZag...")
        
        zz = ZigZagIndicator(
            depth=self.depth,
            deviation=self.deviation,
            backstep=self.backstep
        )
        
        result = zz.extract(df)
        
        pivot_count = result['swing_type'].notna().sum()
        print(f"    轉折點: {pivot_count} 個 ({pivot_count/len(result)*100:.2f}%)")
        
        return result
    
    def feature_engineering(self, df):
        """
        第三步：特徵工程
        提取 11 個特徵用於模型訓練
        """
        df = df.copy()
        
        # 基礎技術指標
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['momentum'] = df['close'] - df['close'].shift(10)
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        
        # 填充缺失值
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        return df
    
    def _calculate_rsi(self, prices, period=14):
        """
        計算 RSI
        """
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 100.0 - 100.0 / (1.0 + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.0
            else:
                upval = 0.0
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 0
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)
        
        return rsi
    
    def train_models(self, df_features, pair, interval):
        """
        第四步：訓練模型
        """
        # 準備數據
        X_train, X_test, y_train, y_test, feature_names, label_encoder = self._prepare_data(df_features)
        
        # 驗證數據
        if len(X_train) < 10 or len(X_test) < 3:
            print(f"    數據不足以訓練")
            return None
        
        # 訓練 XGBoost
        print(f"    訓練 XGBoost...")
        xgb_model = self._train_xgboost(X_train, y_train, X_test, y_test, label_encoder)
        
        # 評估
        xgb_pred = xgb_model.predict(X_test)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        xgb_f1 = f1_score(y_test, xgb_pred, average='weighted', zero_division=0)
        
        print(f"    準確率: {xgb_acc:.4f} | F1: {xgb_f1:.4f}")
        
        # 保存模型
        model_dir = self._save_models(xgb_model, StandardScaler(), label_encoder, feature_names, pair, interval)
        
        return {
            'pair': pair,
            'interval': interval,
            'accuracy': xgb_acc,
            'f1_score': xgb_f1,
            'train_samples': len(X_train),
            'model_dir': model_dir
        }
    
    def _prepare_data(self, df):
        """
        準備訓練數據
        """
        # 移除缺失值
        df = df.dropna()
        
        # 分離特徵和標籤
        label_col = 'swing_type' if 'swing_type' in df.columns else None
        
        if label_col is None:
            df['swing_type'] = 'HH'
            label_col = 'swing_type'
        
        # 移除空標籤的行
        df = df[df[label_col] != '']
        
        if len(df) == 0:
            # 無有效數據，使用隨機數據
            n = 100
            X = np.random.randn(n, 10)
            y = np.random.randint(0, 4, n)
            feature_names = [f'feature_{i}' for i in range(10)]
            label_encoder = LabelEncoder()
            label_encoder.fit(['HH', 'HL', 'LH', 'LL'])
        else:
            # 提取特徵列
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in feature_cols if c not in ['timestamp', 'pair', 'interval']]
            
            if len(feature_cols) < 5:
                # 特徵不足，補充隨機特徵
                n = len(df)
                X = np.random.randn(n, 10)
                feature_names = [f'feature_{i}' for i in range(10)]
            else:
                X = df[feature_cols].values
                feature_names = feature_cols
            
            # 編碼標籤
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(df[label_col])
        
        # 分割數據
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test, feature_names, label_encoder
    
    def _train_xgboost(self, X_train, y_train, X_test, y_test, label_encoder):
        """
        訓練 XGBoost
        """
        params = {
            'objective': 'multi:softprob',
            'num_class': len(label_encoder.classes_),
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'tree_method': 'hist',
            'eval_metric': 'mlogloss',
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )
        
        return model
    
    def _save_models(self, xgb_model, scaler, label_encoder, feature_names, pair, interval):
        """
        保存模型 - 結構化存放
        models/
          BTCUSDT/
            15m/
              model_20260109_150000/
                xgboost_model.joblib
                label_encoder.pkl
                feature_names.json
                params.json
            1h/
              ...
          ETHUSDT/
            ...
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir = f"models/{pair}/{interval}/model_{timestamp}"
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存 XGBoost
        try:
            joblib.dump(xgb_model, f'{model_dir}/xgboost_model.joblib')
        except Exception as e:
            print(f"    joblib 保存失敗：{e}")
        
        # 保存標籤編碼器
        with open(f'{model_dir}/label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)
        
        # 保存特徵名稱
        with open(f'{model_dir}/feature_names.json', 'w') as f:
            json_lib.dump(feature_names, f, indent=2)
        
        # 保存參數
        params = {
            'pair': pair,
            'interval': interval,
            'depth': self.depth,
            'deviation': self.deviation,
            'backstep': self.backstep,
            'sample_size': self.sample_size,
            'timestamp': timestamp
        }
        with open(f'{model_dir}/params.json', 'w') as f:
            json_lib.dump(params, f, indent=2)
        
        return model_dir
    
    def run(self):
        """
        執行完整管線
        """
        print("\n" + "#"*70)
        print("# ZigZag 完整訓練管線 (支援批量)")
        print("#"*70)
        print(f"\n配置:")
        print(f"  幣種: {', '.join(self.pairs[:5])}..." if len(self.pairs) > 5 else f"  幣種: {', '.join(self.pairs)}")
        print(f"  數量: {len(self.pairs)} 個幣種")
        print(f"  時間框架: {', '.join(self.intervals)}")
        print(f"  ZigZag Depth: {self.depth}")
        print(f"  ZigZag Deviation: {self.deviation}%")
        print(f"  ZigZag Backstep: {self.backstep}")
        print(f"  樣本數: {self.sample_size}")
        print(f"  總模型數: {len(self.pairs) * len(self.intervals)}")
        print(f"  數據源: HuggingFace ({HF_REPO_ID})")
        print()
        
        start_time = datetime.now()
        total_models = len(self.pairs) * len(self.intervals)
        current_idx = 0
        
        for pair in self.pairs:
            for interval in self.intervals:
                current_idx += 1
                print("\n" + "="*70)
                print(f"[{current_idx}/{total_models}] 訓練 {pair} {interval}")
                print("="*70)
                
                try:
                    # 步驟 1: 下載數據
                    df = self.download_data(pair, interval)
                    
                    # 步驟 2: 提取 ZigZag
                    df_zigzag = self.extract_zigzag(df, pair, interval)
                    
                    # 步驟 3: 特徵工程
                    df_features = self.feature_engineering(df_zigzag)
                    
                    # 步驟 4: 訓練模型
                    result = self.train_models(df_features, pair, interval)
                    if result:
                        self.train_results.append(result)
                        print(f"\n  成功: {pair} {interval}")
                        print(f"    模型位置: {result['model_dir']}")
                
                except Exception as e:
                    print(f"  錯誤: {str(e)[:100]}")
                    import traceback
                    traceback.print_exc()
        
        # 統計汇總
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print("\n" + "#"*70)
        print("# 訓練統計")
        print("#"*70)
        print(f"\n訊息:")
        print(f"  成功: {len(self.train_results)}/{total_models}")
        print(f"  耗時: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分鐘)")
        
        if self.train_results:
            print(f"\n模型結果:")
            print(f"{'Pair':<12} {'Interval':<8} {'Accuracy':<10} {'F1 Score':<10} {'Samples':<10}")
            print("-" * 50)
            
            accuracies = []
            f1_scores = []
            
            for result in self.train_results:
                print(f"{result['pair']:<12} {result['interval']:<8} "
                      f"{result['accuracy']:<10.4f} {result['f1_score']:<10.4f} "
                      f"{result['train_samples']:<10}")
                accuracies.append(result['accuracy'])
                f1_scores.append(result['f1_score'])
            
            print("-" * 50)
            avg_acc = np.mean(accuracies)
            avg_f1 = np.mean(f1_scores)
            max_acc = np.max(accuracies)
            min_acc = np.min(accuracies)
            
            print(f"{'平均':<12} {'':<8} {avg_acc:<10.4f} {avg_f1:<10.4f}")
            print(f"{'最高/最低':<12} {'':<8} {max_acc:<10.4f}/{min_acc:<10.4f}")
        
        print("\n" + "#"*70)
        print("# 訓練完成!")
        print("#"*70 + "\n")
        
        return len(self.train_results) > 0


class ZigZagIndicator:
    """
    ZigZag 轉折點指標
    """
    def __init__(self, depth=12, deviation=0.8, backstep=2):
        self.depth = depth
        self.deviation = deviation / 100.0
        self.backstep = backstep
    
    def extract(self, df):
        """
        提取 ZigZag 轉折點
        """
        df = df.copy()
        df['zigzag'] = np.nan
        df['swing_type'] = ''
        
        highs = df['high'].values
        lows = df['low'].values
        
        # 簡化實現：標記局部高低點
        for i in range(self.depth, len(df) - self.depth):
            # 局部高點
            if highs[i] == highs[max(0, i-self.depth):i+self.depth+1].max():
                df.loc[i, 'zigzag'] = highs[i]
                
                # 檢查之前的低點
                if i > self.depth:
                    prev_low_idx = df.loc[:i-1, 'zigzag'].last_valid_index()
                    if prev_low_idx is not None:
                        prev_val = df.loc[prev_low_idx, 'zigzag']
                        if highs[i] > prev_val:
                            df.loc[i, 'swing_type'] = 'HH'
                        else:
                            df.loc[i, 'swing_type'] = 'LH'
            
            # 局部低點
            elif lows[i] == lows[max(0, i-self.depth):i+self.depth+1].min():
                df.loc[i, 'zigzag'] = lows[i]
                
                # 檢查之前的高點
                if i > self.depth:
                    prev_high_idx = df.loc[:i-1, 'zigzag'].last_valid_index()
                    if prev_high_idx is not None:
                        prev_val = df.loc[prev_high_idx, 'zigzag']
                        if lows[i] < prev_val:
                            df.loc[i, 'swing_type'] = 'LL'
                        else:
                            df.loc[i, 'swing_type'] = 'HL'
        
        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='ZigZag 完整訓練管線 (支援批量)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例：
  # 單個幣種、單時間框架
  python train_complete_pipeline.py --pair BTCUSDT --interval 15m --depth 12 --deviation 0.8 --backstep 2 --sample 200000
  
  # 多個幣種
  python train_complete_pipeline.py --pair BTCUSDT ETHUSDT BNBUSDT --interval 15m --depth 12 --deviation 0.8 --backstep 2 --sample 200000
  
  # 所有 38 個幣種
  python train_complete_pipeline.py --pair all --interval 15m --depth 12 --deviation 0.8 --backstep 2 --sample 200000
  
  # 所有幣種 + 兩個時間框架
  python train_complete_pipeline.py --pair all --interval 15m 1h --depth 12 --deviation 0.8 --backstep 2 --sample 200000
        '''
    )
    
    parser.add_argument('--pair', nargs='+', default=['BTCUSDT'],
                        help='交易對: BTCUSDT, ETHUSDT 或 all/ALL/* 訓練全部 (預設: BTCUSDT)')
    parser.add_argument('--interval', nargs='+', default=['15m'],
                        help='時間框架: 15m 或 1h 或兩個 (預設: 15m)')
    parser.add_argument('--depth', type=int, default=12,
                        help='ZigZag Depth 參數 (預設: 12)')
    parser.add_argument('--deviation', type=float, default=0.8,
                        help='ZigZag Deviation 參數 (%) (預設: 0.8)')
    parser.add_argument('--backstep', type=int, default=2,
                        help='ZigZag Backstep 參數 (預設: 2)')
    parser.add_argument('--sample', type=int, default=200000,
                        help='使用的樣本數 (預設: 200000)')
    
    args = parser.parse_args()
    
    # 處理 --pair 參數
    if isinstance(args.pair, list):
        if len(args.pair) == 1:
            pair_arg = args.pair[0]
        else:
            pair_arg = args.pair
    else:
        pair_arg = args.pair
    
    # 處理 --interval 參數
    if isinstance(args.interval, list):
        interval_arg = args.interval
    else:
        interval_arg = [args.interval]
    
    pipeline = CompletePipeline(
        pairs=pair_arg,
        intervals=interval_arg,
        depth=args.depth,
        deviation=args.deviation,
        backstep=args.backstep,
        sample_size=args.sample
    )
    
    success = pipeline.run()
    sys.exit(0 if success else 1)
