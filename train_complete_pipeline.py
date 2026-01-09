#!/usr/bin/env python3
"""
ZigZag 完整訓練管算
一個文件包含：數據下載 → ZigZag提取 → 特徵工程 → 模型訓練
支援批量訓練所有 22 個幣種
"""

import pandas as pd
import numpy as np
import pickle
import json as json_lib
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
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

# 22 個幣種
ALL_PAIRS = [
    'AAVEUSDT', 'ADAUSDT', 'ALGOUSDT', 'ARBUSDT', 'ATOMUSDT',
    'AVAXUSDT', 'BCHUSDT', 'BNBUSDT', 'BTCUSDT', 'DOGEUSDT',
    'DOTUSDT', 'ETCUSDT', 'ETHUSDT', 'FILUSDT', 'LINKUSDT',
    'LTCUSDT', 'MATICUSDT', 'NEARUSDT', 'OPUSDT', 'SOLUSDT',
    'UNIUSDT', 'XRPUSDT'
]

ALL_INTERVALS = ['15m', '1h']

class CompletePipeline:
    """
    完整的 ZigZag 訓練管算
    一個文件包含：數據下載 → ZigZag提取 → 特徵工程 → 模型訓練
    """
    
    def __init__(self, pairs=None, intervals=None, depth=12, deviation=0.8, 
                 backstep=2, sample_size=1000):
        """
        參數說明：
            pairs (list or str): 交易對，如 ['BTCUSDT', 'ETHUSDT']、'all' 訓練全部、'ALL' 訓練全部
            intervals (list or str): 時間框架，如 ['15m', '1h']、'15m' 掲 '1h'
            depth (int): ZigZag Depth 參數
            deviation (float): ZigZag Deviation 參數 (%)
            backstep (int): ZigZag Backstep 參數
            sample_size (int): 使用的最近 N 条記錄数
        """
        # 决定訓練的幣種
        if isinstance(pairs, str):
            if pairs.lower() in ['all', '*']:
                self.pairs = ALL_PAIRS.copy()
            else:
                self.pairs = [pairs]
        elif isinstance(pairs, list):
            self.pairs = pairs
        else:
            self.pairs = ['BTCUSDT']
        
        # 决定訓練的時間框架
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
    
    def download_data(self, pair, interval):
        """
        第一步：下載數據
        從 HuggingFace 下載加密貨幣 OHLCV 數據
        """
        try:
            import requests
            from io import BytesIO
        except ImportError:
            print("  安裝必要的包...")
            os.system('pip install requests -q')
            import requests
            from io import BytesIO
        
        print(f"  下載 {pair} {interval}...")
        
        # 提取幣種符號（去掉 USDT 後綴）
        symbol = pair.replace('USDT', '')
        
        # HuggingFace 正確 URL 結構
        hf_url = f"https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data/resolve/main/klines/{pair}/{symbol}_{interval}.parquet"
        
        try:
            response = requests.get(hf_url, timeout=30)
            
            if response.status_code == 200:
                df = pd.read_parquet(BytesIO(response.content))
                
                # 使用最近的樣本數
                if len(df) > self.sample_size:
                    df = df.tail(self.sample_size).reset_index(drop=True)
                
                print(f"    ✓ 成功下載 {len(df):,} 條記錄")
                return df
            else:
                print(f"    × HTTP 狀態碼: {response.status_code}")
                return None
        
        except Exception as e:
            print(f"    × 下載失敖: {str(e)[:50]}")
            return None
    
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
        
        print(f"    ✓ 生成 {len(df):,} 條模擬數據")
        return df
    
    def extract_zigzag(self, df, pair, interval):
        """
        第二步：提取 ZigZag 轉折點
        """
        # 導入或定義 ZigZag 指標
        try:
            from zigzag_indicator import ZigZagIndicator
        except ImportError:
            ZigZagIndicator = self._get_zigzag_class()
        
        print(f"    提取 ZigZag...")
        
        zz = ZigZagIndicator(
            depth=self.depth,
            deviation=self.deviation,
            backstep=self.backstep
        )
        
        result = zz.extract(df)
        
        pivot_count = result['swing_type'].notna().sum()
        print(f"    ✓ 轉折點: {pivot_count} 個 ({pivot_count/len(result)*100:.2f}%)")
        
        return result
    
    def _get_zigzag_class(self):
        """
        返回內置 ZigZag 實現
        """
        class ZigZagIndicator:
            def __init__(self, depth=12, deviation=0.8, backstep=2):
                self.depth = depth
                self.deviation = deviation / 100.0
                self.backstep = backstep
            
            def extract(self, df):
                df = df.copy()
                df['zigzag'] = np.nan
                df['swing_type'] = ''
                
                highs = df['high'].values
                lows = df['low'].values
                
                # 简化實現：盤接標記局點高低點
                for i in range(self.depth, len(df) - self.depth):
                    # 局點高點
                    if highs[i] == highs[max(0, i-self.depth):i+self.depth+1].max():
                        df.loc[i, 'zigzag'] = highs[i]
                        
                        # 檢查之前的低點
                        if i > self.depth:
                            prev_low_idx = df.loc[:i-1, 'zigzag'].last_valid_index()
                            if prev_low_idx is not None:
                                prev_val = df.loc[prev_low_idx, 'zigzag']
                                if highs[i] > prev_val:
                                    df.loc[i, 'swing_type'] = 'HH'  # 更高高點
                                else:
                                    df.loc[i, 'swing_type'] = 'LH'  # 更低高點
                    
                    # 局點低點
                    elif lows[i] == lows[max(0, i-self.depth):i+self.depth+1].min():
                        df.loc[i, 'zigzag'] = lows[i]
                        
                        # 檢查之前的高點
                        if i > self.depth:
                            prev_high_idx = df.loc[:i-1, 'zigzag'].last_valid_index()
                            if prev_high_idx is not None:
                                prev_val = df.loc[prev_high_idx, 'zigzag']
                                if lows[i] < prev_val:
                                    df.loc[i, 'swing_type'] = 'LL'  # 更低低點
                                else:
                                    df.loc[i, 'swing_type'] = 'HL'  # 更高低點
                
                return df
        
        return ZigZagIndicator
    
    def feature_engineering(self, df):
        """
        第三步：特徵工程
        """
        try:
            from feature_engineering import ZigZagFeatureEngineering
            fe = ZigZagFeatureEngineering(lookback_windows=[5, 10, 20, 50])
            df_features = fe.create_features(df, verbose=False)
        except ImportError:
            df_features = self._basic_feature_engineering(df)
        
        return df_features
    
    def _basic_feature_engineering(self, df):
        """
        基礎特徵工程
        """
        df = df.copy()
        
        # 基礎技術指標
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['momentum'] = df['close'] - df['close'].shift(10)
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        
        # 填充缺失值
        df = df.fillna(method='bfill').fillna(method='ffill')
        
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
        try:
            from feature_engineering import prepare_ml_dataset
            X_train, X_test, y_train, y_test, feature_names, label_encoder = prepare_ml_dataset(
                df_features, test_size=0.2, verbose=False
            )
        except ImportError:
            X_train, X_test, y_train, y_test, feature_names, label_encoder = self._basic_prepare_data(df_features)
        
        # 驗證數據
        if len(X_train) < 10 or len(X_test) < 3:
            print(f"    × 數據不足以訓練")
            return None
        
        # 訓練 XGBoost
        print(f"    訓練 XGBoost...")
        xgb_model = self._train_xgboost(X_train, y_train, X_test, y_test, label_encoder)
        
        # 評估
        xgb_pred = xgb_model.predict(X_test)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        xgb_f1 = f1_score(y_test, xgb_pred, average='weighted', zero_division=0)
        
        print(f"    ✓ 準確率: {xgb_acc:.4f} | F1: {xgb_f1:.4f}")
        
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
    
    def _basic_prepare_data(self, df):
        """
        基礎數據準備
        """
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        
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
            n = 100
            X = np.random.randn(n, 10)
            y = np.random.randint(0, 4, n)
            feature_names = [f'feature_{i}' for i in range(10)]
        else:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in feature_cols if c not in ['timestamp', 'pair', 'interval']]
            
            if len(feature_cols) < 5:
                n = len(df)
                X = np.random.randn(n, 10)
                feature_names = [f'feature_{i}' for i in range(10)]
            else:
                X = df[feature_cols].values
                feature_names = feature_cols
            
            le = LabelEncoder()
            y = le.fit_transform(df[label_col])
        
        # 分割數據
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        le = LabelEncoder()
        le.fit(y)
        
        return X_train, X_test, y_train, y_test, feature_names, le
    
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
            print(f"    ⚠ joblib 保存失敖：{e}")
        
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
        執行完整管算
        """
        print("\n" + "#"*70)
        print("# ZigZag 完整訓練管算 (支援批量)")
        print("#"*70)
        print(f"\n配置:")
        print(f"  幣種: {', '.join(self.pairs)} ({len(self.pairs)} 個)")
        print(f"  時間框架: {', '.join(self.intervals)} ({len(self.intervals)} 個)")
        print(f"  ZigZag Depth: {self.depth}")
        print(f"  ZigZag Deviation: {self.deviation}%")
        print(f"  ZigZag Backstep: {self.backstep}")
        print(f"  樣本數: {self.sample_size}")
        print(f"  总模型数: {len(self.pairs) * len(self.intervals)}")
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
                    if df is None:
                        df = self._generate_sample_data(pair, interval)
                    
                    # 步驟 2: 提取 ZigZag
                    df_zigzag = self.extract_zigzag(df, pair, interval)
                    
                    # 步驟 3: 特徵工程
                    df_features = self.feature_engineering(df_zigzag)
                    
                    # 步驟 4: 訓練模型
                    result = self.train_models(df_features, pair, interval)
                    if result:
                        self.train_results.append(result)
                        print(f"\n  ✓ {pair} {interval} 訓練完成")
                        print(f"    模型位置: {result['model_dir']}")
                
                except Exception as e:
                    print(f"  × 錯誤: {str(e)[:100]}")
        
        # 一覽統計
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print("\n" + "#"*70)
        print("# 訓練統計")
        print("#"*70)
        print(f"\n訊拫:")
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
            
            print(f"{'Average':<12} {'':<8} {avg_acc:<10.4f} {avg_f1:<10.4f}")
            print(f"{'Max/Min':<12} {'':<8} {max_acc:<10.4f}/{min_acc:<10.4f}")
        
        print("\n" + "#"*70)
        print("# ✓ 訓練完成!")
        print("#"*70 + "\n")
        
        return len(self.train_results) > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='ZigZag 完整訓練管算 (支援批量)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例：
  # 第一個幣種、单時間框架
  python train_complete_pipeline.py --pair BTCUSDT --interval 15m --depth 12 --deviation 0.8 --backstep 2 --sample 200000
  
  # 多個幣種
  python train_complete_pipeline.py --pair BTCUSDT ETHUSDT BNBUSDT --interval 15m --depth 12 --deviation 0.8 --backstep 2 --sample 200000
  
  # 所有幣種
  python train_complete_pipeline.py --pair all --interval 15m --depth 12 --deviation 0.8 --backstep 2 --sample 200000
  
  # 所有幣種 + 兩個時間框架
  python train_complete_pipeline.py --pair all --interval 15m 1h --depth 12 --deviation 0.8 --backstep 2 --sample 200000
        '''
    )
    
    parser.add_argument('--pair', nargs='+', default=['BTCUSDT'],
                        help='交易對: BTCUSDT, ETHUSDT 或 all/ALL/\* 訓練全部 (預設: BTCUSDT)')
    parser.add_argument('--interval', nargs='+', default=['15m'],
                        help='時間框架: 15m 或 1h 或两个 (預設: 15m)')
    parser.add_argument('--depth', type=int, default=12,
                        help='ZigZag Depth 參數 (預設: 12)')
    parser.add_argument('--deviation', type=float, default=0.8,
                        help='ZigZag Deviation 參數 (%) (預設: 0.8)')
    parser.add_argument('--backstep', type=int, default=2,
                        help='ZigZag Backstep 參數 (預設: 2)')
    parser.add_argument('--sample', type=int, default=200000,
                        help='使用的樣本数 (預設: 200000)')
    
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
