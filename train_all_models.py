#!/usr/bin/env python3
"""
批量訓練所有 ZigZag 預測模型
支持 23 個幣種 × 2 個時間框架 = 46 個模型

用法：
  python train_all_models.py
  
  # 或指定子集
  python train_all_models.py --pairs BTCUSDT ETHUSDT --intervals 15m 1h
  
  # 或指定採樣大小
  python train_all_models.py --sample 100000
"""

import argparse
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib


# 所有可用幣種
ALL_PAIRS = [
    'AAVEUSDT', 'ADAUSDT', 'ALGOUSDT', 'ARBUSDT', 'ATOMUSDT',
    'AVAXUSDT', 'BCHUSDT', 'BNBUSDT', 'BTCUSDT', 'DOGEUSDT',
    'DOTUSDT', 'ETCUSDT', 'ETHUSDT', 'FILUSDT', 'LINKUSDT',
    'LTCUSDT', 'MATICUSDT', 'NEARUSDT', 'OPUSDT', 'SOLUSDT',
    'UNIUSDT', 'XRPUSDT'
]

ALL_INTERVALS = ['15m', '1h']


class CompletePipelineTrainer:
    """
    完整訓練管道 - 支援多個交易對和時間框架
    """
    
    def __init__(self):
        self.config = {}
        self.df_zigzag = None
        self.df_features = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.results = []
    
    def download_data(self, pair, interval, sample_size=200000):
        """
        從 HuggingFace 下載 OHLCV 數據
        """
        try:
            # 幣種代碼轉換（去掉 USDT 後綴）
            symbol = pair.replace('USDT', '')
            url = f"https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data/resolve/main/klines/{pair}/{symbol}_{interval}.parquet"
            
            df = pd.read_parquet(url)
            
            if len(df) == 0:
                print(f"    警告: 下載到空數據")
                return None
            
            # 取最新的 N 條
            if len(df) > sample_size:
                df = df.tail(sample_size).reset_index(drop=True)
            
            return df
        except Exception as e:
            print(f"    下載失敗: {str(e)[:50]}")
            return None
    
    def extract_zigzag_peaks(self, prices, depth=12, deviation=0.8):
        """
        提取 ZigZag 轉折點
        返回: 1(高點), -1(低點), 0(無轉折)
        """
        if len(prices) < depth * 2:
            return np.zeros(len(prices))
        
        peaks = np.zeros(len(prices))
        trend = 0
        last_peak_idx = 0
        last_peak_price = prices.iloc[0]
        
        for i in range(depth, len(prices) - depth):
            current_price = prices.iloc[i]
            
            if last_peak_price > 0:
                deviation_pct = abs(current_price - last_peak_price) / last_peak_price * 100
            else:
                deviation_pct = 0
            
            if trend == 0:
                if current_price > last_peak_price:
                    trend = 1
                else:
                    trend = -1
            
            elif trend == 1 and current_price < last_peak_price * (1 - deviation / 100):
                peaks[last_peak_idx] = 1
                trend = -1
                last_peak_idx = i
                last_peak_price = current_price
            
            elif trend == -1 and current_price > last_peak_price * (1 + deviation / 100):
                peaks[last_peak_idx] = -1
                trend = 1
                last_peak_idx = i
                last_peak_price = current_price
        
        return peaks
    
    def extract_features(self, df, depth=12, deviation=0.8):
        """
        提取技術指標特徵
        """
        df = df.copy()
        
        # 基礎價格轉換
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        # ZigZag 轉折點
        df['zigzag'] = self.extract_zigzag_peaks(
            df['close'], 
            depth=depth, 
            deviation=deviation
        )
        
        # 技術指標
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
    
    def prepare_data(self, df):
        """
        準備訓練數據
        """
        # 篩選有 ZigZag 轉折的樣本
        df_labeled = df[df['zigzag'] != 0].copy().reset_index(drop=True)
        
        if len(df_labeled) == 0:
            return None
        
        # 特徵準備
        self.feature_names = ['open', 'high', 'low', 'close', 'volume', 
                             'zigzag', 'returns', 'volatility', 'momentum', 'rsi', 'macd']
        X = df_labeled[self.feature_names].values
        y = df_labeled['zigzag'].values
        
        # 劃分
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """
        訓練 XGBoost 模型
        """
        # 標籤編碼
        self.label_encoder.fit(y_train)
        y_train_encoded = self.label_encoder.transform(y_train)
        
        # XGBoost 模型
        self.model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=len(self.label_encoder.classes_),
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        self.model.fit(X_train, y_train_encoded, verbose=False)
    
    def evaluate_model(self, X_test, y_test):
        """
        計算模型效能
        """
        y_test_encoded = self.label_encoder.transform(y_test)
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test_encoded, y_pred)
        f1 = f1_score(y_test_encoded, y_pred, average='weighted')
        
        return accuracy, f1
    
    def save_model(self, pair, interval, params):
        """
        保存模型
        """
        # 建造模型目錄
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir = f"models/{pair}_{interval}_{timestamp}"
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        joblib.dump(self.model, f"{model_dir}/xgboost_model.joblib")
        
        # 保存標籤編碼器
        with open(f"{model_dir}/label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)
        
        # 保存特徵名稱
        with open(f"{model_dir}/feature_names.json", "w") as f:
            json.dump(self.feature_names, f, indent=2)
        
        # 保存參數
        with open(f"{model_dir}/params.json", "w") as f:
            json.dump(params, f, indent=2)
        
        return model_dir
    
    def train(self, pair, interval, depth=12, deviation=0.8, sample_size=200000):
        """
        完整訓練流程
        """
        # 下載數據
        df = self.download_data(pair, interval, sample_size)
        if df is None or len(df) == 0:
            return None
        
        # 提取特徵
        df = self.extract_features(df, depth=depth, deviation=deviation)
        
        # 準備訓練數據
        data = self.prepare_data(df)
        if data is None:
            return None
        
        X_train, X_test, y_train, y_test = data
        
        if len(X_train) < 10 or len(X_test) < 3:
            return None
        
        # 訓練模型
        self.train_model(X_train, y_train)
        
        # 評估模型
        accuracy, f1 = self.evaluate_model(X_test, y_test)
        
        # 保存模型
        params = {
            'pair': pair,
            'interval': interval,
            'depth': depth,
            'deviation': deviation,
            'accuracy': accuracy,
            'f1_score': f1,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
        model_dir = self.save_model(pair, interval, params)
        
        return {
            'pair': pair,
            'interval': interval,
            'model_dir': model_dir,
            'accuracy': accuracy,
            'f1_score': f1,
            'train_samples': len(X_train)
        }


def main():
    parser = argparse.ArgumentParser(description='ZigZag 批量模型訓練')
    parser.add_argument('--pairs', nargs='+', default=ALL_PAIRS, help='交易對序列')
    parser.add_argument('--intervals', nargs='+', default=ALL_INTERVALS, help='時間框架序列')
    parser.add_argument('--depth', type=int, default=12, help='ZigZag Depth')
    parser.add_argument('--deviation', type=float, default=0.8, help='ZigZag Deviation (%)')
    parser.add_argument('--sample', type=int, default=200000, help='樣本數')
    parser.add_argument('--skip-failed', action='store_true', help='跳過失敗的訓練')
    
    args = parser.parse_args()
    
    # 訓練器
    trainer = CompletePipelineTrainer()
    
    print("\n" + "="*70)
    print("ZigZag 批量模型訓練")
    print("="*70)
    print(f"幣種數: {len(args.pairs)}")
    print(f"時間框架數: {len(args.intervals)}")
    print(f"總模型數: {len(args.pairs) * len(args.intervals)}")
    print(f"\n參數:")
    print(f"  Depth: {args.depth}")
    print(f"  Deviation: {args.deviation}%")
    print(f"  Sample Size: {args.sample:,}")
    print("="*70 + "\n")
    
    # 開始訓練
    total_models = len(args.pairs) * len(args.intervals)
    trained_count = 0
    failed_count = 0
    
    start_time = datetime.now()
    
    for pair in args.pairs:
        for interval in args.intervals:
            trained_count += 1
            status = f"[{trained_count:2d}/{total_models}] {pair:<12} {interval:<5}"
            
            try:
                result = trainer.train(
                    pair=pair,
                    interval=interval,
                    depth=args.depth,
                    deviation=args.deviation,
                    sample_size=args.sample
                )
                
                if result:
                    acc = result['accuracy']
                    f1 = result['f1_score']
                    samples = result['train_samples']
                    print(f"{status} Acc: {acc:.4f} F1: {f1:.4f} Samples: {samples:,}")
                    trainer.results.append(result)
                else:
                    print(f"{status} 失敗: 數據不足")
                    failed_count += 1
            except Exception as e:
                print(f"{status} 錯誤: {str(e)[:40]}")
                failed_count += 1
                if not args.skip_failed:
                    pass
    
    # 統計
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "="*70)
    print("訓練統計")
    print("="*70)
    print(f"成功: {len(trainer.results)}/{total_models}")
    print(f"失敗: {failed_count}")
    print(f"耗時: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分鐘)")
    
    if trainer.results:
        print(f"\n模型結果:")
        print(f"{'Pair':<12} {'Interval':<8} {'Accuracy':<10} {'F1 Score':<10} {'Samples':<10}")
        print("-" * 50)
        
        accuracies = []
        f1_scores = []
        
        for result in trainer.results:
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
    
    print("\n" + "="*70)
    print("訓練完成!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
