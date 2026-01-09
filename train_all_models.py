#!/usr/bin/env python3
"""
批量訓練多個交易對的 ZigZag 預測模型

用法：
  python train_all_models.py
  
  # 或下輇批次訓練（可選）
  python train_all_models.py --pairs BTCUSDT ETHUSDT --intervals 15m 1h --sample 200000
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
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib


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
        print(f"\n下載 {pair} {interval}...")
        try:
            url = f"https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data/resolve/main/klines/{pair}/{pair.replace('USDT', '')}_{interval}.parquet"
            df = pd.read_parquet(url)
            
            print(f"  ✓ 成功下載 {len(df):,} 条記錄")
            
            # 取最新的 N 条
            if len(df) > sample_size:
                df = df.tail(sample_size).reset_index(drop=True)
                print(f"  ✓ 使用最近 {sample_size:,} 条記錄")
            
            return df
        except Exception as e:
            print(f"  ✗ 下載失敖: {str(e)}")
            return None
    
    def extract_zigzag_peaks(self, prices, depth=12, deviation=0.8, backstep=2):
        """
        提取 ZigZag 轉折點 - 詳另章 ZigZag 寶寶算法
        返回：
          - 1: 高點
          - -1: 低點
          - 0: 沒有轉折
        """
        if len(prices) < depth * 2:
            return np.zeros(len(prices))
        
        peaks = np.zeros(len(prices))
        trend = 0  # 1: 上升, -1: 下降
        last_peak_idx = 0
        last_peak_price = prices.iloc[0]
        
        for i in range(depth, len(prices) - depth):
            window_high = prices.iloc[i:i+depth].max()
            window_low = prices.iloc[i:i+depth].min()
            current_price = prices.iloc[i]
            
            # 計算偏差百分比
            if last_peak_price > 0:
                deviation_pct = abs(current_price - last_peak_price) / last_peak_price * 100
            else:
                deviation_pct = 0
            
            # 判斷趨勢
            if trend == 0:  # 初始化
                if current_price > last_peak_price:
                    trend = 1  # 開始第一肢上升
                else:
                    trend = -1  # 開始第一肢下降
            
            # 上升轉下降
            elif trend == 1 and current_price < last_peak_price * (1 - deviation / 100):
                peaks[last_peak_idx] = 1  # 標記高點
                trend = -1
                last_peak_idx = i
                last_peak_price = current_price
            
            # 下降轉上升
            elif trend == -1 and current_price > last_peak_price * (1 + deviation / 100):
                peaks[last_peak_idx] = -1  # 標記低點
                trend = 1
                last_peak_idx = i
                last_peak_price = current_price
        
        return peaks
    
    def classify_zigzag_turns(self, df):
        """
        分類 ZigZag 轉折
        策略：
        - 上上 (HH): 高點 > 前一個高點
        - 上下 (HL): 高點 < 前一個高點
        - 下上 (LH): 低點 > 前一個低點
        - 下下 (LL): 低點 < 前一個低點
        """
        df = df.copy()
        df['zigzag_type'] = 0  # 0: 沒有轉折
        
        high_prices = []
        low_prices = []
        
        for i in range(len(df)):
            if df['zigzag_peak'].iloc[i] == 1:  # 高點
                if high_prices:
                    if df['high'].iloc[i] > high_prices[-1]:
                        df.loc[i, 'zigzag_type'] = 1  # HH
                    else:
                        df.loc[i, 'zigzag_type'] = 2  # HL
                high_prices.append(df['high'].iloc[i])
            
            elif df['zigzag_peak'].iloc[i] == -1:  # 低點
                if low_prices:
                    if df['low'].iloc[i] > low_prices[-1]:
                        df.loc[i, 'zigzag_type'] = 3  # LH
                    else:
                        df.loc[i, 'zigzag_type'] = 4  # LL
                low_prices.append(df['low'].iloc[i])
        
        return df
    
    def extract_features(self, df, depth=12, deviation=0.8, backstep=2):
        """
        提取技術指標特徵
        """
        df = df.copy()
        
        # 1. ZigZag 轉折點
        df['zigzag_peak'] = self.extract_zigzag_peaks(
            df['close'], 
            depth=depth, 
            deviation=deviation, 
            backstep=backstep
        )
        
        # 2. 简单特徵
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        # 3. 技術指標
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['momentum'] = df['close'] - df['close'].shift(10)
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['zigzag'] = df['zigzag_peak']  # 填充 zigzag 特徵
        
        # 填充缺失值
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        return df
    
    def _calculate_rsi(self, prices, period=14):
        """
        RSI 計算
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
        準備訓練資料
        仅保留有 ZigZag 轉折的樣本
        """
        # 篤選有 ZigZag 轉折的樣本
        df_labeled = df[df['zigzag_peak'] != 0].copy().reset_index(drop=True)
        
        if len(df_labeled) == 0:
            print("  ✗ 找不到 ZigZag 轉折")
            return None, None, None
        
        print(f"  ✓ 找到 {len(df_labeled)} 個 ZigZag 轉折樣本")
        
        # 特徵準備
        self.feature_names = ['open', 'high', 'low', 'close', 'volume', 'zigzag', 'returns', 'volatility', 'momentum', 'rsi', 'macd']
        X = df_labeled[self.feature_names].values
        y = df_labeled['zigzag_peak'].values
        
        # 棘分
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """
        訓練 XGBoost 模型
        """
        print(f"  訓練資料數: {len(X_train)}")
        
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
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train_encoded)
        print(f"  ✓ 模型訓練完成")
    
    def evaluate_model(self, X_test, y_test):
        """
        計算模型效能
        """
        y_test_encoded = self.label_encoder.transform(y_test)
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test_encoded, y_pred)
        f1 = f1_score(y_test_encoded, y_pred, average='weighted')
        
        print(f"\n  模型效能:")
        print(f"    準確率: {accuracy:.4f}")
        print(f"    F1 Score: {f1:.4f}")
        
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
        
        print(f"  ✓ 模型已保存: {model_dir}")
        return model_dir
    
    def train(self, pair, interval, depth=12, deviation=0.8, backstep=2, sample_size=200000):
        """
        完整訓練流程
        """
        print(f"\n{'='*60}")
        print(f"訓練: {pair} {interval}")
        print(f"{'='*60}")
        
        # 下載數據
        df = self.download_data(pair, interval, sample_size)
        if df is None or len(df) == 0:
            return False
        
        # 提取特徵
        print(f"\n提取特徵...")
        df = self.extract_features(df, depth=depth, deviation=deviation, backstep=backstep)
        
        # 準備訓練資料
        print(f"\n準備訓練資料...")
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        if X_train is None:
            return False
        
        # 訓練模型
        print(f"\n訓練模型...")
        self.train_model(X_train, y_train)
        
        # 計算效能
        print(f"\n計算效能...")
        accuracy, f1 = self.evaluate_model(X_test, y_test)
        
        # 保存模型
        print(f"\n保存模型...")
        params = {
            'pair': pair,
            'interval': interval,
            'depth': depth,
            'deviation': deviation,
            'backstep': backstep,
            'accuracy': accuracy,
            'f1_score': f1
        }
        model_dir = self.save_model(pair, interval, params)
        
        self.results.append({
            'pair': pair,
            'interval': interval,
            'model_dir': model_dir,
            'accuracy': accuracy,
            'f1_score': f1
        })
        
        return True


def main():
    parser = argparse.ArgumentParser(description='ZigZag 多模型訓練')
    parser.add_argument('--pairs', nargs='+', default=['BTCUSDT', 'ETHUSDT'], help='交易對序列')
    parser.add_argument('--intervals', nargs='+', default=['15m', '1h'], help='時間框架序列')
    parser.add_argument('--depth', type=int, default=12, help='ZigZag Depth')
    parser.add_argument('--deviation', type=float, default=0.8, help='ZigZag Deviation (%)')
    parser.add_argument('--backstep', type=int, default=2, help='ZigZag Backstep')
    parser.add_argument('--sample', type=int, default=200000, help='样本數')
    
    args = parser.parse_args()
    
    # 訓練器
    trainer = CompletePipelineTrainer()
    
    print("\n" + "="*60)
    print("ZigZag 多模型批量訓練酋始")
    print("="*60)
    print(f"交易對: {args.pairs}")
    print(f"時間框架: {args.intervals}")
    print(f"\u914b始參數:")
    print(f"  Depth: {args.depth}")
    print(f"  Deviation: {args.deviation}%")
    print(f"  Backstep: {args.backstep}")
    print(f"  Sample Size: {args.sample:,}")
    print("="*60)
    
    # 訓練每一個交易對/時間框架組合
    total_models = len(args.pairs) * len(args.intervals)
    trained_count = 0
    
    for pair in args.pairs:
        for interval in args.intervals:
            trained_count += 1
            print(f"\n[{trained_count}/{total_models}] ", end="")
            
            success = trainer.train(
                pair=pair,
                interval=interval,
                depth=args.depth,
                deviation=args.deviation,
                backstep=args.backstep,
                sample_size=args.sample
            )
            
            if not success:
                print(f"警告: 訓練 {pair} {interval} 失敖")
    
    # 求和統計
    print(f"\n\n" + "="*60)
    print(f"訓練統計")
    print("="*60)
    print(f"成功訓練數: {len(trainer.results)}/{total_models}")
    
    if trainer.results:
        print(f"\n模型統計:")
        print(f"{'Pair':<12} {'Interval':<10} {'Accuracy':<10} {'F1 Score':<10}")
        print("-" * 42)
        
        for result in trainer.results:
            print(f"{result['pair']:<12} {result['interval']:<10} {result['accuracy']:<10.4f} {result['f1_score']:<10.4f}")
        
        avg_accuracy = np.mean([r['accuracy'] for r in trainer.results])
        avg_f1 = np.mean([r['f1_score'] for r in trainer.results])
        print("-" * 42)
        print(f"{'Average':<12} {'':<10} {avg_accuracy:<10.4f} {avg_f1:<10.4f}")
    
    print("\n" + "="*60)
    print("訓練完成!")
    print("="*60)


if __name__ == '__main__':
    main()
