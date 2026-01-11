#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZigZag 量化系統 v1.2.0 - 特徵工程模組

實現功能：
1. 分數階差分特徵 (Fractional Differentiation)
2. 衷生品特徵 (Derivatives Features)
3. 微結構特徵 (Market Microstructure)
4. 綜合特徵集成與質量骗警

作者: ZigZag 開發團隊
日期: 2026-01-11
版本: 1.2.0
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings('ignore')


class FractionalDifferentiationEngine:
    """
    分數階差分特徵生成引擎
    
    用於解決標準差分導致的信息丟失和非平穩性問題
    數学原理: Δ^d_t = Σ(k=1 to t) C(d,k) * (-1)^k * X_{t-k}
    其中 C(d,k) 是廣義二項式係數
    """
    
    @staticmethod
    def get_weights(d: float, size: int) -> np.ndarray:
        """
        計算廣義二項式係數作為權重
        
        參數:
            d: 差分階數 (0 < d < 1)
            size: 時間序列長度
        
        返回:
            weights: 權重數組 [w_1, w_2, ..., w_size]
        """
        weights = np.ones(size)
        k = 1
        while k < size:
            weight = -weights[k - 1] * (d - k + 1) / k
            weights[k] = weight
            k += 1
        
        return weights
    
    @staticmethod
    def adf_test(series: pd.Series) -> Dict[str, float]:
        """
        Augmented Dickey-Fuller 平穩性検骗
        
        參數:
            series: 時間序列 (Pandas Series)
        
        返回:
            検骗結果
        """
        result = adfuller(series.dropna(), autolag='AIC')
        return {
            'statistic': result[0],
            'p_value': result[1],
            'is_stationary': result[1] < 0.05
        }
    
    def fractional_differentiation(self, series: pd.Series, d: float = 0.4) -> pd.Series:
        """
        實現分數階差分
        """
        weights = self.get_weights(d, len(series))
        diffed = pd.Series(index=series.index, dtype=float)
        
        for idx in range(len(weights), len(series)):
            diffed.iloc[idx] = np.dot(weights, series.iloc[idx-len(weights)+1:idx+1].values)
        
        return diffed


class FeatureBuilder:
    """
    綜合特徵構建引擎
    """
    
    def __init__(self):
        self.frac_diff_engine = FractionalDifferentiationEngine()
    
    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加基础技術指標特徵
        """
        features = pd.DataFrame(index=df.index)
        
        features['open'] = df['open']
        features['high'] = df['high']
        features['low'] = df['low']
        features['close'] = df['close']
        features['volume'] = df['volume']
        
        features['returns'] = df['close'].pct_change() * 100
        features['hl_ratio'] = (df['high'] - df['low']) / df['low'] * 100
        features['oc_ratio'] = (df['close'] - df['open']) / df['open'] * 100
        
        features['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                np.abs(df['high'] - df['close'].shift()),
                np.abs(df['low'] - df['close'].shift())
            )
        )
        features['atr_14'] = features['tr'].rolling(14).mean()
        
        features['volume_ma_20'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / (features['volume_ma_20'] + 1e-8)
        
        features['momentum_10'] = df['close'].diff(10)
        features['momentum_20'] = df['close'].diff(20)
        
        return features
    
    def add_fibonacci_features(self, df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
        """
        添加斐珤那契回佈特徵
        """
        features = pd.DataFrame(index=df.index)
        
        high_max = df['high'].rolling(window).max()
        low_min = df['low'].rolling(window).min()
        amplitude = high_max - low_min
        
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.705, 0.786]
        
        for level in fib_levels:
            col_name = f'fib_{level}'
            features[col_name] = low_min + amplitude * level
        
        for level in fib_levels:
            fib_col = f'fib_{level}'
            features[f'dist_to_fib_{level}'] = (
                (df['close'] - features[fib_col]) / features[fib_col] * 100
            ).fillna(0)
        
        return features
    
    def add_bollinger_bands_features(self, df: pd.DataFrame, period: int = 20, std_mult: float = 2.0) -> pd.DataFrame:
        """
        添加布林帶特徵
        """
        features = pd.DataFrame(index=df.index)
        
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        
        features['bb_middle'] = sma
        features['bb_upper'] = sma + (std * std_mult)
        features['bb_lower'] = sma - (std * std_mult)
        features['bb_width'] = features['bb_upper'] - features['bb_lower']
        
        features['bb_position'] = (
            (df['close'] - features['bb_lower']) / 
            (features['bb_upper'] - features['bb_lower'])
        ).fillna(0.5)
        
        features['bb_width_pct'] = (
            features['bb_width'] / features['bb_middle'] * 100
        ).fillna(0)
        
        return features
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加時間特徵
        """
        features = pd.DataFrame(index=df.index)
        
        features['hour'] = df.index.hour
        features['dayofweek'] = df.index.dayofweek
        features['dayofmonth'] = df.index.day
        features['month'] = df.index.month
        
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['dayofweek_sin'] = np.sin(2 * np.pi * features['dayofweek'] / 7)
        features['dayofweek_cos'] = np.cos(2 * np.pi * features['dayofweek'] / 7)
        
        return features
    
    def normalize_features(self, df: pd.DataFrame, fit_on_index: Optional[slice] = None) -> pd.DataFrame:
        """
        使用 Z-score 標準化特徵
        """
        normalized = df.copy()
        
        if fit_on_index is None:
            fit_data = df
        else:
            fit_data = df.iloc[fit_on_index]
        
        for col in df.columns:
            if df[col].dtype in ['float64', 'float32']:
                mean = fit_data[col].mean()
                std = fit_data[col].std()
                
                if std > 0:
                    normalized[col] = (df[col] - mean) / std
                else:
                    normalized[col] = 0
        
        return normalized
    
    def build_feature_matrix(self, df: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
        """
        構建完整的特徵矩陣
        """
        print("[FeatureBuilder] 開始構建特徵矩陣...")
        
        all_features = pd.DataFrame(index=df.index)
        
        print("[1/5] 添加技術指標特徵...")
        tech_features = self.add_technical_features(df)
        all_features = pd.concat([all_features, tech_features], axis=1)
        
        print("[2/5] 計算斐珤那契回佈特徵...")
        fib_features = self.add_fibonacci_features(df, window=100)
        all_features = pd.concat([all_features, fib_features], axis=1)
        
        print("[3/5] 計算布林帶特徵...")
        bb_features = self.add_bollinger_bands_features(df, period=20, std_mult=2.0)
        all_features = pd.concat([all_features, bb_features], axis=1)
        
        print("[4/5] 添加時間特徵...")
        time_features = self.add_time_features(df)
        all_features = pd.concat([all_features, time_features], axis=1)
        
        if normalize:
            print("[5/5] 進行特徵標準化...")
            all_features = self.normalize_features(all_features)
        else:
            print("[5/5] 跳過標準化步驟")
        
        print("[FeatureBuilder] 特徵矩陣構建完成!")
        print(f"総特徵數: {len(all_features.columns)}")
        
        return all_features


if __name__ == "__main__":
    print("ZigZag 量化系統 v1.2.0 - 特徵工程模組\n")
    print("模組已成功加載，準備與數據加載器集成...")
