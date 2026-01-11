#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZigZag 量化系統 v1.2.0 - 特徵工程模組
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')


class FeatureBuilder:
    """
    綜合特徵構建引擎
    """
    
    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加基础技術指標特徵
        """
        features = pd.DataFrame(index=df.index)
        
        # 基本 K 線 數據
        features['open'] = df['open'].astype(float)
        features['high'] = df['high'].astype(float)
        features['low'] = df['low'].astype(float)
        features['close'] = df['close'].astype(float)
        features['volume'] = df['volume'].astype(float)
        
        # 計算 returns（会有 NaN 在第一行）
        features['returns'] = df['close'].pct_change() * 100
        
        # 計算价格波动性（高低比）
        features['hl_ratio'] = ((df['high'] - df['low']) / (df['low'] + 1e-8)) * 100
        
        # 計算价格潋度（開盤比）
        features['oc_ratio'] = ((df['close'] - df['open']) / (df['open'] + 1e-8)) * 100
        
        # 計算我費算指數（简化版）
        features['vwap'] = (df['close'] * df['volume']).rolling(window=5).mean() / (df['volume'].rolling(window=5).mean() + 1e-8)
        
        # 鈨除不合法的值
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # 填汛 NaN 值（下作後上作）
        features = features.fillna(method='bfill', limit=5).fillna(method='ffill', limit=5)
        
        # 移除什么也帻不了的 NaN
        features = features.dropna()
        
        return features
    
    def build_feature_matrix(self, df: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
        """
        構建完整特徵矩陣
        """
        print("[FeatureBuilder] 開始構建特徵矩阵...")
        
        all_features = self.add_technical_features(df)
        
        print("[FeatureBuilder] 特徵構建完成!")
        print(f"[FeatureBuilder] 最終特徵矩阵大小: {all_features.shape}")
        
        return all_features


if __name__ == "__main__":
    print("ZigZag 量化系統 v1.2.0 - 特徵工程模組\n")
    print("模組已成功加載，準備與數據加載器集成...")
