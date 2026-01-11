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
        features['open'] = df['open']
        features['high'] = df['high']
        features['low'] = df['low']
        features['close'] = df['close']
        features['volume'] = df['volume']
        features['returns'] = df['close'].pct_change() * 100
        features['hl_ratio'] = (df['high'] - df['low']) / df['low'] * 100
        features['oc_ratio'] = (df['close'] - df['open']) / df['open'] * 100
        return features
    
    def build_feature_matrix(self, df: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
        """
        構建完整特徵矩阵
        """
        print("[FeatureBuilder] 開始構建特徵矩阵...")
        all_features = self.add_technical_features(df)
        print("[FeatureBuilder] 特徵矩陣構建完成!")
        return all_features


if __name__ == "__main__":
    print("ZigZag 量化系統 v1.2.0 - 特徵工程模組\n")
    print("模組已成功加載，準備與數據加載器集成...")
