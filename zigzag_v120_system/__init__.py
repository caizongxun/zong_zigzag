#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZigZag 量化交易系統 v1.2.0

一整套模組化的量化訓練系統，主要包括:

1. data_loader - 數據加載
   - HuggingFaceDataLoader: 從 HuggingFace 加載 OHLCV 數據
   - DataProcessor: 數據準備與準览

2. feature_engineering - 特徵工程
   - FeatureBuilder: 技術指標特徵構建
   - FractionalDifferentiationEngine: 分數階差分

3. model_architecture - 模組架構
   - LSTMPredictor: LSTM 運動語家模組
   - XGBoostEnsemble: XGBoost 集合模組
   - DualLayerDefenseSystem: 雙層防御系統

4. strategy_executor - 策略執行
   - StrategyExecutor: 交易信號執行與風險管理

使用方法:

    from zigzag_v120_system.data_loader import HuggingFaceDataLoader, DataProcessor
    from zigzag_v120_system.feature_engineering import FeatureBuilder
    from zigzag_v120_system.model_architecture import LSTMPredictor, DualLayerDefenseSystem
    from zigzag_v120_system.strategy_executor import StrategyExecutor

作者: ZigZag 開發團隊
版本: 1.2.0
日期: 2026-01-11
"""

__version__ = '1.2.0'
__author__ = 'ZigZag Development Team'
__all__ = [
    'data_loader',
    'feature_engineering',
    'model_architecture',
    'strategy_executor'
]
