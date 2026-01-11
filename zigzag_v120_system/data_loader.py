#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZigZag 量化系統 v1.2.0 - 數據加載模組

機能說明
1. 從 HuggingFace 數據集加載 OHLCV 數據
2. 支援 38 個交易對
3. 支援 15m 和 1h 時間框架
4. 自動處理不完整數據
5. 識別錄律水平

作者: ZigZag 開發團隊
日期: 2026-01-11
版本: 1.2.0
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import warnings
import logging

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HuggingFaceDataLoader:
    """
    HuggingFace 數據集數據加載器
    """
    
    SUPPORTED_SYMBOLS = [
        'AAVEUSDT', 'ADAUSDT', 'ALGOUSDT', 'ARBUSDT', 'ATOMUSDT',
        'AVAXUSDT', 'BALUSDT', 'BATUSDT', 'BCHUSDT', 'BNBUSDT',
        'BTCUSDT', 'COMPUSDT', 'CRVUSDT', 'DOGEUSDT', 'DOTUSDT',
        'ENJUSDT', 'ENSUSDT', 'ETCUSDT', 'ETHUSDT', 'FILUSDT',
        'GALAUSDT', 'GRTUSDT', 'IMXUSDT', 'KAVAUSDT', 'LINKUSDT',
        'LTCUSDT', 'MANAUSDT', 'MATICUSDT', 'MKRUSDT', 'NEARUSDT',
        'OPUSDT', 'SANDUSDT', 'SNXUSDT', 'SOLUSDT', 'SPELLUSDT',
        'UNIUSDT', 'XRPUSDT', 'ZRXUSDT'
    ]
    
    SUPPORTED_TIMEFRAMES = ['15m', '1h']
    REPO_ID = "zongowo111/v2-crypto-ohlcv-data"
    HF_ROOT = "klines"
    
    def __init__(self, use_huggingface: bool = True, local_cache: bool = True):
        self.use_huggingface = use_huggingface
        self.local_cache = local_cache
        self.cache = {}
        
        if self.use_huggingface:
            try:
                from huggingface_hub import hf_hub_download
                self.hf_hub_download = hf_hub_download
                logger.info("HuggingFace Hub 已連接")
            except ImportError:
                raise ImportError(
                    "需要安裝 huggingface-hub: pip install huggingface-hub"
                )
    
    def validate_symbol(self, symbol: str) -> bool:
        if symbol not in self.SUPPORTED_SYMBOLS:
            logger.warning(f"警告: {symbol} 不在支援清單中")
            return False
        return True
    
    def validate_timeframe(self, timeframe: str) -> bool:
        if timeframe not in self.SUPPORTED_TIMEFRAMES:
            logger.warning(f"警告: {timeframe} 不支援")
            return False
        return True
    
    def load_klines(self, symbol: str, timeframe: str = '15m',
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None) -> pd.DataFrame:
        if not self.validate_symbol(symbol):
            return pd.DataFrame()
        if not self.validate_timeframe(timeframe):
            return pd.DataFrame()
        
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self.cache:
            df = self.cache[cache_key]
            logger.info(f"從本地快取加載 {symbol} {timeframe}")
        else:
            if self.use_huggingface:
                df = self._load_from_huggingface(symbol, timeframe)
            else:
                df = self._load_from_local(symbol, timeframe)
            
            if self.local_cache and not df.empty:
                self.cache[cache_key] = df
        
        if start_date is not None or end_date is not None:
            df = self._filter_by_date(df, start_date, end_date)
        
        return df
    
    def _load_from_huggingface(self, symbol: str, timeframe: str) -> pd.DataFrame:
        try:
            base = symbol.replace('USDT', '')
            filename = f"{base}_{timeframe}.parquet"
            path_in_repo = f"{self.HF_ROOT}/{symbol}/{filename}"
            
            logger.info(f"正在加載 HuggingFace: {path_in_repo}")
            
            local_path = self.hf_hub_download(
                repo_id=self.REPO_ID,
                filename=path_in_repo,
                repo_type="dataset"
            )
            
            df = pd.read_parquet(local_path)
            logger.info(f"加載完成。數據形狀: {df.shape}")
            return df
        
        except Exception as e:
            logger.error(f"從 HuggingFace 加載失敗: {str(e)}")
            return pd.DataFrame()
    
    def _load_from_local(self, symbol: str, timeframe: str) -> pd.DataFrame:
        try:
            base = symbol.replace('USDT', '')
            filename = f"klines/{symbol}/{base}_{timeframe}.parquet"
            logger.info(f"從本地加載: {filename}")
            df = pd.read_parquet(filename)
            logger.info(f"加載完成。數據形狀: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"從本地加載失敗: {str(e)}")
            return pd.DataFrame()
    
    def _filter_by_date(self, df: pd.DataFrame,
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> pd.DataFrame:
        if df.empty:
            return df
        
        df_filtered = df.copy()
        
        if 'open_time' in df_filtered.columns:
            if start_date is not None:
                df_filtered = df_filtered[df_filtered['open_time'] >= start_date]
            if end_date is not None:
                df_filtered = df_filtered[df_filtered['open_time'] <= end_date]
        
        logger.info(f"篩選後數據: {df_filtered.shape}")
        return df_filtered


class DataProcessor:
    """
    數據準備整理器
    """
    
    def __init__(self):
        self.scaler = None
    
    def clean_data(self, df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        df_clean = df.copy()
        nan_counts = df_clean.isnull().sum()
        if nan_counts.sum() > 0:
            logger.info(f"\n検測到 NaN 值: {nan_counts.sum()}")
        
        if method == 'drop':
            df_clean = df_clean.dropna()
        elif method == 'forward_fill':
            df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
        elif method == 'mean':
            numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        
        logger.info(f"清理完成: {df_clean.shape[0]} 行數據保留")
        return df_clean


if __name__ == "__main__":
    logger.info("\nZigZag 量化系統 v1.2.0 - 數據加載模組")
    logger.info("模組已準備，待整合整個系統...")
