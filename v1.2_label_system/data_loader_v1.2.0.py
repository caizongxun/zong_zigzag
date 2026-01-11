#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZigZag 量化系統 v1.2.0 - 數據加載模塊

機能說明
1. 從 HuggingFace 數據集加載 OHLCV 數據
2. 支援 38 個交易對 (上ETH, BTC, SOL 等)
3. 支援 15m 和 1h 時間框架
4. 自動处理不完整數據
5. 辨識錯況水平

作者: ZigZag 開發團隊
日期: 2026-01-11
版本: 1.2.0
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class HuggingFaceDataLoader:
    """
    HuggingFace 數據集數據加載器
    
    支援的交易對 (38 個):
    AAVEUSDT, ADAUSDT, ALGOUSDT, ARBUSDT, ATOMUSDT, AVAXUSDT,
    BALUSDT, BATUSDT, BCHUSDT, BNBUSDT, BTCUSDT, COMPUSDT,
    CRVUSDT, DOGEUSDT, DOTUSDT, ENJUSDT, ENSUSDT, ETCUSDT,
    ETHUSDT, FILUSDT, GALAUSDT, GRTUSDT, IMXUSDT, KAVAUSDT,
    LINKUSDT, LTCUSDT, MANAUSDT, MATICUSDT, MKRUSDT, NEARUSDT,
    OPUSDT, SANDUSDT, SNXUSDT, SOLUSDT, SPELLUSDT, UNIUSDT,
    XRPUSDT, ZRXUSDT
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
        """
        数据加載器初始化
        
        參数:
            use_huggingface: 是否使用 HuggingFace 上源
            local_cache: 是否使用本地缓存
        """
        self.use_huggingface = use_huggingface
        self.local_cache = local_cache
        self.cache = {}
        
        if self.use_huggingface:
            try:
                from huggingface_hub import hf_hub_download
                self.hf_hub_download = hf_hub_download
                print("HuggingFace 克止後是了。")
            except ImportError:
                raise ImportError(
                    "需要安裝 huggingface-hub: pip install huggingface-hub"
                )
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        檢骗交易對是否有效
        """
        if symbol not in self.SUPPORTED_SYMBOLS:
            print(f"譩試: {symbol} 不在支援清單中")
            print(f"支援的交易對: {', '.join(self.SUPPORTED_SYMBOLS)}")
            return False
        return True
    
    def validate_timeframe(self, timeframe: str) -> bool:
        """
        檢骗時間框架是否有效
        """
        if timeframe not in self.SUPPORTED_TIMEFRAMES:
            print(f"譩誖: {timeframe} 不正常")
            print(f"支援的時間框架: {', '.join(self.SUPPORTED_TIMEFRAMES)}")
            return False
        return True
    
    def load_klines(
        self,
        symbol: str,
        timeframe: str = '15m',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        加載 K 線數據
        
        參数:
            symbol: 交易對 (例: 'BTCUSDT')
            timeframe: 時間框架 ('15m' 或 '1h')
            start_date: 開始日期 (可選)
            end_date: 結束日期 (可選)
        
        返回:
            pandas DataFrame
        """
        # 驗證輸入
        if not self.validate_symbol(symbol):
            return pd.DataFrame()
        
        if not self.validate_timeframe(timeframe):
            return pd.DataFrame()
        
        # 檢查本地缓存
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self.cache:
            df = self.cache[cache_key]
            print(f"從本地缓存加載 {symbol} {timeframe}")
        else:
            if self.use_huggingface:
                df = self._load_from_huggingface(symbol, timeframe)
            else:
                df = self._load_from_local(symbol, timeframe)
            
            if self.local_cache:
                self.cache[cache_key] = df
        
        # 日期範圍篩選
        if start_date is not None or end_date is not None:
            df = self._filter_by_date(df, start_date, end_date)
        
        return df
    
    def _load_from_huggingface(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        從 HuggingFace 數據集加載
        """
        try:
            base = symbol.replace('USDT', '')
            filename = f"{base}_{timeframe}.parquet"
            path_in_repo = f"{self.HF_ROOT}/{symbol}/{filename}"
            
            print(f"從 HuggingFace 加載: {path_in_repo}...")
            
            local_path = self.hf_hub_download(
                repo_id=self.REPO_ID,
                filename=path_in_repo,
                repo_type="dataset"
            )
            
            df = pd.read_parquet(local_path)
            print(f"加載完成。數據形狀: {df.shape}")
            
            return df
        
        except Exception as e:
            print(f"加載失敗: {str(e)}")
            return pd.DataFrame()
    
    def _load_from_local(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        從本地檔案加載
        """
        try:
            base = symbol.replace('USDT', '')
            filename = f"klines/{symbol}/{base}_{timeframe}.parquet"
            
            print(f"從本地加載: {filename}...")
            
            df = pd.read_parquet(filename)
            print(f"加載完成。數據形狀: {df.shape}")
            
            return df
        
        except Exception as e:
            print(f"加載失敗: {str(e)}")
            return pd.DataFrame()
    
    def _filter_by_date(
        self,
        df: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        按日整篩選數據
        """
        df_filtered = df.copy()
        
        if 'open_time' in df_filtered.columns:
            if start_date is not None:
                df_filtered = df_filtered[df_filtered['open_time'] >= start_date]
            
            if end_date is not None:
                df_filtered = df_filtered[df_filtered['open_time'] <= end_date]
        
        return df_filtered
    
    def get_data_info(self, df: pd.DataFrame) -> Dict:
        """
        取得數據信息
        """
        return {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'date_range': {
                'start': df['open_time'].min() if 'open_time' in df.columns else None,
                'end': df['open_time'].max() if 'open_time' in df.columns else None
            },
            'price_range': {
                'open_min': df['open'].min(),
                'open_max': df['open'].max(),
                'close_min': df['close'].min(),
                'close_max': df['close'].max(),
                'high_max': df['high'].max(),
                'low_min': df['low'].min()
            }
        }


class DataProcessor:
    """
    數據准備整理器
    
    准備任務:
    1. 処理不完整數據 (NaN, 異適值)
    2. 正觀化數據
    3. 輸出技術指標
    4. 割分訓練/驗證/測試集
    """
    
    def __init__(self):
        self.scaler = None
    
    def clean_data(self, df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """
        清理不完整數據
        
        參數:
            df: 輸入 DataFrame
            method: 处理方法
              'drop': 削除 NaN 行
              'forward_fill': 前償填充
              'mean': 用平均值填充
        
        返回:
            清理例的 DataFrame
        """
        df_clean = df.copy()
        
        # 記錄統計
        print(f"\n數據清理後的NaN數據量:")
        nan_counts = df_clean.isnull().sum()
        print(nan_counts[nan_counts > 0])
        
        if method == 'drop':
            df_clean = df_clean.dropna()
        
        elif method == 'forward_fill':
            df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
        
        elif method == 'mean':
            numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        
        print(f"\n清理後: {df_clean.shape[0]} 行數據 保留")
        
        return df_clean
    
    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        移除異適值 (Z-Score 方法)
        
        參数:
            df: 輸入 DataFrame
            columns: 要梨查的欄位
            threshold: Z-Score 閾值 (default 3.0)
        
        返回:
            移除異適值的 DataFrame
        """
        df_clean = df.copy()
        
        outliers_removed = 0
        for col in columns:
            if col in df_clean.columns:
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                
                if std > 0:
                    z_scores = np.abs((df_clean[col] - mean) / std)
                    mask = z_scores <= threshold
                    removed = (~mask).sum()
                    outliers_removed += removed
                    df_clean = df_clean[mask]
        
        print(f"\n移除了 {outliers_removed} 条異適值")
        return df_clean
    
    def calculate_enhanced_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        計算強化会客特徵
        
        除了基本 OHLCV 特徵外，還有：
        - Volume 物犉: Volume / 平均 Volume
        - 买方比例: Taker Buy / Total
        - K 線身体: 正線 / 資標
        - 成交筆數 列足
        """
        df_enhanced = df.copy()
        
        # 1. 买方比例
        if 'taker_buy_quote_asset_volume' in df_enhanced.columns and 'quote_asset_volume' in df_enhanced.columns:
            df_enhanced['buy_sell_ratio'] = (
                df_enhanced['taker_buy_quote_asset_volume'] / 
                (df_enhanced['quote_asset_volume'] + 1e-8)
            )
        
        # 2. 成交量挪檕率
        if 'volume' in df_enhanced.columns:
            rolling_mean_volume = df_enhanced['volume'].rolling(20).mean()
            df_enhanced['volume_ratio'] = (
                df_enhanced['volume'] / (rolling_mean_volume + 1e-8)
            )
        
        # 3. K 線身体 (正線 = 收盐 > 開矗, 資標 = 收盐 < 開矗)
        df_enhanced['candle_body'] = abs(df_enhanced['close'] - df_enhanced['open'])
        df_enhanced['candle_wick_up'] = df_enhanced['high'] - df_enhanced[['open', 'close']].max(axis=1)
        df_enhanced['candle_wick_down'] = df_enhanced[['open', 'close']].min(axis=1) - df_enhanced['low']
        
        # 4. 拐動诺物
        if 'number_of_trades' in df_enhanced.columns:
            df_enhanced['trades_per_volume'] = (
                df_enhanced['number_of_trades'] / (df_enhanced['volume'] + 1e-8)
            )
        
        return df_enhanced
    
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        sequence_length: int = 30,
        test_ratio: float = 0.2,
        validation_ratio: float = 0.1
    ) -> Dict:
        """
        準備訓練數據
        
        參数:
            df: 批毆準備的 DataFrame
            sequence_length: 序栳長度
            test_ratio: 測試集比例
            validation_ratio: 驗證集比例
        
        返回:
            流程昨阻字典
        """
        from sklearn.preprocessing import StandardScaler
        
        # 提取數值特徵
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        data = df[numeric_cols].values
        
        # 正觀化
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(data)
        
        # 創建序栳
        sequences = []
        labels = []
        
        for i in range(len(scaled_data) - sequence_length):
            sequences.append(scaled_data[i:i+sequence_length])
            
            # 提供受僺況 (0=DOWN, 1=FLAT, 2=UP)
            price_change = (df['close'].iloc[i+sequence_length] - df['close'].iloc[i]) / df['close'].iloc[i] * 100
            
            if price_change < -0.5:
                label = 0  # DOWN
            elif price_change > 0.5:
                label = 2  # UP
            else:
                label = 1  # FLAT
            
            labels.append(label)
        
        sequences = np.array(sequences)
        labels = np.array(labels)
        
        # 分隔訓練/驗證/測試集
        n_total = len(sequences)
        n_test = int(n_total * test_ratio)
        n_val = int(n_total * validation_ratio)
        n_train = n_total - n_test - n_val
        
        X_train = sequences[:n_train]
        y_train = labels[:n_train]
        
        X_val = sequences[n_train:n_train+n_val]
        y_val = labels[n_train:n_train+n_val]
        
        X_test = sequences[n_train+n_val:]
        y_test = labels[n_train+n_val:]
        
        print(f"\n訓練集: {X_train.shape}")
        print(f"驗證集: {X_val.shape}")
        print(f"測試集: {X_test.shape}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'scaler': self.scaler,
            'feature_columns': numeric_cols.tolist()
        }


if __name__ == "__main__":
    print("ZigZag 量化系統 v1.2.0 - 數據加載模塊\n")
    print("模塊已準備，待整合整個系統...")
