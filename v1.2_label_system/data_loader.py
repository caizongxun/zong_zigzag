import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from huggingface_hub import hf_hub_download
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, repo_id: str, hf_root: str = "klines", cache_dir: Optional[str] = None):
        self.repo_id = repo_id
        self.hf_root = hf_root
        self.cache_dir = cache_dir or ".cache/hf_datasets"
        
    def load_klines(self, symbol: str, timeframe: str) -> pd.DataFrame:
        base = symbol.replace("USDT", "")
        filename = f"{base}_{timeframe}.parquet"
        path_in_repo = f"{self.hf_root}/{symbol}/{filename}"
        
        logger.info(f"Loading {symbol} {timeframe} from HuggingFace...")
        
        local_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=path_in_repo,
            repo_type="dataset",
            cache_dir=self.cache_dir
        )
        
        df = pd.read_parquet(local_path)
        
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        
        df = df.sort_values("open_time").reset_index(drop=True)
        df["symbol"] = symbol
        df["timeframe"] = timeframe
        
        logger.info(f"Loaded {len(df)} candles for {symbol} {timeframe}")
        
        return df
    
    def load_multi_symbol_multi_timeframe(
        self,
        symbols: list,
        timeframes: list
    ) -> dict:
        data_dict = {}
        
        for symbol in symbols:
            for tf in timeframes:
                try:
                    key = f"{symbol}_{tf}"
                    data_dict[key] = self.load_klines(symbol, tf)
                except Exception as e:
                    logger.error(f"Failed to load {symbol} {tf}: {str(e)}")
                    continue
        
        return data_dict
    
    def ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df["open_time"].dtype, "datetime64"):
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        return df
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> bool:
        required_cols = ["open_time", "open", "high", "low", "close", "volume"]
        return all(col in df.columns for col in required_cols)
    
    @staticmethod
    def get_base_symbol(symbol: str) -> str:
        return symbol.replace("USDT", "")
    
    @staticmethod
    def get_info(df: pd.DataFrame) -> dict:
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "date_range": f"{df['open_time'].min()} to {df['open_time'].max()}",
            "symbols": df["symbol"].unique().tolist() if "symbol" in df.columns else [],
            "timeframes": df["timeframe"].unique().tolist() if "timeframe" in df.columns else [],
            "missing_values": df.isnull().sum().sum()
        }
