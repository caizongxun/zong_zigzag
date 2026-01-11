import pandas as pd
import yaml
from pathlib import Path
from typing import Optional, Dict
import logging
from datetime import datetime

from .data_loader import DataLoader
from .feature_engineering import FeatureEngineer
from .entry_validator import EntryValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LabelGenerator:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        self.data_loader = DataLoader(
            repo_id=self.config["huggingface"]["repo_id"],
            hf_root=self.config["huggingface"]["hf_root"],
            cache_dir=self.config["huggingface"].get("cache_dir")
        )
        
        self.feature_engineer = FeatureEngineer(self.config)
        self.entry_validator = EntryValidator(self.config)
    
    @staticmethod
    def _load_config(config_path: str) -> dict:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    
    def generate_labels(
        self,
        symbol: str,
        timeframe: str,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        logger.info(f"Starting label generation for {symbol} {timeframe}")
        start_time = datetime.now()
        
        df = self.data_loader.load_klines(symbol, timeframe)
        logger.info(f"Loaded {len(df)} candles")
        
        df = self.feature_engineer.extract_all_features(df)
        logger.info("Features extracted")
        
        fib_cols = [col for col in df.columns if col.startswith("fib_") and not col.endswith("_dist")]
        bb_cols = ["bb_upper", "bb_lower"]
        
        df = self.entry_validator.generate_all_labels(df, fib_cols, bb_cols)
        logger.info("Labels generated")
        
        if save_path:
            df.to_parquet(save_path)
            logger.info(f"Saved labeled data to {save_path}")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Label generation completed in {elapsed:.2f} seconds")
        
        return df
    
    def generate_batch(
        self,
        symbols: Optional[list] = None,
        timeframes: Optional[list] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        if symbols is None:
            symbols = self.config.get("symbols", [])
        if timeframes is None:
            timeframes = self.config.get("timeframes", ["15m", "1h"])
        
        results = {}
        
        for symbol in symbols:
            for tf in timeframes:
                try:
                    key = f"{symbol}_{tf}"
                    
                    save_path = None
                    if output_dir:
                        Path(output_dir).mkdir(parents=True, exist_ok=True)
                        save_path = f"{output_dir}/{key}_labeled.parquet"
                    
                    df = self.generate_labels(symbol, tf, save_path)
                    results[key] = df
                    
                except Exception as e:
                    logger.error(f"Failed to generate labels for {symbol} {tf}: {str(e)}")
                    continue
        
        return results
    
    def get_label_statistics(self, df: pd.DataFrame) -> dict:
        if "is_entry_candidate" not in df.columns:
            return {}
        
        candidates = df[df["is_entry_candidate"] == True]
        
        if len(candidates) == 0:
            return {
                "total_candles": len(df),
                "entry_candidates": 0,
                "candidate_pct": 0.0
            }
        
        successful = (candidates["entry_success"] == 1).sum()
        
        return {
            "total_candles": len(df),
            "entry_candidates": len(candidates),
            "candidate_pct": len(candidates) / len(df) * 100,
            "successful_entries": successful,
            "success_rate": successful / len(candidates) * 100 if len(candidates) > 0 else 0,
            "avg_quality_score": candidates["entry_quality_score"].mean(),
            "avg_optimal_return": candidates["optimal_entry_return"].mean(),
            "entry_reason_distribution": candidates["entry_reason"].value_counts().to_dict()
        }
