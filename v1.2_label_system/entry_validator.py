import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntryValidator:
    def __init__(self, config: dict):
        self.config = config
        self.lookahead_bars = config.get("entry_validation", {}).get("lookahead_bars", 20)
        self.profit_threshold = config.get("entry_validation", {}).get("profit_threshold", 1.5)
        self.fib_proximity = config.get("entry_validation", {}).get("fib_proximity", 0.01)
        self.bb_proximity = config.get("entry_validation", {}).get("bb_proximity", 0.01)
    
    def is_entry_candidate(
        self,
        df: pd.DataFrame,
        idx: int,
        fib_cols: List[str],
        bb_cols: List[str]
    ) -> Tuple[bool, str]:
        if idx >= len(df) - 1:
            return False, "not_enough_data"
        
        close = df.iloc[idx]["close"]
        
        fib_triggered = self._check_fib_proximity(df, idx, fib_cols)
        bb_triggered = self._check_bb_proximity(df, idx, bb_cols)
        
        if fib_triggered:
            return True, "fib"
        if bb_triggered:
            return True, "bb"
        
        return False, "no_trigger"
    
    def _check_fib_proximity(self, df: pd.DataFrame, idx: int, fib_cols: List[str]) -> bool:
        close = df.iloc[idx]["close"]
        
        for fib_col in fib_cols:
            if fib_col not in df.columns:
                continue
            
            fib_level = df.iloc[idx][fib_col]
            if pd.isna(fib_level):
                continue
            
            dist = abs(close - fib_level) / fib_level
            if dist < self.fib_proximity:
                return True
        
        return False
    
    def _check_bb_proximity(self, df: pd.DataFrame, idx: int, bb_cols: List[str]) -> bool:
        close = df.iloc[idx]["close"]
        
        if "bb_upper" in df.columns and "bb_lower" in df.columns:
            bb_upper = df.iloc[idx]["bb_upper"]
            bb_lower = df.iloc[idx]["bb_lower"]
            
            if pd.notna(bb_upper) and pd.notna(bb_lower):
                upper_dist = abs(close - bb_upper) / bb_upper
                lower_dist = abs(close - bb_lower) / bb_lower
                
                if upper_dist < self.bb_proximity or lower_dist < self.bb_proximity:
                    return True
        
        return False
    
    def label_entry_success(
        self,
        df: pd.DataFrame,
        entry_idx: int,
        direction: str = "long"
    ) -> int:
        if entry_idx >= len(df) - self.lookahead_bars:
            return 0
        
        entry_price = df.iloc[entry_idx]["close"]
        future_prices = df.iloc[entry_idx:entry_idx + self.lookahead_bars]["close"]
        
        if direction == "long":
            max_profit = (future_prices.max() - entry_price) / entry_price
            max_loss = (future_prices.min() - entry_price) / entry_price
        else:
            max_profit = (entry_price - future_prices.min()) / entry_price
            max_loss = (entry_price - future_prices.max()) / entry_price
        
        if abs(max_loss) > 0:
            ratio = max_profit / abs(max_loss)
            if ratio > self.profit_threshold:
                return 1
        
        return 0
    
    def label_entry_quality(
        self,
        df: pd.DataFrame,
        entry_idx: int,
        direction: str = "long"
    ) -> int:
        if entry_idx >= len(df) - self.lookahead_bars:
            return 0
        
        entry_price = df.iloc[entry_idx]["close"]
        future_prices = df.iloc[entry_idx:entry_idx + self.lookahead_bars]["close"]
        
        if direction == "long":
            returns = (future_prices - entry_price) / entry_price
        else:
            returns = (entry_price - future_prices) / entry_price
        
        score = 0
        
        max_return = returns.max()
        score += min(40, max(0, max_return * 100))
        
        max_drawdown = -returns.min()
        score += max(0, 30 - max(0, max_drawdown * 100))
        
        if max_drawdown > 0:
            rrr = max_return / max_drawdown
            if rrr > 3:
                score += 20
            elif rrr > 2:
                score += 15
            elif rrr > 1:
                score += 10
        
        win_rate = (returns > 0).sum() / len(returns)
        score += win_rate * 10
        
        return int(min(100, score))
    
    def label_optimal_entry(
        self,
        df: pd.DataFrame,
        entry_idx: int,
        direction: str = "long"
    ) -> Tuple[float, float]:
        if entry_idx >= len(df) - self.lookahead_bars:
            return df.iloc[entry_idx]["close"], 0
        
        entry_price = df.iloc[entry_idx]["close"]
        future_prices = df.iloc[entry_idx:entry_idx + self.lookahead_bars]["close"]
        
        if direction == "long":
            max_price = future_prices.max()
            optimal_return = (max_price - entry_price) / entry_price
        else:
            min_price = future_prices.min()
            optimal_return = (entry_price - min_price) / entry_price
        
        return max_price if direction == "long" else min_price, optimal_return
    
    def generate_all_labels(
        self,
        df: pd.DataFrame,
        fib_cols: Optional[List[str]] = None,
        bb_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        df = df.copy()
        
        if fib_cols is None:
            fib_cols = [col for col in df.columns if col.startswith("fib_") and col.endswith("_dist") == False and col != "fib_proximity"]
        if bb_cols is None:
            bb_cols = ["bb_upper", "bb_lower"]
        
        logger.info(f"Generating labels for {len(df)} candles...")
        
        df["is_entry_candidate"] = False
        df["entry_reason"] = "no_trigger"
        df["entry_success"] = 0
        df["entry_quality_score"] = 0
        df["optimal_entry_price"] = df["close"]
        df["optimal_entry_return"] = 0.0
        
        for idx in range(len(df) - self.lookahead_bars):
            is_candidate, reason = self.is_entry_candidate(df, idx, fib_cols, bb_cols)
            
            if is_candidate:
                df.at[idx, "is_entry_candidate"] = True
                df.at[idx, "entry_reason"] = reason
                
                df.at[idx, "entry_success"] = self.label_entry_success(df, idx, direction="long")
                
                df.at[idx, "entry_quality_score"] = self.label_entry_quality(df, idx, direction="long")
                
                optimal_price, return_pct = self.label_optimal_entry(df, idx, direction="long")
                df.at[idx, "optimal_entry_price"] = optimal_price
                df.at[idx, "optimal_entry_return"] = return_pct
        
        logger.info(f"Generated labels. Entry candidates: {df['is_entry_candidate'].sum()}")
        logger.info(f"Entry success rate: {(df['entry_success']==1).sum() / df['is_entry_candidate'].sum() * 100:.2f}%" 
                   if df['is_entry_candidate'].sum() > 0 else "No candidates")
        
        return df
