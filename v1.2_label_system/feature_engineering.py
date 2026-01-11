import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self, config: dict):
        self.config = config
        self.fib_levels = config.get("indicators", {}).get("fibonacci_levels", [0.236, 0.382, 0.5, 0.618, 0.705, 0.786])
        self.bb_period = config.get("indicators", {}).get("bollinger_period", 20)
        self.bb_std = config.get("indicators", {}).get("bollinger_std", 2)
        self.atr_period = config.get("indicators", {}).get("atr_period", 14)
    
    def calculate_atr(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        if period is None:
            period = self.atr_period
        
        df = df.copy()
        df["tr1"] = df["high"] - df["low"]
        df["tr2"] = abs(df["high"] - df["close"].shift(1))
        df["tr3"] = abs(df["low"] - df["close"].shift(1))
        df["tr"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
        
        atr = df["tr"].rolling(window=period).mean()
        return atr
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = None, std_dev: int = None) -> Tuple[pd.Series, pd.Series, pd.Series]:
        if period is None:
            period = self.bb_period
        if std_dev is None:
            std_dev = self.bb_std
        
        sma = df["close"].rolling(window=period).mean()
        std = df["close"].rolling(window=period).std()
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return upper, sma, lower
    
    def find_zigzag_swings(self, df: pd.DataFrame, threshold: float = 1.0) -> dict:
        close = df["close"].values
        swings = {
            "highs": [],
            "lows": [],
            "high_indices": [],
            "low_indices": []
        }
        
        if len(close) < 3:
            return swings
        
        i = 0
        direction = None
        
        while i < len(close) - 1:
            if direction is None:
                if close[i] > close[i + 1]:
                    direction = "down"
                    swings["highs"].append(close[i])
                    swings["high_indices"].append(i)
                elif close[i] < close[i + 1]:
                    direction = "up"
                    swings["lows"].append(close[i])
                    swings["low_indices"].append(i)
                i += 1
            elif direction == "down":
                if close[i] < close[i - 1]:
                    i += 1
                else:
                    if close[i - 1] < swings["highs"][-1] * (1 - threshold / 100):
                        swings["lows"].append(close[i - 1])
                        swings["low_indices"].append(i - 1)
                        direction = "up"
                    else:
                        i += 1
            elif direction == "up":
                if close[i] > close[i - 1]:
                    i += 1
                else:
                    if close[i - 1] > swings["lows"][-1] * (1 + threshold / 100):
                        swings["highs"].append(close[i - 1])
                        swings["high_indices"].append(i - 1)
                        direction = "down"
                    else:
                        i += 1
        
        return swings
    
    def calculate_fibonacci_levels(self, high: float, low: float) -> dict:
        diff = high - low
        levels = {}
        
        for level in self.fib_levels:
            levels[f"fib_{level}"] = high - (diff * level)
        
        return levels
    
    def get_distance_to_level(self, current_price: float, level: float) -> float:
        if level == 0:
            return float('inf')
        return abs(current_price - level) / level
    
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        logger.info("Calculating ATR...")
        df["atr"] = self.calculate_atr(df)
        
        logger.info("Calculating Bollinger Bands...")
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = self.calculate_bollinger_bands(df)
        df["bb_width"] = df["bb_upper"] - df["bb_lower"]
        df["bb_upper_dist"] = (df["bb_upper"] - df["close"]) / df["close"]
        df["bb_lower_dist"] = (df["close"] - df["bb_lower"]) / df["close"]
        
        logger.info("Finding ZigZag swings...")
        swings = self.find_zigzag_swings(df, self.config.get("indicators", {}).get("zigzag_threshold", 1.0))
        
        if swings["highs"] and swings["lows"]:
            latest_high = max(swings["highs"]) if swings["highs"] else df["high"].max()
            latest_low = min(swings["lows"]) if swings["lows"] else df["low"].min()
            
            logger.info(f"Latest swing high: {latest_high}, low: {latest_low}")
            fib_levels = self.calculate_fibonacci_levels(latest_high, latest_low)
            
            for fib_name, fib_price in fib_levels.items():
                df[fib_name] = fib_price
                df[f"{fib_name}_dist"] = (df["close"] - fib_price) / df["close"]
        
        logger.info("Feature engineering completed")
        
        return df
    
    @staticmethod
    def get_required_columns() -> List[str]:
        return [
            "atr",
            "bb_upper", "bb_middle", "bb_lower", "bb_width",
            "bb_upper_dist", "bb_lower_dist"
        ]
