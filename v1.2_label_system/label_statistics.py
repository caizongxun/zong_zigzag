import pandas as pd
import numpy as np
from typing import Dict
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LabelStatistics:
    def __init__(self):
        pass
    
    @staticmethod
    def analyze_entry_candidates(df: pd.DataFrame) -> dict:
        if "is_entry_candidate" not in df.columns:
            return {}
        
        candidates = df[df["is_entry_candidate"] == True]
        
        if len(candidates) == 0:
            return {
                "total_candles": len(df),
                "entry_candidates": 0,
                "candidate_pct": 0.0,
                "message": "No entry candidates found"
            }
        
        stats = {
            "total_candles": len(df),
            "entry_candidates": len(candidates),
            "candidate_pct": round(len(candidates) / len(df) * 100, 2),
            "successful_entries": int((candidates["entry_success"] == 1).sum()),
            "failed_entries": int((candidates["entry_success"] == 0).sum()),
        }
        
        if len(candidates) > 0:
            stats["success_rate"] = round(stats["successful_entries"] / len(candidates) * 100, 2)
        
        return stats
    
    @staticmethod
    def analyze_quality_scores(df: pd.DataFrame) -> dict:
        if "entry_quality_score" not in df.columns:
            return {}
        
        candidates = df[df["is_entry_candidate"] == True]
        
        if len(candidates) == 0:
            return {}
        
        scores = candidates["entry_quality_score"]
        
        return {
            "mean_quality_score": round(float(scores.mean()), 2),
            "median_quality_score": round(float(scores.median()), 2),
            "std_quality_score": round(float(scores.std()), 2),
            "min_quality_score": int(scores.min()),
            "max_quality_score": int(scores.max()),
            "q25_quality_score": round(float(scores.quantile(0.25)), 2),
            "q75_quality_score": round(float(scores.quantile(0.75)), 2),
        }
    
    @staticmethod
    def analyze_entry_reasons(df: pd.DataFrame) -> dict:
        if "entry_reason" not in df.columns:
            return {}
        
        candidates = df[df["is_entry_candidate"] == True]
        
        if len(candidates) == 0:
            return {}
        
        reason_dist = candidates["entry_reason"].value_counts()
        
        result = {}
        for reason, count in reason_dist.items():
            result[reason] = {
                "count": int(count),
                "pct": round(count / len(candidates) * 100, 2)
            }
        
        return result
    
    @staticmethod
    def analyze_optimal_returns(df: pd.DataFrame) -> dict:
        if "optimal_entry_return" not in df.columns:
            return {}
        
        candidates = df[df["is_entry_candidate"] == True]
        
        if len(candidates) == 0:
            return {}
        
        returns = candidates["optimal_entry_return"]
        
        profitable = (returns > 0).sum()
        loss = (returns < 0).sum()
        breakeven = (returns == 0).sum()
        
        return {
            "mean_optimal_return": round(float(returns.mean()) * 100, 2),
            "median_optimal_return": round(float(returns.median()) * 100, 2),
            "std_optimal_return": round(float(returns.std()) * 100, 2),
            "min_optimal_return": round(float(returns.min()) * 100, 2),
            "max_optimal_return": round(float(returns.max()) * 100, 2),
            "profitable_count": int(profitable),
            "loss_count": int(loss),
            "breakeven_count": int(breakeven),
            "profitable_pct": round(profitable / len(candidates) * 100, 2),
        }
    
    @staticmethod
    def generate_full_report(df: pd.DataFrame, symbol: str, timeframe: str) -> dict:
        report = {
            "metadata": {
                "symbol": symbol,
                "timeframe": timeframe,
                "total_candles": len(df),
                "date_range": f"{df['open_time'].min()} to {df['open_time'].max()}" if "open_time" in df.columns else "N/A"
            },
            "entry_candidates": LabelStatistics.analyze_entry_candidates(df),
            "quality_scores": LabelStatistics.analyze_quality_scores(df),
            "entry_reasons": LabelStatistics.analyze_entry_reasons(df),
            "optimal_returns": LabelStatistics.analyze_optimal_returns(df),
        }
        
        return report
    
    @staticmethod
    def print_report(report: dict):
        print("\n" + "="*60)
        print("LABEL GENERATION REPORT")
        print("="*60)
        
        print(f"\nMetadata:")
        for key, value in report["metadata"].items():
            print(f"  {key}: {value}")
        
        print(f"\nEntry Candidates:")
        for key, value in report["entry_candidates"].items():
            if key != "message":
                print(f"  {key}: {value}")
        
        print(f"\nQuality Scores:")
        for key, value in report["quality_scores"].items():
            print(f"  {key}: {value}")
        
        print(f"\nEntry Reasons:")
        if report["entry_reasons"]:
            for reason, stats in report["entry_reasons"].items():
                print(f"  {reason}: {stats['count']} ({stats['pct']}%)")
        
        print(f"\nOptimal Returns:")
        for key, value in report["optimal_returns"].items():
            print(f"  {key}: {value}")
        
        print("\n" + "="*60 + "\n")
    
    @staticmethod
    def save_report(report: dict, path: str):
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Report saved to {path}")
