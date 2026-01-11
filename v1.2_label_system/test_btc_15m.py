#!/usr/bin/env python

import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent))

from label_generator import LabelGenerator
from label_statistics import LabelStatistics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    print("\n" + "="*70)
    print("v1.2 Label System Test - BTC 15m")
    print("="*70)
    
    config_path = Path(__file__).parent / "config.yaml"
    
    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        return
    
    logger.info(f"Using config: {config_path}")
    
    try:
        generator = LabelGenerator(str(config_path))
        logger.info("LabelGenerator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize LabelGenerator: {str(e)}")
        return
    
    logger.info("\nGenerating labels for BTCUSDT 15m...")
    logger.info("This may take a few minutes depending on internet speed...\n")
    
    try:
        df = generator.generate_labels(
            symbol="BTCUSDT",
            timeframe="15m",
            save_path=None
        )
        logger.info(f"Successfully generated labels for {len(df)} candles")
    except Exception as e:
        logger.error(f"Failed to generate labels: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "-"*70)
    print("Data Overview")
    print("-"*70)
    print(f"Total candles: {len(df)}")
    print(f"Date range: {df['open_time'].min()} to {df['open_time'].max()}")
    print(f"Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    print("\n" + "-"*70)
    print("Columns in dataset")
    print("-"*70)
    print(f"Total columns: {len(df.columns)}")
    print("\nKey columns:")
    key_cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "atr", "bb_upper", "bb_middle", "bb_lower",
        "is_entry_candidate", "entry_reason", "entry_success",
        "entry_quality_score", "optimal_entry_price", "optimal_entry_return"
    ]
    for col in key_cols:
        if col in df.columns:
            print(f"  {col}")
    
    print("\n" + "-"*70)
    print("Entry Candidate Statistics")
    print("-"*70)
    
    candidates = df[df["is_entry_candidate"] == True]
    print(f"Entry candidates: {len(candidates)} ({len(candidates)/len(df)*100:.2f}%)")
    
    if len(candidates) > 0:
        successful = (candidates["entry_success"] == 1).sum()
        failed = (candidates["entry_success"] == 0).sum()
        
        print(f"\nEntry Success Breakdown:")
        print(f"  Successful: {successful} ({successful/len(candidates)*100:.2f}%)")
        print(f"  Failed: {failed} ({failed/len(candidates)*100:.2f}%)")
        
        print(f"\nEntry Quality Score:")
        print(f"  Mean: {candidates['entry_quality_score'].mean():.2f}")
        print(f"  Median: {candidates['entry_quality_score'].median():.2f}")
        print(f"  Min: {candidates['entry_quality_score'].min():.2f}")
        print(f"  Max: {candidates['entry_quality_score'].max():.2f}")
        print(f"  Std Dev: {candidates['entry_quality_score'].std():.2f}")
        
        print(f"\nEntry Reason Distribution:")
        for reason, count in candidates["entry_reason"].value_counts().items():
            print(f"  {reason}: {count} ({count/len(candidates)*100:.2f}%)")
        
        print(f"\nOptimal Entry Return:")
        returns = candidates["optimal_entry_return"] * 100
        print(f"  Mean: {returns.mean():.2f}%")
        print(f"  Median: {returns.median():.2f}%")
        print(f"  Min: {returns.min():.2f}%")
        print(f"  Max: {returns.max():.2f}%")
        
        profitable = (returns > 0).sum()
        print(f"  Profitable: {profitable} ({profitable/len(candidates)*100:.2f}%)")
    else:
        print("No entry candidates found!")
    
    print("\n" + "-"*70)
    print("Sample Entry Candidates (Top 10 by Quality Score)")
    print("-"*70)
    
    if len(candidates) > 0:
        top_candidates = candidates.nlargest(10, "entry_quality_score")
        for idx, row in top_candidates.iterrows():
            print(f"\n{row['open_time']} | Close: {row['close']:.2f}")
            print(f"  Entry Success: {row['entry_success']} | Quality Score: {row['entry_quality_score']:.2f}")
            print(f"  Reason: {row['entry_reason']} | Optimal Price: {row['optimal_entry_price']:.2f}")
            print(f"  Return: {row['optimal_entry_return']*100:.2f}%")
    else:
        print("No candidates to display")
    
    print("\n" + "-"*70)
    print("Technical Indicators Sample")
    print("-"*70)
    
    sample_idx = len(df) - 1
    sample = df.iloc[sample_idx]
    
    print(f"\nLatest candle ({sample['open_time']}):")
    print(f"  Close: {sample['close']:.2f}")
    print(f"  ATR: {sample['atr']:.2f}")
    print(f"  BB Upper: {sample['bb_upper']:.2f}")
    print(f"  BB Middle: {sample['bb_middle']:.2f}")
    print(f"  BB Lower: {sample['bb_lower']:.2f}")
    print(f"  BB Width: {sample['bb_width']:.2f}")
    
    fib_cols = [col for col in df.columns if col.startswith("fib_") and not col.endswith("_dist")]
    if fib_cols:
        print(f"\n  Fibonacci Levels:")
        for fib_col in sorted(fib_cols):
            print(f"    {fib_col}: {sample[fib_col]:.2f}")
    
    print("\n" + "-"*70)
    print("Generating Statistical Report")
    print("-"*70)
    
    try:
        report = LabelStatistics.generate_full_report(df, "BTCUSDT", "15m")
        LabelStatistics.print_report(report)
        
        report_path = Path("./output/BTCUSDT_15m_report.json")
        Path("./output").mkdir(exist_ok=True)
        LabelStatistics.save_report(report, str(report_path))
        logger.info(f"Report saved to {report_path}")
    except Exception as e:
        logger.error(f"Failed to generate report: {str(e)}")
    
    print("\n" + "="*70)
    print("Test completed successfully!")
    print("="*70 + "\n")
    
    print("Next steps:")
    print("1. Review the statistics above")
    print("2. Adjust config.yaml parameters if needed")
    print("3. Generate labels for other symbols/timeframes")
    print("4. Train Entry Validity classifier on the labeled data")
    print("\n")


if __name__ == "__main__":
    main()
