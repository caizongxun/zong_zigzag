import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from label_generator import LabelGenerator
from label_statistics import LabelStatistics


def example_single_symbol():
    print("\n=== Example 1: Generate labels for single symbol ===")
    
    config_path = str(Path(__file__).parent.parent / "config.yaml")
    generator = LabelGenerator(config_path)
    
    df = generator.generate_labels(
        symbol="BTCUSDT",
        timeframe="15m",
        save_path="./output/BTCUSDT_15m_labeled.parquet"
    )
    
    print(f"\nGenerated labels shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df[["open_time", "close", "is_entry_candidate", "entry_reason", "entry_success", "entry_quality_score"]].head(10))
    
    return df


def example_batch_generation():
    print("\n=== Example 2: Batch generate labels for multiple symbols ===")
    
    config_path = str(Path(__file__).parent.parent / "config.yaml")
    generator = LabelGenerator(config_path)
    
    symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
    timeframes = ["15m", "1h"]
    
    results = generator.generate_batch(
        symbols=symbols,
        timeframes=timeframes,
        output_dir="./output/labeled_data"
    )
    
    print(f"\nGenerated labels for {len(results)} symbol-timeframe pairs")
    
    return results


def example_statistics():
    print("\n=== Example 3: Analyze label statistics ===")
    
    config_path = str(Path(__file__).parent.parent / "config.yaml")
    generator = LabelGenerator(config_path)
    
    df = generator.generate_labels(
        symbol="ETHUSDT",
        timeframe="1h"
    )
    
    report = LabelStatistics.generate_full_report(df, "ETHUSDT", "1h")
    LabelStatistics.print_report(report)
    
    LabelStatistics.save_report(report, "./output/ETHUSDT_1h_report.json")
    
    return report


def example_entry_candidate_filtering():
    print("\n=== Example 4: Filter and analyze entry candidates ===")
    
    config_path = str(Path(__file__).parent.parent / "config.yaml")
    generator = LabelGenerator(config_path)
    
    df = generator.generate_labels(
        symbol="SOLUSDT",
        timeframe="15m"
    )
    
    candidates = df[df["is_entry_candidate"] == True]
    print(f"\nTotal candles: {len(df)}")
    print(f"Entry candidates: {len(candidates)} ({len(candidates)/len(df)*100:.2f}%)")
    
    successful = candidates[candidates["entry_success"] == 1]
    print(f"Successful entries: {len(successful)} ({len(successful)/len(candidates)*100:.2f}%)")
    
    print(f"\nAverage quality score: {candidates['entry_quality_score'].mean():.2f}")
    print(f"Average optimal return: {candidates['optimal_entry_return'].mean()*100:.2f}%")
    
    print(f"\nEntry reason distribution:")
    print(candidates["entry_reason"].value_counts())
    
    print(f"\nHigh quality entries (score > 70):")
    high_quality = candidates[candidates["entry_quality_score"] > 70]
    print(high_quality[["open_time", "close", "entry_quality_score", "optimal_entry_return", "entry_reason"]].head(10))
    
    return candidates


if __name__ == "__main__":
    import os
    os.makedirs("./output", exist_ok=True)
    os.makedirs("./output/labeled_data", exist_ok=True)
    
    print("\n" + "="*70)
    print("V1.2 Label System Examples")
    print("="*70)
    
    example_single_symbol()
    
    example_statistics()
    
    example_entry_candidate_filtering()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70 + "\n")
