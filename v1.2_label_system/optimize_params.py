#!/usr/bin/env python

import json
import yaml
from pathlib import Path
from label_generator import LabelGenerator
from label_statistics import LabelStatistics

def test_parameters(config_updates: dict, test_name: str):
    """
    Test label generation with updated parameters
    """
    print(f"\n{'='*70}")
    print(f"Test: {test_name}")
    print(f"{'='*70}")
    
    # Load base config
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Update with test parameters
    for key, value in config_updates.items():
        if isinstance(value, dict):
            if key not in config:
                config[key] = {}
            config[key].update(value)
        else:
            config[key] = value
    
    print(f"\nParameters:")
    print(f"  fib_proximity: {config['entry_validation']['fib_proximity']}")
    print(f"  bb_proximity: {config['entry_validation']['bb_proximity']}")
    print(f"  zigzag_threshold: {config['indicators']['zigzag_threshold']}")
    
    # Generate labels with temp config
    temp_config_path = f'config_test_{test_name.replace(" ", "_")}.yaml'
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
    
    try:
        generator = LabelGenerator(temp_config_path)
        df = generator.generate_labels('BTCUSDT', '15m', save_path=None)
        
        # Generate report
        report = LabelStatistics.generate_full_report(df, 'BTCUSDT', '15m')
        
        # Extract key metrics
        metrics = {
            'entry_candidates_pct': report['entry_candidates']['candidate_pct'],
            'success_rate': report['entry_candidates']['success_rate'],
            'mean_return': report['optimal_returns']['mean_optimal_return'],
            'max_return': report['optimal_returns']['max_optimal_return'],
            'profitable_pct': report['optimal_returns']['profitable_pct'],
            'mean_quality': report['quality_scores']['mean_quality_score'],
            'entry_reason': report['entry_reasons']
        }
        
        # Print results
        print(f"\nResults:")
        print(f"  Entry candidates: {metrics['entry_candidates_pct']:.2f}%")
        print(f"  Success rate: {metrics['success_rate']:.2f}%")
        print(f"  Mean return: {metrics['mean_return']:.2f}%")
        print(f"  Max return: {metrics['max_return']:.2f}%")
        print(f"  Profitable: {metrics['profitable_pct']:.2f}%")
        print(f"  Mean quality score: {metrics['mean_quality']:.2f}")
        print(f"  Entry reasons: {metrics['entry_reason']}")
        
        # Evaluation
        print(f"\nEvaluation:")
        
        # Entry candidate ratio
        if metrics['entry_candidates_pct'] < 20:
            print(f"  ✓ Entry candidates ({metrics['entry_candidates_pct']:.2f}%) - Good")
        elif metrics['entry_candidates_pct'] < 50:
            print(f"  ~ Entry candidates ({metrics['entry_candidates_pct']:.2f}%) - Acceptable")
        else:
            print(f"  ✗ Entry candidates ({metrics['entry_candidates_pct']:.2f}%) - Too many")
        
        # Mean return
        if metrics['mean_return'] > 0.5:
            print(f"  ✓ Mean return ({metrics['mean_return']:.2f}%) - Good")
        elif metrics['mean_return'] > 0:
            print(f"  ~ Mean return ({metrics['mean_return']:.2f}%) - Acceptable")
        else:
            print(f"  ✗ Mean return ({metrics['mean_return']:.2f}%) - Negative")
        
        # Profitable rate
        if metrics['profitable_pct'] > 80:
            print(f"  ✓ Profitable ({metrics['profitable_pct']:.2f}%) - Excellent")
        elif metrics['profitable_pct'] > 60:
            print(f"  ~ Profitable ({metrics['profitable_pct']:.2f}%) - Good")
        else:
            print(f"  ✗ Profitable ({metrics['profitable_pct']:.2f}%) - Low")
        
        # Quality score
        if metrics['mean_quality'] > 50:
            print(f"  ✓ Quality score ({metrics['mean_quality']:.2f}) - Good")
        elif metrics['mean_quality'] > 40:
            print(f"  ~ Quality score ({metrics['mean_quality']:.2f}) - Acceptable")
        else:
            print(f"  ✗ Quality score ({metrics['mean_quality']:.2f}) - Low")
        
        # Save results
        results_path = Path('./output')
        results_path.mkdir(exist_ok=True)
        with open(results_path / f'test_{test_name.replace(" ", "_")}.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
        
    finally:
        # Cleanup temp config
        Path(temp_config_path).unlink()


def main():
    print("\n" + "="*70)
    print("Parameter Optimization Tests")
    print("="*70)
    
    # Test 1: Current settings (baseline)
    test1 = test_parameters(
        {'entry_validation': {'fib_proximity': 0.005, 'bb_proximity': 0.005},
         'indicators': {'zigzag_threshold': 0.5}},
        'Current Settings'
    )
    
    # Test 2: Medium strict
    test2 = test_parameters(
        {'entry_validation': {'fib_proximity': 0.005, 'bb_proximity': 0.002},
         'indicators': {'zigzag_threshold': 0.3}},
        'Medium Strict'
    )
    
    # Test 3: Very strict
    test3 = test_parameters(
        {'entry_validation': {'fib_proximity': 0.002, 'bb_proximity': 0.001},
         'indicators': {'zigzag_threshold': 0.2}},
        'Very Strict'
    )
    
    # Test 4: Balanced Fib/BB
    test4 = test_parameters(
        {'entry_validation': {'fib_proximity': 0.002, 'bb_proximity': 0.003},
         'indicators': {'zigzag_threshold': 0.2}},
        'Balanced Fib/BB'
    )
    
    # Summary
    print(f"\n{'='*70}")
    print("Summary Comparison")
    print(f"{'='*70}")
    
    tests = [
        ('Current Settings', test1),
        ('Medium Strict', test2),
        ('Very Strict', test3),
        ('Balanced Fib/BB', test4)
    ]
    
    print(f"\n{'Test Name':<20} {'Candidates':<12} {'Mean Return':<15} {'Profitable':<12} {'Quality':<10}")
    print("-" * 70)
    
    for name, metrics in tests:
        print(f"{name:<20} {metrics['entry_candidates_pct']:>10.2f}% {metrics['mean_return']:>13.2f}% {metrics['profitable_pct']:>10.2f}% {metrics['mean_quality']:>8.2f}")
    
    print(f"\n{'='*70}")
    print("Recommendation:")
    print("Choose the test that best balances:")
    print("  - Entry candidates: 8-15% (sparse but meaningful)")
    print("  - Mean return: > 0.5% (positive returns)")
    print("  - Profitable: > 80% (high success rate)")
    print("  - Quality: > 45 (good quality entries)")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
