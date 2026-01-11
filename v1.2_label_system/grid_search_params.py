#!/usr/bin/env python

import json
import yaml
from pathlib import Path
import pandas as pd
from label_generator import LabelGenerator
from label_statistics import LabelStatistics
import logging

logging.basicConfig(level=logging.WARNING)

def test_parameters(config_updates: dict, test_name: str, verbose: bool = False):
    """
    Test label generation with updated parameters
    Returns metrics dict or None if failed
    """
    if verbose:
        print(f"Testing: {test_name}...", end=" ", flush=True)
    
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
    
    # Generate labels with temp config
    temp_config_path = f'config_temp.yaml'
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
    
    try:
        generator = LabelGenerator(temp_config_path)
        df = generator.generate_labels('BTCUSDT', '15m', save_path=None)
        
        # Generate report
        report = LabelStatistics.generate_full_report(df, 'BTCUSDT', '15m')
        
        # Extract key metrics
        metrics = {
            'fib_proximity': config['entry_validation']['fib_proximity'],
            'bb_proximity': config['entry_validation']['bb_proximity'],
            'zigzag_threshold': config['indicators']['zigzag_threshold'],
            'entry_candidates_pct': report['entry_candidates']['candidate_pct'],
            'success_rate': report['entry_candidates']['success_rate'],
            'mean_return': report['optimal_returns']['mean_optimal_return'],
            'max_return': report['optimal_returns']['max_optimal_return'],
            'profitable_pct': report['optimal_returns']['profitable_pct'],
            'mean_quality': report['quality_scores']['mean_quality_score'],
            'median_quality': report['quality_scores']['median_quality_score'],
            'min_return': report['optimal_returns']['min_optimal_return'],
        }
        
        if verbose:
            print(f"OK (candidates: {metrics['entry_candidates_pct']:.1f}%, return: {metrics['mean_return']:.2f}%)")
        
        return metrics
        
    except Exception as e:
        if verbose:
            print(f"FAILED ({str(e)[:30]}...)")
        return None
        
    finally:
        # Cleanup temp config
        if Path(temp_config_path).exists():
            Path(temp_config_path).unlink()


def calculate_score(metrics: dict) -> float:
    """
    Calculate overall score for the parameter combination.
    Considers multiple objectives:
    - Entry candidates: target 8-15% (penalty for too high or too low)
    - Mean return: target > 0.5% (more is better)
    - Profitable %: target > 80% (more is better)
    - Quality score: target > 45 (more is better)
    - Success rate: target 40-60% (reasonable)
    """
    score = 0
    penalties = 0
    
    # Entry candidates score (optimal: 8-15%)
    candidates = metrics['entry_candidates_pct']
    if 8 <= candidates <= 15:
        score += 30  # Perfect range
    elif 5 <= candidates <= 25:
        score += 20 - abs(candidates - 15) * 0.5  # Good range
    elif 3 <= candidates <= 40:
        score += 10 - abs(candidates - 15) * 0.1  # Acceptable range
    else:
        penalties += abs(candidates - 15) * 0.5  # Bad range
    
    # Mean return score (more is better, target > 0.5%)
    mean_return = metrics['mean_return']
    if mean_return > 0.5:
        score += min(15, mean_return * 10)  # Up to 15 points
    elif mean_return > 0:
        score += mean_return * 20  # Up to 10 points
    else:
        penalties += 20  # Negative return is bad
    
    # Profitable % score (target > 80%)
    profitable = metrics['profitable_pct']
    if profitable > 85:
        score += 20
    elif profitable > 80:
        score += 15
    elif profitable > 70:
        score += 10
    else:
        penalties += (80 - profitable) * 0.1
    
    # Quality score (target > 45)
    quality = metrics['mean_quality']
    if quality > 50:
        score += 15
    elif quality > 45:
        score += 12
    elif quality > 40:
        score += 8
    else:
        penalties += (45 - quality) * 0.5
    
    # Success rate (target 30-50%)
    success = metrics['success_rate']
    if 30 <= success <= 50:
        score += 10
    elif 25 <= success <= 60:
        score += 5
    else:
        penalties += (50 - success) * 0.1 if success < 25 else (success - 50) * 0.1
    
    # Stability bonus: lower std of quality is better
    median_quality = metrics['median_quality']
    if metrics['mean_quality'] > 0 and median_quality > 0:
        std_quality = metrics['mean_quality'] - median_quality
        if std_quality < 5:
            score += 5  # Good stability
    
    return score - penalties


def main():
    print("\n" + "="*80)
    print("Grid Search: Comprehensive Parameter Optimization")
    print("="*80)
    
    # Define parameter ranges to test
    fib_proximities = [0.001, 0.002, 0.003, 0.005, 0.007, 0.01]
    bb_proximities = [0.001, 0.002, 0.003, 0.005, 0.007, 0.01]
    zigzag_thresholds = [0.2, 0.3, 0.5, 0.7, 1.0]
    
    total_combinations = len(fib_proximities) * len(bb_proximities) * len(zigzag_thresholds)
    print(f"\nTesting {total_combinations} parameter combinations...")
    print(f"Expected time: ~{total_combinations * 2.5 / 60:.1f} minutes\n")
    
    results = []
    tested = 0
    
    # Grid search
    for fib_prox in fib_proximities:
        for bb_prox in bb_proximities:
            for zigzag in zigzag_thresholds:
                tested += 1
                print(f"[{tested}/{total_combinations}] fib={fib_prox:.3f}, bb={bb_prox:.3f}, zigzag={zigzag:.1f}", end=" ")
                
                config_updates = {
                    'entry_validation': {
                        'fib_proximity': fib_prox,
                        'bb_proximity': bb_prox,
                    },
                    'indicators': {
                        'zigzag_threshold': zigzag,
                    }
                }
                
                metrics = test_parameters(config_updates, f"fib{fib_prox}_bb{bb_prox}_zz{zigzag}")
                
                if metrics:
                    score = calculate_score(metrics)
                    metrics['score'] = score
                    results.append(metrics)
                    print(f"Score: {score:.2f}")
                else:
                    print("FAILED")
    
    # Sort by score
    results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
    
    # Display results
    print("\n" + "="*80)
    print("TOP 10 BEST PARAMETER COMBINATIONS")
    print("="*80)
    
    columns = ['Rank', 'Score', 'Fib', 'BB', 'ZigZag', 'Candidates%', 'Return%', 'Profitable%', 'Quality']
    print(f"\n{columns[0]:<5} {columns[1]:<7} {columns[2]:<8} {columns[3]:<8} {columns[4]:<8} {columns[5]:<12} {columns[6]:<10} {columns[7]:<12} {columns[8]:<10}")
    print("-" * 80)
    
    for i, result in enumerate(results_sorted[:10], 1):
        print(f"{i:<5} {result['score']:<7.2f} {result['fib_proximity']:<8.4f} {result['bb_proximity']:<8.4f} {result['zigzag_threshold']:<8.1f} {result['entry_candidates_pct']:<12.2f} {result['mean_return']:<10.2f} {result['profitable_pct']:<12.2f} {result['mean_quality']:<10.2f}")
    
    # Detailed analysis of top 3
    print("\n" + "="*80)
    print("DETAILED ANALYSIS OF TOP 3")
    print("="*80)
    
    for i, result in enumerate(results_sorted[:3], 1):
        print(f"\nRank #{i} - Score: {result['score']:.2f}")
        print(f"  Parameters:")
        print(f"    fib_proximity: {result['fib_proximity']:.4f} (0.{int(result['fib_proximity']*1000)}%)")
        print(f"    bb_proximity: {result['bb_proximity']:.4f} (0.{int(result['bb_proximity']*1000)}%)")
        print(f"    zigzag_threshold: {result['zigzag_threshold']:.1f}")
        print(f"  Results:")
        print(f"    Entry candidates: {result['entry_candidates_pct']:.2f}%")
        print(f"    Success rate: {result['success_rate']:.2f}%")
        print(f"    Mean return: {result['mean_return']:.2f}%")
        print(f"    Max return: {result['max_return']:.2f}%")
        print(f"    Min return: {result['min_return']:.2f}%")
        print(f"    Profitable: {result['profitable_pct']:.2f}%")
        print(f"    Mean quality score: {result['mean_quality']:.2f}")
        print(f"    Median quality score: {result['median_quality']:.2f}")
        
        # Evaluation
        print(f"  Evaluation:")
        if 8 <= result['entry_candidates_pct'] <= 15:
            print(f"    Entry candidates - EXCELLENT")
        elif 5 <= result['entry_candidates_pct'] <= 25:
            print(f"    Entry candidates - GOOD")
        else:
            print(f"    Entry candidates - NEEDS IMPROVEMENT")
        
        if result['mean_return'] > 1:
            print(f"    Return - EXCELLENT")
        elif result['mean_return'] > 0.5:
            print(f"    Return - GOOD")
        else:
            print(f"    Return - ACCEPTABLE")
        
        if result['profitable_pct'] > 85:
            print(f"    Profitable rate - EXCELLENT")
        elif result['profitable_pct'] > 80:
            print(f"    Profitable rate - GOOD")
        else:
            print(f"    Profitable rate - ACCEPTABLE")
        
        if result['mean_quality'] > 50:
            print(f"    Quality score - EXCELLENT")
        elif result['mean_quality'] > 45:
            print(f"    Quality score - GOOD")
        else:
            print(f"    Quality score - NEEDS IMPROVEMENT")
    
    # Save all results
    results_path = Path('./output')
    results_path.mkdir(exist_ok=True)
    
    # Save as CSV for easy analysis
    df_results = pd.DataFrame(results_sorted)
    df_results.to_csv(results_path / 'grid_search_results.csv', index=False)
    print(f"\nFull results saved to: {results_path / 'grid_search_results.csv'}")
    
    # Save top result as recommended config
    best = results_sorted[0]
    recommended_config = {
        'entry_validation': {
            'lookahead_bars': 20,
            'profit_threshold': 1.5,
            'fib_proximity': float(best['fib_proximity']),
            'bb_proximity': float(best['bb_proximity']),
        },
        'indicators': {
            'bollinger_period': 20,
            'bollinger_std': 2,
            'atr_period': 14,
            'zigzag_threshold': float(best['zigzag_threshold']),
            'fibonacci_levels': [0.236, 0.382, 0.5, 0.618, 0.705, 0.786],
        },
        'processing': {
            'batch_size': 1000,
            'n_workers': 4,
        },
    }
    
    with open(results_path / 'recommended_config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(recommended_config, f)
    
    print(f"Recommended config saved to: {results_path / 'recommended_config.yaml'}")
    print(f"\nTo use the best parameters, copy the content of recommended_config.yaml to config.yaml")
    
    print("\n" + "="*80)
    print(f"Grid search completed! Tested {len(results)} combinations.")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
