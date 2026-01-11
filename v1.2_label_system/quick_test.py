#!/usr/bin/env python3
"""
快速驗證茶誕 - 测試推薦參數

推薦參數:
  fib_proximity: 0.002
  bb_proximity: 0.002
  zigzag_threshold: 0.3
"""

import sys
import os
from pathlib import Path

try:
    from label_generator import LabelGenerator
    from label_statistics import LabelStatistics
except ImportError as e:
    print(f"需要的模組未找到: {e}")
    print("請確保已安裝必要的依賴関係")
    sys.exit(1)


def main():
    print("\n" + "="*70)
    print("快速驗證测試")
    print("="*70)
    print()
    print("推薦參數:")
    print("  fib_proximity: 0.002")
    print("  bb_proximity: 0.002")
    print("  zigzag_threshold: 0.3")
    print()
    print("正在測試 BTCUSDT 15分鐘...")
    print("="*70)
    print()

    try:
        # 使用當前 config.yaml
        generator = LabelGenerator('config.yaml')
        
        print("第一步: 加載數據...")
        df = generator.generate_labels('BTCUSDT', '15m', save_path=None)
        print(f"✓ 已加載 {len(df)} 點數據")
        print()
        
        print("第二步: 生成推薦遮你...")
        report = LabelStatistics.generate_full_report(df, 'BTCUSDT', '15m')
        print("✓ 報告結果:")
        print()
        
        # 打印關鍵指標
        print("【進場詳情】")
        entry_info = report['entry_candidates']
        print(f"  進場比例: {entry_info['candidate_pct']:.2f}%")
        print(f"  成功率: {entry_info['success_rate']:.2f}%")
        print(f"  总進場数: {entry_info['total_candidates']}")
        print()
        
        print("【回報詳情】")
        returns_info = report['optimal_returns']
        print(f"  平均回報: {returns_info['mean_optimal_return']:.2f}%")
        print(f"  最大回報: {returns_info['max_optimal_return']:.2f}%")
        print(f"  最小回報: {returns_info['min_optimal_return']:.2f}%")
        print(f"  盈利比率: {returns_info['profitable_pct']:.2f}%")
        print()
        
        print("【品質詳情】")
        quality_info = report['quality_scores']
        print(f"  平均計算分: {quality_info['mean_quality_score']:.2f}")
        print(f"  中位數: {quality_info['median_quality_score']:.2f}")
        print(f"  最高: {quality_info['max_quality_score']:.2f}")
        print(f"  最低: {quality_info['min_quality_score']:.2f}")
        print()
        
        # 計算总体分數
        score = 0
        candidates = entry_info['candidate_pct']
        if 8 <= candidates <= 15:
            score += 30
        elif 5 <= candidates <= 25:
            score += 20 - abs(candidates - 15) * 0.5
        
        mean_return = returns_info['mean_optimal_return']
        if mean_return > 0.5:
            score += min(15, mean_return * 10)
        elif mean_return > 0:
            score += mean_return * 20
        
        profitable = returns_info['profitable_pct']
        if profitable > 85:
            score += 20
        elif profitable > 80:
            score += 15
        
        quality = quality_info['mean_quality_score']
        if quality > 50:
            score += 15
        elif quality > 45:
            score += 12
        
        print("="*70)
        print(f"淘推推薦骨 (recommendation score): {score:.2f}")
        print("="*70)
        print()
        
        # 評估
        print("【評估】")
        if 8 <= candidates <= 15:
            print(f"  ✓ 進場比例: 优勋 (8-15%)")
        elif 5 <= candidates <= 25:
            print(f"  ✓ 進場比例: 良好 (5-25%)")
        else:
            print(f"  ✗ 進場比例: 需要改進")
        
        if mean_return > 0.5:
            print(f"  ✓ 回報: 優秋 (>0.5%)")
        elif mean_return > 0:
            print(f"  ✓ 回報: 良好")
        else:
            print(f"  ✗ 回報: 低于预期")
        
        if profitable > 85:
            print(f"  ✓ 盈利比率: 优秋 (>85%)")
        elif profitable > 80:
            print(f"  ✓ 盈利比率: 良好 (>80%)")
        else:
            print(f"  ✗ 盈利比率: 需要改進")
        
        if quality > 50:
            print(f"  ✓ 品質分數: 优秋 (>50)")
        elif quality > 45:
            print(f"  ✓ 品質分數: 良好 (>45)")
        else:
            print(f"  ✗ 品質分數: 需要改進")
        
        print()
        print("="*70)
        print("测試完成!")
        print("="*70)
        print()
        print("第一步: 测試推薦參數 - 完成! ‼️")
        print()
        print("下一步建議:")
        print("  如果好 -> 混深粒度 Backtest")
        print("  如果候 -> 改進參數並繼續测試")
        print()
        
    except Exception as e:
        print(f"错誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
