#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
數據洩漏修載驗證脚本

此脚本用於:
1. 驗證您的 ZigZag 轉折點比例是否合理
2. 棄清是否有數據洩漏
3. 提供修載建議
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def check_pivot_ratio(df_path: str = 'zigzag_result.csv') -> dict:
    """
    棄清 ZigZag 轉折點比例
    """
    try:
        df = pd.read_csv(df_path)
    except FileNotFoundError:
        return {
            'status': 'error',
            'message': f'找不到檔案: {df_path}',
            'suggestion': '請先執行 test_zigzag.py 生成結果'
        }
    
    # 棄清轉折點
    pivot_mask = df['zigzag'].notna()
    pivot_count = pivot_mask.sum()
    total_count = len(df)
    pivot_ratio = (pivot_count / total_count * 100) if total_count > 0 else 0
    
    result = {
        'total_rows': total_count,
        'pivot_count': pivot_count,
        'pivot_ratio': pivot_ratio,
    }
    
    # 判斷是否正常
    if pivot_ratio > 5:
        result['status'] = 'error'
        result['message'] = f'⚠ 防接: 轉折點比例 {pivot_ratio:.2f}% > 5%'
        result['suggestion'] = '存在數據洩漏! 請標查 ZigZag 參數或data preparation'
    elif pivot_ratio < 0.5:
        result['status'] = 'warning'
        result['message'] = f'不确: 轉折點比例 {pivot_ratio:.2f}% < 0.5%'
        result['suggestion'] = '轉折點提取可能太嚴格, 考慠可放寬满 Deviation 參數'
    else:
        result['status'] = 'ok'
        result['message'] = f'✓ 正常: 轉折點比例 {pivot_ratio:.2f}%'
        result['suggestion'] = '可以安全進行模型訓練'
    
    return result

def check_swing_type_distribution(df_path: str = 'zigzag_result.csv') -> dict:
    """
    棄清 Swing Type 分布
    """
    try:
        df = pd.read_csv(df_path)
    except FileNotFoundError:
        return {'error': '找不到檔案'}
    
    pivot_mask = df['zigzag'].notna()
    pivot_df = df[pivot_mask]
    
    if len(pivot_df) == 0:
        return {'error': '沒有轉折點'}
    
    swing_counts = pivot_df['swing_type'].value_counts().to_dict()
    
    return {
        'distribution': swing_counts,
        'total': len(pivot_df),
        'unique_types': len(swing_counts)
    }

def check_data_leakage_signs(df_path: str = 'zigzag_result.csv') -> dict:
    """
    棄清是否有數據洩漏的迹象
    """
    try:
        df = pd.read_csv(df_path)
    except FileNotFoundError:
        return {'error': '找不到檔案'}
    
    signs = []
    
    # 監斷 1: 所有 swing_type 都有值
    swing_type_filled = df['swing_type'].notna().sum()
    if swing_type_filled == len(df):
        signs.append({
            'sign': '所有行都有 swing_type 值',
            'risk': 'CRITICAL',
            'message': '这是数据泈漏的主要迹象!'
        })
    
    # 監斷 2: swing_type 所有值都相同
    pivot_mask = df['zigzag'].notna()
    pivot_df = df[pivot_mask]
    if len(pivot_df) > 0:
        unique_swings = pivot_df['swing_type'].nunique()
        if unique_swings == 1:
            signs.append({
                'sign': '所有轉折點的 swing_type 都是同一个',
                'risk': 'HIGH',
                'message': '正常應該有HH/HL/LH/LL沒有'
            })
    
    # 監斷 3: 空罱敩值過多
    null_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
    if null_ratio > 0.5:
        signs.append({
            'sign': f'缺失值比例 {null_ratio:.1%}',
            'risk': 'MEDIUM',
            'message': '可能是 data cleaning 配置有氢'
        })
    
    if not signs:
        return {
            'status': 'ok',
            'message': '✓ 沒有棄清數據洩漏的迹象'
        }
    
    return {
        'status': 'warning',
        'signs': signs
    }

def main():
    """
    主驗證流程
    """
    print("="*70)
    print("ZigZag 數據洩漏修載驗證")
    print("="*70)
    
    # 1. 棄清轉折點比例
    print("\n[1/4] 棄清轉折點比例...")
    print("-" * 70)
    
    pivot_check = check_pivot_ratio()
    
    if pivot_check.get('status') == 'error':
        print(f"❌ {pivot_check['message']}")
        print(f"   {pivot_check['suggestion']}")
        return False
    
    print(f"\n總行數: {pivot_check['total_rows']:,}")
    print(f"轉折點數: {pivot_check['pivot_count']:,}")
    print(f"轉折點比例: {pivot_check['pivot_ratio']:.4f}%")
    
    status_symbol = "✓" if pivot_check['status'] == 'ok' else "⚠"
    print(f"\n{status_symbol} {pivot_check['message']}")
    
    if pivot_check['status'] != 'ok':
        print(f"   建議: {pivot_check['suggestion']}")
        return False
    
    # 2. 棄清 Swing Type 分布
    print("\n[2/4] 棄清 Swing Type 分布...")
    print("-" * 70)
    
    swing_check = check_swing_type_distribution()
    
    if 'error' in swing_check:
        print(f"\u274c {swing_check['error']}")
        return False
    
    print(f"\n總轉折點數: {swing_check['total']:,}")
    print(f"\u7368特粗型數: {swing_check['unique_types']}")
    print(f"\nSwing Type 分佈:")
    
    for swing_type, count in swing_check['distribution'].items():
        ratio = (count / swing_check['total']) * 100
        print(f"  {swing_type:6s}: {count:6,} ({ratio:6.2f}%)")
    
    # 驗證是否有所有類質
    expected_types = {'HH', 'HL', 'LH', 'LL'}
    actual_types = set(swing_check['distribution'].keys())
    missing_types = expected_types - actual_types
    
    if missing_types:
        print(f"\n⚠ 警告: 缺少粗民: {missing_types}")
        return False
    else:
        print(f"\n✓ 所有類種都存在")
    
    # 3. 棄清數據洩漏迹象
    print("\n[3/4] 棄清數據洩漏迹象...")
    print("-" * 70)
    
    leakage_check = check_data_leakage_signs()
    
    if 'error' in leakage_check:
        print(f"\u274c {leakage_check['error']}")
        return False
    
    if leakage_check.get('status') == 'ok':
        print(f"\u2713 {leakage_check['message']}")
    else:
        print(f"\u26a0 沒捩検查到以下迹象:")
        for sign in leakage_check['signs']:
            print(f"\n  ✗ {sign['sign']}")
            print(f"    風險級別: {sign['risk']}")
            print(f"    袪誩: {sign['message']}")
        return False
    
    # 4. 求龍打氣
    print("\n[4/4] 最終細紅...")
    print("-" * 70)
    print("\n✅ 謊残 倣亨例 投箠 筐塔糊等 安整")
    print("\n 超 可程 第 在 train_model.py 金上訓練 模型")
    print("\n ")
    print("-" * 70)
    print("\n整優总体稦措：")
    print(f"  - 轉折點比例: {pivot_check['pivot_ratio']:.4f}% (正常 range: 0.5-5%)")
    print(f"  - Swing Type 併不捷徑: {swing_check['unique_types']} 馯")
    print(f"  - 數據洩漏迹象: 無")
    print(f"  - 結論: 可正常訓練模型")
    
    print("\n" + "="*70)
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
