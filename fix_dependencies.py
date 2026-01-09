#!/usr/bin/env python3
"""
修復 NumPy 和 Pandas 版本相容性問題

症狀:
  ValueError: numpy.dtype size changed, may indicate binary incompatibility.
  Expected 96 from C header, got 88 from PyObject

原因:
  NumPy 和 Pandas 編譯時使用的 NumPy 版本不一致

解決方案:
  1. 清除快取
  2. 重新安裝相容版本
"""

import subprocess
import sys
import os

print("="*70)
print("修復 NumPy-Pandas 相容性問題")
print("="*70)

# 步驟 1: 清除 pip 快取
print("\n步驟 1: 清除 pip 快取...")
try:
    subprocess.run(
        [sys.executable, "-m", "pip", "cache", "purge"],
        check=True,
        capture_output=True
    )
    print("✓ 快取已清除")
except Exception as e:
    print(f"⚠ 快取清除失敗: {e}")

# 步驟 2: 卸載衝突的套件
print("\n步驟 2: 卸載舊版本...")
packages_to_remove = ['pandas', 'numpy', 'xgboost']

for package in packages_to_remove:
    try:
        print(f"  卸載 {package}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", package],
            capture_output=True,
            check=False
        )
    except Exception as e:
        print(f"  ⚠ {package} 卸載失敗: {e}")

print("✓ 舊版本已卸載")

# 步驟 3: 安裝相容的版本
print("\n步驟 3: 安裝相容的版本...")
print("  安裝 numpy (最新穩定版)...")
try:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "numpy"],
        check=True
    )
    print("  ✓ numpy 安裝成功")
except Exception as e:
    print(f"  ✗ numpy 安裝失敗: {e}")
    sys.exit(1)

print("  安裝 pandas (相容版本)...")
try:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pandas"],
        check=True
    )
    print("  ✓ pandas 安裝成功")
except Exception as e:
    print(f"  ✗ pandas 安裝失敗: {e}")
    sys.exit(1)

print("  安裝其他依賴...")
try:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", 
         "xgboost", "scikit-learn", "Flask", "Flask-CORS", "yfinance"],
        check=True
    )
    print("  ✓ 其他依賴安裝成功")
except Exception as e:
    print(f"  ⚠ 部分依賴安裝失敗: {e}")

# 步驟 4: 驗證安裝
print("\n步驟 4: 驗證安裝...")
try:
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    
    print(f"  ✓ NumPy 版本: {np.__version__}")
    print(f"  ✓ Pandas 版本: {pd.__version__}")
    print(f"  ✓ XGBoost 版本: {xgb.__version__}")
    
    print("\n" + "="*70)
    print("✓ 修復完成！所有依賴已安裝")
    print("="*70)
    print("\n現在可以執行訓練:")
    print("  python train_complete_pipeline.py --pair BTCUSDT --interval 15m")
    print()
    
except ImportError as e:
    print(f"\n✗ 驗證失敗: {e}")
    print("\n請手動執行:")
    print(f"  {sys.executable} -m pip install --upgrade --force-reinstall numpy pandas")
    sys.exit(1)
