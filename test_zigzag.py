import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import requests
from io import BytesIO
import sys
import argparse
import time

# 檢查依賴套件
def check_dependencies():
    """
    檢查必要的依賴套件是否安裝
    """
    missing = []
    
    try:
        import pyarrow
    except ImportError:
        missing.append('pyarrow')
    
    if missing:
        print("\n錯誤: 缺少必要的依賴套件")
        print("\n請執行以下指令安裝:")
        print(f"  pip install {' '.join(missing)}")
        print("\n或者安裝所有建議的套件:")
        print("  pip install pandas numpy requests pyarrow")
        sys.exit(1)

class ZigZagMT4:
    """
    MT4風格ZigZag指標實現
    根據Pine Script原始邏輯轉換而來
    
    參數:
        depth: 回溯深度,尋找局部極值的最小窗口 (建議: 10-15為15m圖)
        deviation: 價格變動閾值(百分比),必須超過此值才確認新轉折點 (建議: 0.8-1.5%為15m圖)
        backstep: 連續極值點之間的最小K棒間隔 (建議: 2-3為15m圖)
    """
    
    def __init__(self, depth: int = 12, deviation: float = 1.0, backstep: int = 3):
        self.depth = depth
        self.deviation = deviation / 100.0  # 轉換為小數
        self.backstep = backstep
        
    def calculate_zigzag(self, data: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        計算ZigZag線和標記HH/HL/LL/LH
        使用MT4風格的算法
        
        Args:
            data: 包含 OHLC 數據的 DataFrame
            verbose: 是否顯示詳細處理資訊
            
        Returns:
            包含ZigZag標記的DataFrame
        """
        if verbose:
            print(f"\nZigZag參數: Depth={self.depth}, Deviation={self.deviation*100}%, Backstep={self.backstep}")
        
        start_time = time.time()
        df = data.copy()
        n = len(df)
        
        if verbose:
            print(f"處理數據量: {n:,} 條記錄")
        
        # 初始化結果陣列
        zigzag_high = np.full(n, np.nan)
        zigzag_low = np.full(n, np.nan)
        zigzag = np.full(n, np.nan)
        direction = np.zeros(n)
        swing_type = [''] * n
        
        # 第一步: 尋找所有局部高點和低點
        if verbose:
            print(f"\n步驟 1/3: 尋找局部極值 (depth={self.depth})...")
        
        step1_start = time.time()
        
        for i in range(self.depth, n):
            # 每處理10000筆顯示進度
            if verbose and i % 10000 == 0:
                progress = (i / n) * 100
                print(f"  進度: {progress:.1f}% ({i:,}/{n:,})")
            
            # 尋找局部高點
            is_high = True
            for j in range(max(0, i - self.depth), i):
                if df['high'].iloc[j] >= df['high'].iloc[i]:
                    is_high = False
                    break
            if is_high:
                zigzag_high[i] = df['high'].iloc[i]
            
            # 尋找局部低點
            is_low = True
            for j in range(max(0, i - self.depth), i):
                if df['low'].iloc[j] <= df['low'].iloc[i]:
                    is_low = False
                    break
            if is_low:
                zigzag_low[i] = df['low'].iloc[i]
        
        high_count = np.sum(~np.isnan(zigzag_high))
        low_count = np.sum(~np.isnan(zigzag_low))
        step1_time = time.time() - step1_start
        
        if verbose:
            print(f"  完成! 找到 {high_count:,} 個局部高點, {low_count:,} 個局部低點")
            print(f"  耗時: {step1_time:.2f} 秒")
        
        # 第二步: 應用Deviation和Backstep篩選
        if verbose:
            print(f"\n步驟 2/3: 應用Deviation={self.deviation*100}%和Backstep={self.backstep}篩選...")
        
        step2_start = time.time()
        
        last_high_idx = -1
        last_low_idx = -1
        last_high_price = 0
        last_low_price = 0
        confirmed_highs = []
        confirmed_lows = []
        
        # 尋找第一個高點
        for i in range(n):
            if not np.isnan(zigzag_high[i]):
                last_high_idx = i
                last_high_price = zigzag_high[i]
                confirmed_highs.append((i, last_high_price))
                break
        
        # 尋找第一個低點
        for i in range(n):
            if not np.isnan(zigzag_low[i]):
                last_low_idx = i
                last_low_price = zigzag_low[i]
                confirmed_lows.append((i, last_low_price))
                break
        
        # 掃描所有點
        for i in range(n):
            # 每處理20000筆顯示進度
            if verbose and i % 20000 == 0 and i > 0:
                progress = (i / n) * 100
                print(f"  進度: {progress:.1f}% ({i:,}/{n:,})")
            
            # 處理高點
            if not np.isnan(zigzag_high[i]):
                if last_high_idx == -1:
                    last_high_idx = i
                    last_high_price = zigzag_high[i]
                    confirmed_highs.append((i, last_high_price))
                else:
                    # 檢查距離和變化幅度
                    if i - last_high_idx >= self.backstep:
                        if last_low_idx > last_high_idx:
                            # 從低點之後的高點,檢查deviation
                            price_change = abs((zigzag_high[i] - last_low_price) / last_low_price)
                            if price_change >= self.deviation:
                                last_high_idx = i
                                last_high_price = zigzag_high[i]
                                confirmed_highs.append((i, last_high_price))
                        else:
                            # 連續高點,取更高的
                            if zigzag_high[i] > last_high_price:
                                # 移除舊的高點
                                if confirmed_highs and confirmed_highs[-1][0] == last_high_idx:
                                    confirmed_highs.pop()
                                last_high_idx = i
                                last_high_price = zigzag_high[i]
                                confirmed_highs.append((i, last_high_price))
            
            # 處理低點
            if not np.isnan(zigzag_low[i]):
                if last_low_idx == -1:
                    last_low_idx = i
                    last_low_price = zigzag_low[i]
                    confirmed_lows.append((i, last_low_price))
                else:
                    # 檢查距離和變化幅度
                    if i - last_low_idx >= self.backstep:
                        if last_high_idx > last_low_idx:
                            # 從高點之後的低點,檢查deviation
                            price_change = abs((zigzag_low[i] - last_high_price) / last_high_price)
                            if price_change >= self.deviation:
                                last_low_idx = i
                                last_low_price = zigzag_low[i]
                                confirmed_lows.append((i, last_low_price))
                        else:
                            # 連續低點,取更低的
                            if zigzag_low[i] < last_low_price:
                                # 移除舊的低點
                                if confirmed_lows and confirmed_lows[-1][0] == last_low_idx:
                                    confirmed_lows.pop()
                                last_low_idx = i
                                last_low_price = zigzag_low[i]
                                confirmed_lows.append((i, last_low_price))
        
        step2_time = time.time() - step2_start
        
        if verbose:
            print(f"  完成! 確認了 {len(confirmed_highs):,} 個高點, {len(confirmed_lows):,} 個低點")
            print(f"  耗時: {step2_time:.2f} 秒")
        
        # 第三步: 合併和排序所有轉折點
        if verbose:
            print(f"\n步驟 3/3: 合併轉折點並標記HH/HL/LL/LH...")
        
        step3_start = time.time()
        
        all_pivots = []
        for idx, price in confirmed_highs:
            all_pivots.append((idx, price, 'HIGH'))
        for idx, price in confirmed_lows:
            all_pivots.append((idx, price, 'LOW'))
        
        # 按時間排序
        all_pivots.sort(key=lambda x: x[0])
        
        # 標記swing type
        last_confirmed_high = None
        last_confirmed_low = None
        
        for i, (idx, price, ptype) in enumerate(all_pivots):
            zigzag[idx] = price
            
            if ptype == 'HIGH':
                direction[idx] = -1
                if last_confirmed_high is not None:
                    if price > last_confirmed_high:
                        swing_type[idx] = 'HH'
                    else:
                        swing_type[idx] = 'LH'
                last_confirmed_high = price
            else:  # LOW
                direction[idx] = 1
                if last_confirmed_low is not None:
                    if price < last_confirmed_low:
                        swing_type[idx] = 'LL'
                    else:
                        swing_type[idx] = 'HL'
                last_confirmed_low = price
        
        # 填充direction
        for i in range(1, n):
            if direction[i] == 0:
                direction[i] = direction[i-1]
        
        df['zigzag'] = zigzag
        df['direction'] = direction
        df['swing_type'] = swing_type
        
        step3_time = time.time() - step3_start
        total_time = time.time() - start_time
        
        if verbose:
            print(f"  完成! 總計 {len(all_pivots):,} 個轉折點")
            print(f"  耗時: {step3_time:.2f} 秒")
            print(f"\n總耗時: {total_time:.2f} 秒 ({total_time/60:.2f} 分鐘)")
            print(f"平均處理速度: {n/total_time:,.0f} 筆/秒")
        
        return df


def download_btc_data(url: str) -> pd.DataFrame:
    """
    從Hugging Face下載BTC 15m數據
    
    Args:
        url: Parquet檔案URL
        
    Returns:
        OHLCV DataFrame
    """
    print(f"正在下載數據: {url}")
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        df = pd.read_parquet(BytesIO(response.content))
        print(f"數據下載成功: {len(df):,} 條記錄")
        return df
    except requests.exceptions.RequestException as e:
        raise Exception(f"下載失敗: {str(e)}")
    except Exception as e:
        raise Exception(f"讀取Parquet檔案失敗: {str(e)}")


def parse_arguments():
    """
    解析命令列參數
    """
    parser = argparse.ArgumentParser(
        description='ZigZag指標測試程式 - MT4風格實現',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
參數說明:
  Depth:     尋找局部極值的回溯窗口大小 (較大值過濾更多小波動)
  Deviation: 價格必須變動的最小百分比 (較大值只顯示重大趨勢)
  Backstep:  連續轉折點的最小間隔K棒數 (防止過密集的信號)

建議配置 (15分鐘圖):
  標準:   --depth 12 --deviation 1.0 --backstep 3
  敷感:   --depth 10 --deviation 0.8 --backstep 2
  保守:   --depth 15 --deviation 1.5 --backstep 4

範例:
  # 使用預設參數,測試1000筆資料
  python test_zigzag.py
  
  # 自訂參數
  python test_zigzag.py --depth 12 --deviation 1.0 --backstep 3
  
  # 使用所有資料 (約21萬筆)
  python test_zigzag.py --all-data
  
  # 指定資料數量
  python test_zigzag.py --samples 5000
  
  # 自動調整參數
  python test_zigzag.py --auto-tune
        """
    )
    
    parser.add_argument('--depth', type=int, default=12,
                        help='Depth參數 (預設: 12, 範圍: 5-30)')
    parser.add_argument('--deviation', type=float, default=1.0,
                        help='Deviation參數,百分比 (預設: 1.0, 範圍: 0.3-10.0)')
    parser.add_argument('--backstep', type=int, default=3,
                        help='Backstep參數 (預設: 3, 範圍: 2-10)')
    parser.add_argument('--samples', type=int, default=1000,
                        help='測試數據筆數 (預設: 1000)')
    parser.add_argument('--all-data', action='store_true',
                        help='使用所有數據(約21萬筆),忽略--samples參數')
    parser.add_argument('--auto-tune', action='store_true',
                        help='自動測試多種參數組合 (仅在非--all-data模式下)')
    parser.add_argument('--output', type=str, default='zigzag_result.csv',
                        help='輸出檔案名稱 (預設: zigzag_result.csv)')
    
    return parser.parse_args()


def test_single_config(df_test: pd.DataFrame, depth: int, deviation: float, backstep: int, verbose: bool = True):
    """
    測試單一參數配置
    """
    zigzag = ZigZagMT4(depth=depth, deviation=deviation, backstep=backstep)
    result = zigzag.calculate_zigzag(df_test, verbose=verbose)
    
    zigzag_points = result[result['zigzag'].notna()]
    hh = (result['swing_type'] == 'HH').sum()
    hl = (result['swing_type'] == 'HL').sum()
    lh = (result['swing_type'] == 'LH').sum()
    ll = (result['swing_type'] == 'LL').sum()
    
    return {
        'result': result,
        'total_points': len(zigzag_points),
        'HH': hh,
        'HL': hl,
        'LH': lh,
        'LL': ll
    }


def auto_tune_parameters(df_test: pd.DataFrame):
    """
    自動測試多種參數組合
    """
    print("\n" + "="*60)
    print("自動參數調整模式")
    print("="*60)
    
    test_configs = [
        {"depth": 12, "deviation": 0.5, "backstep": 3, "name": "極敏感"},
        {"depth": 12, "deviation": 0.8, "backstep": 3, "name": "敏感"},
        {"depth": 12, "deviation": 1.0, "backstep": 3, "name": "標準"},
        {"depth": 12, "deviation": 1.5, "backstep": 3, "name": "保守"},
        {"depth": 10, "deviation": 0.8, "backstep": 2, "name": "短周期"},
    ]
    
    results = []
    for i, config in enumerate(test_configs, 1):
        name = config.pop('name')
        print(f"\n[{i}/{len(test_configs)}] 測試 {name} 配置: {config}")
        print("-" * 60)
        
        result_data = test_single_config(df_test, verbose=True, **config)
        config['name'] = name
        
        results.append({
            'config': config,
            **result_data
        })
        
        print(f"\n結果: {result_data['total_points']:,} 個轉折點")
        print(f"Swing分佈: HH={result_data['HH']}, HL={result_data['HL']}, LH={result_data['LH']}, LL={result_data['LL']}")
    
    # 選擇最佳配置
    print(f"\n\n{'='*60}")
    print("參數分析結果")
    print("="*60)
    
    print(f"\n{'\u914d\u7f6e':<12} {'Depth':<8} {'Dev%':<8} {'Back':<6} {'\u8f49\u6298\u9ede':<10} {'HH':<5} {'HL':<5} {'LH':<5} {'LL':<5}")
    print("-" * 70)
    for r in results:
        c = r['config']
        print(f"{c['name']:<12} {c['depth']:<8} {c['deviation']:<8.1f} {c['backstep']:<6} {r['total_points']:<10,} {r['HH']:<5} {r['HL']:<5} {r['LH']:<5} {r['LL']:<5}")
    
    # 選擇轉折點數量在理想範圍的
    best = None
    for r in results:
        if 20 <= r['total_points'] <= 50:
            best = r
            break
    
    if best is None:
        best = min(results, key=lambda x: abs(x['total_points'] - 35))
    
    print(f"\n建議使用: {best['config']['name']} 配置")
    print(f"  Depth={best['config']['depth']}, Deviation={best['config']['deviation']}%, Backstep={best['config']['backstep']}")
    print(f"  轉折點數量: {best['total_points']:,}")
    
    return best


def main():
    """
    主測試函數
    """
    # 檢查依賴
    check_dependencies()
    
    # 解析命令列參數
    args = parse_arguments()
    
    # 顯示參數資訊
    print("="*60)
    print("ZigZag指標測試程式")
    print("="*60)
    
    # 下載BTC 15m數據
    url = "https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data/resolve/main/klines/BTCUSDT/BTC_15m.parquet"
    
    try:
        df = download_btc_data(url)
        
        # 確認欄位名稱(轉換為小寫)
        df.columns = df.columns.str.lower()
        
        # 檢查必要的欄位
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise Exception(f"資料缺少必要欄位: {missing_cols}")
        
        # 根據參數決定使用的數據量
        if args.all_data:
            df_test = df.copy()
            print(f"\n使用所有數據: {len(df_test):,} 條記錄")
        else:
            df_test = df.tail(args.samples).copy()
            print(f"\n使用最近 {args.samples:,} 條記錄")
        
        df_test = df_test.reset_index(drop=True)
        
        if 'timestamp' in df_test.columns:
            print(f"時間範圏: {df_test['timestamp'].min()} 至 {df_test['timestamp'].max()}")
        
        # 根據模式執行
        if args.auto_tune and not args.all_data:
            # 自動調整模式 (仅在非全量模式)
            best = auto_tune_parameters(df_test)
            result = best['result']
            config_info = f"Depth={best['config']['depth']}, Deviation={best['config']['deviation']}%, Backstep={best['config']['backstep']}"
        else:
            # 單一參數測試
            if args.all_data:
                print(f"\n警告: 使用所有數據(約21萬筆)進行計算,預計需要2-5分鐘")
            
            print(f"\n使用指定參數: Depth={args.depth}, Deviation={args.deviation}%, Backstep={args.backstep}")
            result_data = test_single_config(df_test, args.depth, args.deviation, args.backstep, verbose=True)
            result = result_data['result']
            config_info = f"Depth={args.depth}, Deviation={args.deviation}%, Backstep={args.backstep}"
            
            # 顯示統計
            print(f"\n\n{'='*60}")
            print("結果統計")
            print("="*60)
            print(f"轉折點總數: {result_data['total_points']:,}")
            print(f"HH (更高高點): {result_data['HH']:,}")
            print(f"HL (更高低點): {result_data['HL']:,}")
            print(f"LH (更低高點): {result_data['LH']:,}")
            print(f"LL (更低低點): {result_data['LL']:,}")
        
        # 顯示部分結果
        zigzag_points = result[result['zigzag'].notna()]
        print(f"\n\n{'='*60}")
        print("部分ZigZag點資訊 (前20個轉折點)")
        print("="*60)
        
        display_cols = ['open', 'high', 'low', 'close', 'zigzag', 'direction', 'swing_type']
        if 'timestamp' in zigzag_points.columns:
            display_cols.insert(0, 'timestamp')
        
        available_cols = [col for col in display_cols if col in zigzag_points.columns]
        print(zigzag_points[available_cols].head(20).to_string(index=False))
        
        # 儲存結果
        print(f"\n\n正在儲存結果...")
        result.to_csv(args.output, index=False)
        
        print(f"\n{'='*60}")
        print(f"結果已儲存至: {args.output}")
        print(f"使用參數: {config_info}")
        print(f"資料筆數: {len(result):,}")
        print(f"轉折點數: {len(zigzag_points):,}")
        print("="*60)
        
    except Exception as e:
        print(f"\n錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
