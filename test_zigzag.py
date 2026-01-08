import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import requests
from io import BytesIO
import sys

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
        depth: 回溯深度,尋找局部極值的最小窗口
        deviation: 價格變動閾值(百分比),必須超過此值才確認新轉折點
        backstep: 連續極值點之間的最小K棒間隔
    """
    
    def __init__(self, depth: int = 12, deviation: float = 5.0, backstep: int = 3):
        self.depth = depth
        self.deviation = deviation / 100.0  # 轉換為小數
        self.backstep = backstep
        print(f"\nZigZag參數: Depth={depth}, Deviation={deviation}%, Backstep={backstep}")
        
    def calculate_zigzag(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        計算ZigZag線和標記HH/HL/LL/LH
        使用MT4風格的算法
        
        Args:
            data: 包含 OHLC 數據的 DataFrame
            
        Returns:
            包含ZigZag標記的DataFrame
        """
        df = data.copy()
        n = len(df)
        
        # 初始化結果陣列
        zigzag_high = np.full(n, np.nan)
        zigzag_low = np.full(n, np.nan)
        zigzag = np.full(n, np.nan)
        direction = np.zeros(n)
        swing_type = [''] * n
        
        # 第一步: 尋找所有局部高點和低點
        print(f"步驟 1: 尋找局部極值 (depth={self.depth})...")
        for i in range(self.depth, n):
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
        print(f"  找到 {high_count} 個局部高點, {low_count} 個局部低點")
        
        # 第二步: 應用Deviation和Backstep篩選
        print(f"\n步驟 2: 應用Deviation={self.deviation*100}%和Backstep={self.backstep}篩選...")
        
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
        
        print(f"  確認了 {len(confirmed_highs)} 個高點, {len(confirmed_lows)} 個低點")
        
        # 第三步: 合併和排序所有轉折點
        print(f"\n步驟 3: 合併轉折點並標記HH/HL/LL/LH...")
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
        
        print(f"  完成! 總計 {len(all_pivots)} 個轉折點")
        
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
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        df = pd.read_parquet(BytesIO(response.content))
        print(f"數據下載成功: {len(df)} 條記錄")
        return df
    except requests.exceptions.RequestException as e:
        raise Exception(f"下載失敗: {str(e)}")
    except Exception as e:
        raise Exception(f"讀取Parquet檔案失敗: {str(e)}")


def test_different_parameters(df_test: pd.DataFrame):
    """
    測試不同的參數組合
    """
    print("\n" + "="*60)
    print("測試不同參數組合")
    print("="*60)
    
    # 根據搜索結果,對於15分鐘圖表,建議使用更小的deviation
    test_configs = [
        {"depth": 12, "deviation": 0.5, "backstep": 3},  # 極小 deviation
        {"depth": 12, "deviation": 1.0, "backstep": 3},  # 小 deviation
        {"depth": 12, "deviation": 2.0, "backstep": 3},  # 中等 deviation
        {"depth": 10, "deviation": 1.0, "backstep": 2},  # 短周期配置
    ]
    
    results = []
    for config in test_configs:
        print(f"\n{'='*60}")
        zigzag = ZigZagMT4(**config)
        result = zigzag.calculate_zigzag(df_test)
        
        zigzag_points = result[result['zigzag'].notna()]
        hh = (result['swing_type'] == 'HH').sum()
        hl = (result['swing_type'] == 'HL').sum()
        lh = (result['swing_type'] == 'LH').sum()
        ll = (result['swing_type'] == 'LL').sum()
        
        results.append({
            'config': config,
            'total_points': len(zigzag_points),
            'HH': hh,
            'HL': hl,
            'LH': lh,
            'LL': ll
        })
        
        print(f"\n結果: {len(zigzag_points)} 個轉折點 (HH:{hh}, HL:{hl}, LH:{lh}, LL:{ll})")
    
    return results


def main():
    """
    主測試函數
    """
    # 檢查依賴
    check_dependencies()
    
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
        
        # 只取最近1000條數據進行測試
        df_test = df.tail(1000).copy()
        df_test = df_test.reset_index(drop=True)
        
        print(f"\n測試數據範圍: {len(df_test)} 條記錄")
        if 'timestamp' in df_test.columns:
            print(f"時間範圏: {df_test['timestamp'].min()} 至 {df_test['timestamp'].max()}")
        
        # 測試不同參數
        results = test_different_parameters(df_test)
        
        # 選擇最佳參數(轉折點數量在20-50之間的)
        print(f"\n\n{'='*60}")
        print("參數選擇建議")
        print("="*60)
        
        best_config = None
        for r in results:
            if 20 <= r['total_points'] <= 50:
                best_config = r
                break
        
        if best_config is None:
            # 如果沒有理想的,選擇轉折點最接近35個的
            best_config = min(results, key=lambda x: abs(x['total_points'] - 35))
        
        print(f"\n建議使用配置: {best_config['config']}")
        print(f"轉折點數量: {best_config['total_points']}")
        print(f"Swing分佈: HH={best_config['HH']}, HL={best_config['HL']}, LH={best_config['LH']}, LL={best_config['LL']}")
        
        # 使用最佳參數重新計算並儲存
        print(f"\n\n{'='*60}")
        print("使用最佳參數生成最終結果")
        print("="*60)
        
        final_zigzag = ZigZagMT4(**best_config['config'])
        result = final_zigzag.calculate_zigzag(df_test)
        
        # 顯示部分結果
        zigzag_points = result[result['zigzag'].notna()]
        print(f"\n部分ZigZag點資訊 (前20個轉折點):")
        display_cols = ['open', 'high', 'low', 'close', 'zigzag', 'direction', 'swing_type']
        if 'timestamp' in zigzag_points.columns:
            display_cols.insert(0, 'timestamp')
        
        available_cols = [col for col in display_cols if col in zigzag_points.columns]
        print(zigzag_points[available_cols].head(20).to_string(index=False))
        
        # 儲存結果
        output_file = 'zigzag_result.csv'
        result.to_csv(output_file, index=False)
        print(f"\n結果已儲存至: {output_file}")
        
    except Exception as e:
        print(f"\n錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
