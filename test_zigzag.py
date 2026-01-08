import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import requests
from io import BytesIO

class ZigZagMT4:
    """
    MT4風格ZigZag指標實現
    根據Pine Script原始邏輯轉換而來
    
    參數:
        depth: 回溯深度,尋找局部極值的最小窗口
        deviation: 價格變動閾值(百分比),必須超過此值才確認新轉折點
        backstep: 連續極值點之間的最小K棒間隔
    """
    
    def __init__(self, depth: int = 12, deviation: float = 5.0, backstep: int = 2):
        self.depth = depth
        self.deviation = deviation / 100.0  # 轉換為小數
        self.backstep = backstep
        
    def find_pivots(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        尋找所有的pivot點(局部極值)
        
        Args:
            data: 必須包含 'high' 和 'low' 欄位的DataFrame
            
        Returns:
            包含pivot點資訊的DataFrame
        """
        df = data.copy()
        n = len(df)
        
        # 初始化陣列
        high_pivots = np.zeros(n, dtype=bool)
        low_pivots = np.zeros(n, dtype=bool)
        
        # 尋找局部高點和低點
        for i in range(self.depth, n - self.depth):
            # 檢查是否為局部高點
            is_high_pivot = True
            for j in range(i - self.depth, i + self.depth + 1):
                if j != i and df['high'].iloc[j] >= df['high'].iloc[i]:
                    is_high_pivot = False
                    break
            if is_high_pivot:
                high_pivots[i] = True
                
            # 檢查是否為局部低點
            is_low_pivot = True
            for j in range(i - self.depth, i + self.depth + 1):
                if j != i and df['low'].iloc[j] <= df['low'].iloc[i]:
                    is_low_pivot = False
                    break
            if is_low_pivot:
                low_pivots[i] = True
                
        df['high_pivot'] = high_pivots
        df['low_pivot'] = low_pivots
        
        return df
    
    def calculate_zigzag(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        計算ZigZag線和標記HH/HL/LL/LH
        
        Args:
            data: 包含 OHLC 數據的 DataFrame
            
        Returns:
            包含ZigZag標記的DataFrame
        """
        df = self.find_pivots(data)
        n = len(df)
        
        # 初始化結果陣列
        zigzag = np.full(n, np.nan)
        direction = np.zeros(n)  # 1=上升, -1=下降
        swing_type = [''] * n
        
        # 尋找第一個有效的pivot點
        last_pivot_idx = -1
        last_pivot_price = 0
        last_pivot_is_high = None
        last_confirmed_high = None
        last_confirmed_low = None
        
        for i in range(n):
            if df['high_pivot'].iloc[i]:
                if last_pivot_idx == -1:
                    # 第一個點
                    last_pivot_idx = i
                    last_pivot_price = df['high'].iloc[i]
                    last_pivot_is_high = True
                    zigzag[i] = last_pivot_price
                    direction[i] = -1  # 從高點開始,接下來是下降
                    last_confirmed_high = last_pivot_price
                elif last_pivot_is_high:
                    # 連續高點,檢查是否更高
                    if df['high'].iloc[i] > last_pivot_price:
                        # 更新為更高的高點
                        zigzag[last_pivot_idx] = np.nan
                        last_pivot_idx = i
                        last_pivot_price = df['high'].iloc[i]
                        zigzag[i] = last_pivot_price
                        direction[i] = direction[i-1]
                else:
                    # 從低點轉到高點,檢查deviation
                    price_change = abs((df['high'].iloc[i] - last_pivot_price) / last_pivot_price)
                    if price_change >= self.deviation and (i - last_pivot_idx) >= self.backstep:
                        # 確認新的高點
                        last_confirmed_low = last_pivot_price
                        last_pivot_idx = i
                        last_pivot_price = df['high'].iloc[i]
                        last_pivot_is_high = True
                        zigzag[i] = last_pivot_price
                        direction[i] = -1
                        
                        # 標記swing type
                        if last_confirmed_high is not None:
                            if last_pivot_price > last_confirmed_high:
                                swing_type[i] = 'HH'
                            else:
                                swing_type[i] = 'LH'
                        last_confirmed_high = last_pivot_price
                        
            elif df['low_pivot'].iloc[i]:
                if last_pivot_idx == -1:
                    # 第一個點
                    last_pivot_idx = i
                    last_pivot_price = df['low'].iloc[i]
                    last_pivot_is_high = False
                    zigzag[i] = last_pivot_price
                    direction[i] = 1  # 從低點開始,接下來是上升
                    last_confirmed_low = last_pivot_price
                elif not last_pivot_is_high:
                    # 連續低點,檢查是否更低
                    if df['low'].iloc[i] < last_pivot_price:
                        # 更新為更低的低點
                        zigzag[last_pivot_idx] = np.nan
                        last_pivot_idx = i
                        last_pivot_price = df['low'].iloc[i]
                        zigzag[i] = last_pivot_price
                        direction[i] = direction[i-1]
                else:
                    # 從高點轉到低點,檢查deviation
                    price_change = abs((df['low'].iloc[i] - last_pivot_price) / last_pivot_price)
                    if price_change >= self.deviation and (i - last_pivot_idx) >= self.backstep:
                        # 確認新的低點
                        last_confirmed_high = last_pivot_price
                        last_pivot_idx = i
                        last_pivot_price = df['low'].iloc[i]
                        last_pivot_is_high = False
                        zigzag[i] = last_pivot_price
                        direction[i] = 1
                        
                        # 標記swing type
                        if last_confirmed_low is not None:
                            if last_pivot_price < last_confirmed_low:
                                swing_type[i] = 'LL'
                            else:
                                swing_type[i] = 'HL'
                        last_confirmed_low = last_pivot_price
        
        # 填充direction
        for i in range(1, n):
            if direction[i] == 0:
                direction[i] = direction[i-1]
        
        df['zigzag'] = zigzag
        df['direction'] = direction
        df['swing_type'] = swing_type
        
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
    response = requests.get(url)
    if response.status_code == 200:
        df = pd.read_parquet(BytesIO(response.content))
        print(f"數據下載成功: {len(df)} 條記錄")
        return df
    else:
        raise Exception(f"下載失敗: HTTP {response.status_code}")


def main():
    """
    主測試函數
    """
    # 下載BTC 15m數據
    url = "https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data/resolve/main/klines/BTCUSDT/BTC_15m.parquet"
    
    try:
        df = download_btc_data(url)
        
        # 確認欄位名稱(轉換為小寫)
        df.columns = df.columns.str.lower()
        
        # 只取最近1000條數據進行測試
        df_test = df.tail(1000).copy()
        df_test = df_test.reset_index(drop=True)
        
        print(f"\n測試數據範圍: {len(df_test)} 條記錄")
        print(f"時間範圍: {df_test['timestamp'].min()} 至 {df_test['timestamp'].max()}")
        
        # 初始化ZigZag指標
        zigzag = ZigZagMT4(depth=12, deviation=5.0, backstep=2)
        
        # 計算ZigZag
        print("\n正在計算ZigZag...")
        result = zigzag.calculate_zigzag(df_test)
        
        # 統計結果
        zigzag_points = result[result['zigzag'].notna()]
        print(f"\n找到 {len(zigzag_points)} 個ZigZag轉折點")
        
        # 統計各類型swing
        swing_counts = result['swing_type'].value_counts()
        print("\nSwing Type分佈:")
        for swing_type, count in swing_counts.items():
            if swing_type != '':
                print(f"  {swing_type}: {count}")
        
        # 顯示部分結果
        print("\n部分ZigZag點資訊:")
        display_cols = ['timestamp', 'open', 'high', 'low', 'close', 'zigzag', 'direction', 'swing_type']
        print(zigzag_points[display_cols].head(20).to_string(index=False))
        
        # 儲存結果
        output_file = 'zigzag_result.csv'
        result.to_csv(output_file, index=False)
        print(f"\n結果已儲存至: {output_file}")
        
    except Exception as e:
        print(f"錯誤: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
