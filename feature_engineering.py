import pandas as pd
import numpy as np
import ta
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

class ZigZagFeatureEngineer:
    """
    ZigZag特徵工程類別
    用於提取技術指標和市場結構特徵來預測 HH/HL/LL/LH
    """
    
    def __init__(self, lookback_periods: List[int] = [5, 10, 20, 50]):
        """
        Args:
            lookback_periods: 不同的回溯期數用於計算特徵
        """
        self.lookback_periods = lookback_periods
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加價格相關特徵
        """
        # 價格變動率
        for period in self.lookback_periods:
            df[f'return_{period}'] = df['close'].pct_change(period)
            df[f'high_low_ratio_{period}'] = (df['high'] - df['low']) / df['close']
        
        # 價格位置 (相對於最高最低點)
        for period in self.lookback_periods:
            df[f'price_position_{period}'] = (
                (df['close'] - df['low'].rolling(period).min()) / 
                (df['high'].rolling(period).max() - df['low'].rolling(period).min())
            )
        
        # 振幅特徵
        df['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-10)
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        return df
    
    def add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加動量指標
        """
        # RSI
        for period in [6, 14, 21]:
            df[f'rsi_{period}'] = ta.momentum.RSIIndicator(
                close=df['close'], window=period
            ).rsi()
        
        # MACD
        macd = ta.trend.MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(
            high=df['high'], low=df['low'], close=df['close']
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # ROC (Rate of Change)
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = ta.momentum.ROCIndicator(
                close=df['close'], window=period
            ).roc()
        
        # Williams %R
        df['williams_r'] = ta.momentum.WilliamsRIndicator(
            high=df['high'], low=df['low'], close=df['close'], lbp=14
        ).williams_r()
        
        return df
    
    def add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加趨勢指標
        """
        # 移動平均線
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = ta.trend.SMAIndicator(
                close=df['close'], window=period
            ).sma_indicator()
            df[f'ema_{period}'] = ta.trend.EMAIndicator(
                close=df['close'], window=period
            ).ema_indicator()
            
            # 價格相對於MA的位置
            df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1
            df[f'price_to_ema_{period}'] = df['close'] / df[f'ema_{period}'] - 1
        
        # ADX (趨勢強度)
        adx = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'])
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        
        # Aroon
        aroon = ta.trend.AroonIndicator(close=df['close'])
        df['aroon_up'] = aroon.aroon_up()
        df['aroon_down'] = aroon.aroon_down()
        df['aroon_indicator'] = aroon.aroon_indicator()
        
        return df
    
    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加波動性指標
        """
        # Bollinger Bands
        for period in [10, 20]:
            bb = ta.volatility.BollingerBands(close=df['close'], window=period)
            df[f'bb_high_{period}'] = bb.bollinger_hband()
            df[f'bb_low_{period}'] = bb.bollinger_lband()
            df[f'bb_mid_{period}'] = bb.bollinger_mavg()
            df[f'bb_width_{period}'] = bb.bollinger_wband()
            df[f'bb_pband_{period}'] = bb.bollinger_pband()
        
        # ATR (Average True Range)
        for period in [7, 14, 21]:
            df[f'atr_{period}'] = ta.volatility.AverageTrueRange(
                high=df['high'], low=df['low'], close=df['close'], window=period
            ).average_true_range()
            df[f'atr_pct_{period}'] = df[f'atr_{period}'] / df['close']
        
        # Keltner Channel
        kc = ta.volatility.KeltnerChannel(high=df['high'], low=df['low'], close=df['close'])
        df['kc_high'] = kc.keltner_channel_hband()
        df['kc_low'] = kc.keltner_channel_lband()
        df['kc_mid'] = kc.keltner_channel_mband()
        
        return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加成交量指標 (如果有volume欄位)
        """
        if 'volume' not in df.columns:
            return df
        
        # Volume SMA
        for period in [5, 10, 20]:
            df[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']
        
        # OBV (On Balance Volume)
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(
            close=df['close'], volume=df['volume']
        ).on_balance_volume()
        
        # Volume Price Trend
        df['vpt'] = ta.volume.VolumePriceTrendIndicator(
            close=df['close'], volume=df['volume']
        ).volume_price_trend()
        
        # MFI (Money Flow Index)
        df['mfi'] = ta.volume.MFIIndicator(
            high=df['high'], low=df['low'], close=df['close'], volume=df['volume']
        ).money_flow_index()
        
        return df
    
    def add_zigzag_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加ZigZag結構特徵
        這些特徵基於最近的zigzag轉折點
        """
        # 計算距離最近轉折點的距離
        zigzag_points = df[df['zigzag'].notna()].copy()
        
        # 初始化特徵
        df['bars_since_pivot'] = 0
        df['price_from_pivot_pct'] = 0.0
        df['pivot_type'] = 0  # 1=high, -1=low
        df['last_pivot_price'] = np.nan
        
        if len(zigzag_points) > 0:
            last_pivot_idx = None
            last_pivot_price = None
            last_pivot_type = 0
            
            for idx in df.index:
                if idx in zigzag_points.index:
                    last_pivot_idx = idx
                    last_pivot_price = df.loc[idx, 'zigzag']
                    last_pivot_type = 1 if df.loc[idx, 'direction'] == -1 else -1
                
                if last_pivot_idx is not None:
                    df.loc[idx, 'bars_since_pivot'] = idx - last_pivot_idx
                    df.loc[idx, 'last_pivot_price'] = last_pivot_price
                    df.loc[idx, 'pivot_type'] = last_pivot_type
                    if last_pivot_price != 0:
                        df.loc[idx, 'price_from_pivot_pct'] = (
                            (df.loc[idx, 'close'] - last_pivot_price) / last_pivot_price
                        )
        
        # 計算最近N個轉折點的統計特徵
        for n in [3, 5, 10]:
            df[f'avg_pivot_distance_{n}'] = 0.0
            df[f'avg_pivot_change_{n}'] = 0.0
            
            if len(zigzag_points) >= n:
                for idx in df.index:
                    recent_pivots = zigzag_points[zigzag_points.index <= idx].tail(n)
                    if len(recent_pivots) >= 2:
                        # 平均轉折點間隔
                        pivot_indices = recent_pivots.index.tolist()
                        distances = np.diff(pivot_indices)
                        if len(distances) > 0:
                            df.loc[idx, f'avg_pivot_distance_{n}'] = np.mean(distances)
                        
                        # 平均轉折點價格變動
                        pivot_prices = recent_pivots['zigzag'].values
                        changes = np.abs(np.diff(pivot_prices) / pivot_prices[:-1])
                        if len(changes) > 0:
                            df.loc[idx, f'avg_pivot_change_{n}'] = np.mean(changes)
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加時間特徵 (如果有timestamp)
        """
        if 'timestamp' not in df.columns:
            return df
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        
        # 周期性編碼
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成所有特徵
        """
        print("正在生成特徵...")
        
        df = df.copy()
        
        print("  - 價格特徵")
        df = self.add_price_features(df)
        
        print("  - 動量指標")
        df = self.add_momentum_features(df)
        
        print("  - 趨勢指標")
        df = self.add_trend_features(df)
        
        print("  - 波動性指標")
        df = self.add_volatility_features(df)
        
        print("  - 成交量指標")
        df = self.add_volume_features(df)
        
        print("  - ZigZag結構特徵")
        df = self.add_zigzag_structure_features(df)
        
        print("  - 時間特徵")
        df = self.add_time_features(df)
        
        print(f"特徵生成完成! 總共 {len(df.columns)} 個欄位")
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame, target_col: str = 'swing_type') -> tuple:
        """
        準備訓練數據
        
        Returns:
            X: 特徵矩陣
            y: 目標變數
            feature_names: 特徵名稱列表
        """
        # 移除不需要的欄位
        exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                       'zigzag', 'direction', 'swing_type']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # 只保留有swing_type標籤的資料 (即zigzag轉折點)
        df_labeled = df[df[target_col].isin(['HH', 'HL', 'LL', 'LH'])].copy()
        
        print(f"訓練數據統計:")
        print(f"  總樣本數: {len(df_labeled)}")
        print(f"  特徵數量: {len(feature_cols)}")
        print(f"  標籤分佈:")
        print(df_labeled[target_col].value_counts())
        
        X = df_labeled[feature_cols].values
        y = df_labeled[target_col].values
        
        # 處理缺失值
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X, y, feature_cols


def main():
    """
    測試特徵工程
    """
    import sys
    
    print("="*60)
    print("特徵工程測試")
    print("="*60)
    
    try:
        # 讀取zigzag結果
        print("\n讀取zigzag_result.csv...")
        df = pd.read_csv('zigzag_result.csv')
        print(f"資料筆數: {len(df):,}")
        
        # 創建FeatureEngineer
        engineer = ZigZagFeatureEngineer()
        
        # 生成特徵
        df_features = engineer.create_all_features(df)
        
        # 儲存特徵數據
        output_file = 'zigzag_features.csv'
        print(f"\n儲存特徵數據至 {output_file}...")
        df_features.to_csv(output_file, index=False)
        
        # 準備訓練數據
        print("\n準備訓練數據...")
        X, y, feature_names = engineer.prepare_training_data(df_features)
        
        print(f"\n\u8a13練數據形狀: X={X.shape}, y={y.shape}")
        
        print("\n" + "="*60)
        print("特徵工程完成!")
        print("="*60)
        print(f"\n輸出檔案:")
        print(f"  - {output_file}")
        print(f"\n下一步: 執行 train_model.py 進行模型訓練")
        
    except FileNotFoundError:
        print("\n錯誤: 找不到 zigzag_result.csv")
        print("請先執行 test_zigzag.py 生成zigzag結果")
        sys.exit(1)
    except Exception as e:
        print(f"\n錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
