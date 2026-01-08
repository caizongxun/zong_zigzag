import pandas as pd
import numpy as np
import ta
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

class ZigZagFeatureEngineering:
    """
    為ZigZag Swing Type預測設計的特徵工程類別
    基於2025年最佳實踐:
    1. 技術指標特徵 (根據Reddit/Kaggle研究,TA指標能大幅降低誤差)
    2. 價格動量特徵 (捕捉市場結構)
    3. 滾動窗口統計特徵 (多時間框架)
    4. 歷史ZigZag模式特徵 (利用過往轉折點資訊)
    """
    
    def __init__(self, lookback_windows: List[int] = [5, 10, 20, 50]):
        """
        Args:
            lookback_windows: 不同時間窗口大小,用於捕捉多時間框架特徵
        """
        self.lookback_windows = lookback_windows
        
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加常用技術指標
        基於研究顯示,技術指標能大幅改善深度學習模型表現
        """
        df = df.copy()
        
        # 趨勢指標
        # SMA - Simple Moving Average
        for window in [7, 14, 21, 50, 100, 200]:
            df[f'sma_{window}'] = ta.trend.sma_indicator(df['close'], window=window)
        
        # EMA - Exponential Moving Average
        for window in [7, 14, 21, 50]:
            df[f'ema_{window}'] = ta.trend.ema_indicator(df['close'], window=window)
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # ADX - Average Directional Index (趨勢強度)
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        
        # 動量指標
        # RSI - Relative Strength Index
        for window in [6, 14, 21]:
            df[f'rsi_{window}'] = ta.momentum.rsi(df['close'], window=window)
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)
        
        # ROC - Rate of Change
        for window in [5, 10, 20]:
            df[f'roc_{window}'] = ta.momentum.roc(df['close'], window=window)
        
        # 波動性指標
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_mid'] = bollinger.bollinger_mavg()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_width'] = bollinger.bollinger_wband()
        df['bb_pband'] = bollinger.bollinger_pband()
        
        # ATR - Average True Range
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        
        # 成交量指標 (如果有volume欄位)
        if 'volume' in df.columns:
            # OBV - On Balance Volume
            df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            
            # Volume SMA
            df['volume_sma_20'] = ta.trend.sma_indicator(df['volume'], window=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            
            # MFI - Money Flow Index
            df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=14)
        
        return df
    
    def add_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加價格行為特徵
        這些特徵直接反映市場結構
        """
        df = df.copy()
        
        # 價格變動
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['close'].diff()
        
        # 對數報酬
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # K線實體大小
        df['body'] = df['close'] - df['open']
        df['body_pct'] = (df['close'] - df['open']) / df['open']
        
        # 上下影線
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # 上下影線比例
        df['upper_shadow_pct'] = df['upper_shadow'] / (df['high'] - df['low'] + 1e-10)
        df['lower_shadow_pct'] = df['lower_shadow'] / (df['high'] - df['low'] + 1e-10)
        
        # 高低點範圍
        df['high_low_range'] = df['high'] - df['low']
        df['high_low_range_pct'] = (df['high'] - df['low']) / df['close']
        
        # 相對於前一根K棒的位置
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_pct'] = df['gap'] / df['close'].shift(1)
        
        return df
    
    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加滾動窗口統計特徵
        捕捉不同時間框架的模式
        """
        df = df.copy()
        
        for window in self.lookback_windows:
            # 價格統計
            df[f'close_mean_{window}'] = df['close'].rolling(window=window).mean()
            df[f'close_std_{window}'] = df['close'].rolling(window=window).std()
            df[f'close_min_{window}'] = df['close'].rolling(window=window).min()
            df[f'close_max_{window}'] = df['close'].rolling(window=window).max()
            
            # 價格相對位置
            df[f'close_position_{window}'] = (df['close'] - df[f'close_min_{window}']) / \
                                              (df[f'close_max_{window}'] - df[f'close_min_{window}'] + 1e-10)
            
            # 報酬統計
            df[f'return_mean_{window}'] = df['price_change'].rolling(window=window).mean()
            df[f'return_std_{window}'] = df['price_change'].rolling(window=window).std()
            df[f'return_skew_{window}'] = df['price_change'].rolling(window=window).skew()
            df[f'return_kurt_{window}'] = df['price_change'].rolling(window=window).kurt()
            
            # 波動率
            df[f'volatility_{window}'] = df['log_return'].rolling(window=window).std() * np.sqrt(252 * 24 * 4)  # 15m年化
        
        return df
    
    def add_zigzag_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加歷史ZigZag轉折點特徵
        利用過往的HH/HL/LL/LH模式
        
        注意: 這裡不能使用ffill填充swing_type,否則會導致數據洩漏
        """
        df = df.copy()
        
        # 距離上一個轉折點的K棒數
        df['bars_since_pivot'] = 0
        pivot_mask = df['zigzag'].notna()
        pivot_indices = df[pivot_mask].index
        
        for i in range(len(df)):
            if i in pivot_indices:
                df.loc[i, 'bars_since_pivot'] = 0
            elif i > 0:
                df.loc[i, 'bars_since_pivot'] = df.loc[i-1, 'bars_since_pivot'] + 1
        
        # 編碼swing type (不使用ffill)
        swing_type_map = {'': 0, 'HH': 1, 'HL': 2, 'LH': 3, 'LL': 4}
        df['swing_type_encoded'] = df['swing_type'].map(swing_type_map).fillna(0)
        
        # 當前價格距離上一個轉折點的距離
        df['distance_from_last_pivot'] = 0.0
        last_pivot_price = df['close'].iloc[0]
        
        for i in range(len(df)):
            if not pd.isna(df['zigzag'].iloc[i]):
                last_pivot_price = df['zigzag'].iloc[i]
                df.loc[df.index[i], 'distance_from_last_pivot'] = 0.0
            else:
                df.loc[df.index[i], 'distance_from_last_pivot'] = \
                    (df['close'].iloc[i] - last_pivot_price) / last_pivot_price
        
        # 最近N個轉折點的統計 (只統計真正的轉折點)
        pivot_df = df[df['zigzag'].notna()].copy()
        if len(pivot_df) > 0:
            for n in [2, 3, 5]:
                for swing in ['HH', 'HL', 'LH', 'LL']:
                    # 計算最近N個轉折點中該類型的比例
                    rolling_count = pivot_df['swing_type'].rolling(window=n, min_periods=1).apply(
                        lambda x: (x == swing).sum() / len(x)
                    )
                    df.loc[pivot_df.index, f'recent_{n}_{swing.lower()}_ratio'] = rolling_count
        
        # 填充非轉折點的統計值為0
        for n in [2, 3, 5]:
            for swing in ['HH', 'HL', 'LH', 'LL']:
                col = f'recent_{n}_{swing.lower()}_ratio'
                if col in df.columns:
                    df[col] = df[col].fillna(0)
        
        return df
    
    def create_features(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        創建所有特徵
        
        Args:
            df: 包含OHLCV和ZigZag結果的DataFrame
            verbose: 是否顯示進度
            
        Returns:
            包含所有特徵的DataFrame
        """
        if verbose:
            print("\n" + "="*60)
            print("開始特徵工程")
            print("="*60)
        
        # 確認必要欄位
        required_cols = ['open', 'high', 'low', 'close', 'zigzag', 'swing_type']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"缺少必要欄位: {missing}")
        
        df = df.copy()
        original_len = len(df)
        
        if verbose:
            print(f"原始資料筆數: {original_len:,}")
        
        # 1. 技術指標
        if verbose:
            print("\n[1/5] 計算技術指標...")
        df = self.add_technical_indicators(df)
        
        # 2. 價格行為特徵
        if verbose:
            print("[2/5] 添加價格行為特徵...")
        df = self.add_price_action_features(df)
        
        # 3. 滾動窗口特徵
        if verbose:
            print("[3/5] 計算滾動窗口特徵...")
        df = self.add_rolling_features(df)
        
        # 4. ZigZag歷史特徵
        if verbose:
            print("[4/5] 添加ZigZag歷史特徵...")
        df = self.add_zigzag_history_features(df)
        
        # 5. 移除缺失值太多的行
        if verbose:
            print("[5/5] 清理缺失值...")
        
        # 移除最前面的行 (因為滾動計算會產生 NaN)
        max_window = max(self.lookback_windows + [200])  # 200是最大的SMA窗口
        df = df.iloc[max_window:].copy()
        
        # 填充剩餘的NaN
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        if verbose:
            print(f"\n清理後資料筆數: {len(df):,}")
            print(f"移除了 {original_len - len(df):,} 筆 ({(original_len - len(df))/original_len*100:.1f}%)")
            print(f"總特徵數: {len(df.columns):,}")
            print("\n" + "="*60)
            print("特徵工程完成")
            print("="*60)
        
        return df


def prepare_ml_dataset(df: pd.DataFrame, target_col: str = 'swing_type', 
                       test_size: float = 0.2, verbose: bool = True) -> Tuple:
    """
    準備機器學習資料集
    
    Args:
        df: 包含特徵的DataFrame
        target_col: 目標變數欄位
        test_size: 測試集比例
        verbose: 是否顯示詳細資訊
        
    Returns:
        X_train, X_test, y_train, y_test, feature_names, label_encoder
    """
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    
    if verbose:
        print("\n" + "="*60)
        print("準備訓練資料集")
        print("="*60)
    
    # 只保留有swing type的資料 (即ZigZag轉折點)
    df_pivots = df[df[target_col].notna() & (df[target_col] != '')].copy()
    
    if verbose:
        print(f"\n原始資料: {len(df):,} 筆")
        print(f"轉折點資料: {len(df_pivots):,} 筆")
        print(f"\nSwing Type分佈:")
        print(df_pivots[target_col].value_counts())
    
    # 分離特徵和標籤 - 排除所有非數值和非特徵欄位
    exclude_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 
                    'zigzag', 'direction', 'swing_type', 'last_swing_type', 'swing_type_encoded']
    
    # 只保留數值類型的欄位
    numeric_cols = df_pivots.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    X = df_pivots[feature_cols].values
    y = df_pivots[target_col].values
    
    # 編碼swing type
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 時間序列切割 (不隨機打亂,保持時間順序)
    split_idx = int(len(X) * (1 - test_size))
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y_encoded[:split_idx]
    y_test = y_encoded[split_idx:]
    
    if verbose:
        print(f"\n特徵數量: {len(feature_cols):,}")
        print(f"訓練集: {len(X_train):,} 筆 ({len(X_train)/len(X)*100:.1f}%)")
        print(f"測試集: {len(X_test):,} 筆 ({len(X_test)/len(X)*100:.1f}%)")
        print(f"\n類別標籤:")
        for i, label in enumerate(le.classes_):
            print(f"  {i}: {label}")
        print("\n" + "="*60)
    
    return X_train, X_test, y_train, y_test, feature_cols, le


if __name__ == "__main__":
    # 測試範例
    import sys
    
    print("="*60)
    print("特徵工程模組測試")
    print("="*60)
    
    try:
        # 讀取ZigZag結果
        print("\n讀取zigzag_result.csv...")
        df = pd.read_csv('zigzag_result.csv')
        print(f"資料筆數: {len(df):,}")
        
        # 創建特徵
        fe = ZigZagFeatureEngineering()
        df_features = fe.create_features(df, verbose=True)
        
        # 準備資料集
        X_train, X_test, y_train, y_test, features, le = prepare_ml_dataset(
            df_features, verbose=True
        )
        
        print("\n特徵工程測試成功!")
        
    except FileNotFoundError:
        print("\n錯誤: 找不到 zigzag_result.csv")
        print("請先執行 test_zigzag.py 生成結果檔案")
        sys.exit(1)
    except Exception as e:
        print(f"\n錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
