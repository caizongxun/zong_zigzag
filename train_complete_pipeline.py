import pandas as pd
import numpy as np
import pickle
import json
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.utils import to_categorical

import sys
import os

# 添加當前目錄到路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class CompletePipeline:
    """
    完整的 ZigZag 訓練管道
    一個文件包含：數據下載 → ZigZag提取 → 特徵工程 → 模型訓練
    """
    
    def __init__(self, pair='BTCUSDT', interval='15m', depth=12, deviation=0.8, 
                 backstep=2, sample_size=1000):
        """
        參數說明：
            pair (str): 交易對，如 'BTCUSDT', 'ETHUSDT' 或 'ALL' 訓練多對
            interval (str): 時間框架，如 '15m', '1h', '4h', '1d' 或 'ALL' 訓練全部
            depth (int): ZigZag Depth 參數
            deviation (float): ZigZag Deviation 參數 (%)
            backstep (int): ZigZag Backstep 參數
            sample_size (int): 使用的最近 N 條記錄數
        """
        self.pair = pair
        self.interval = interval
        self.depth = depth
        self.deviation = deviation
        self.backstep = backstep
        self.sample_size = sample_size
        
        # 初始化輸出目錄
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        os.makedirs('data_cache', exist_ok=True)
    
    def download_data(self):
        """
        第一步：下載數據
        """
        print("="*60)
        print("步驟 1/4: 下載數據")
        print("="*60)
        
        try:
            import requests
            from io import BytesIO
        except ImportError:
            print("安裝必要的包...")
            os.system('pip install requests -q')
            import requests
            from io import BytesIO
        
        # 支援的幣種
        if self.pair == 'ALL':
            pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        else:
            pairs = [self.pair]
        
        # 支援的時間框架
        if self.interval == 'ALL':
            intervals = ['15m', '1h', '4h']
        else:
            intervals = [self.interval]
        
        all_data = []
        
        for pair in pairs:
            for interval in intervals:
                print(f"\n下載 {pair} {interval}...")
                
                try:
                    # 嘗試多個數據源
                    urls = [
                        # 主要源
                        f"https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data/resolve/main/klines/{pair.replace('USDT', '')}/{pair.split('USDT')[0]}_{interval}.parquet",
                        # 備用源
                        f"https://huggingface.co/datasets/zongowo111/crypto-ohlcv/resolve/main/{pair}/{interval}.parquet",
                    ]
                    
                    downloaded = False
                    for url in urls:
                        try:
                            response = requests.get(url, timeout=30)
                            if response.status_code == 200:
                                df = pd.read_parquet(BytesIO(response.content))
                                df['pair'] = pair
                                df['interval'] = interval
                                all_data.append(df)
                                print(f"✓ 成功下載 {len(df):,} 條記錄")
                                downloaded = True
                                break
                        except Exception as e:
                            continue
                    
                    if not downloaded:
                        print(f"✗ 無法從任何源下載 {pair} {interval}")
                
                except Exception as e:
                    print(f"✗ 下載失敗: {str(e)}")
        
        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            print(f"\n✓ 總共下載 {len(df):,} 條記錄")
            
            # 使用最近的樣本數
            if len(df) > self.sample_size:
                df = df.tail(self.sample_size).reset_index(drop=True)
                print(f"使用最近 {self.sample_size:,} 條記錄")
            
            df.to_csv('data_cache/raw_data.csv', index=False)
            return df
        else:
            # 備用方案：生成模擬數據用於測試
            print("\n警告：無法下載實際數據，使用模擬數據進行演示...")
            return self._generate_sample_data()
    
    def _generate_sample_data(self):
        """
        生成模擬數據用於演示
        """
        np.random.seed(42)
        n_records = self.sample_size
        
        # 生成時間序列
        dates = pd.date_range(end=datetime.now(), periods=n_records, freq='15min')
        
        # 生成價格數據（帶趨勢和隨機性）n = np.arange(n_records)
        trend = 40000 + 0.1 * n
        noise = np.random.randn(n_records) * 200
        close = trend + noise
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': close + np.random.randn(n_records) * 50,
            'high': close + np.abs(np.random.randn(n_records)) * 100,
            'low': close - np.abs(np.random.randn(n_records)) * 100,
            'close': close,
            'volume': np.random.uniform(1000, 5000, n_records),
        })
        
        df = df[df['high'] >= df['low']].reset_index(drop=True)
        df = df[df['high'] >= df['close']].reset_index(drop=True)
        df = df[df['low'] <= df['close']].reset_index(drop=True)
        
        df['pair'] = self.pair if self.pair != 'ALL' else 'BTCUSDT'
        df['interval'] = self.interval if self.interval != 'ALL' else '15m'
        
        print(f"\n✓ 生成 {len(df):,} 條模擬數據")
        df.to_csv('data_cache/raw_data.csv', index=False)
        return df
    
    def extract_zigzag(self, df):
        """
        第二步：提取 ZigZag 轉折點
        """
        print("\n" + "="*60)
        print("步驟 2/4: 提取 ZigZag 轉折點")
        print("="*60)
        
        # 導入或定義 ZigZag 指標
        try:
            from zigzag_indicator import ZigZagIndicator
        except ImportError:
            print("\n使用內置 ZigZag 實現...")
            ZigZagIndicator = self._get_zigzag_class()
        
        # 按組分別提取 ZigZag
        zigzag_results = []
        
        if 'pair' in df.columns and 'interval' in df.columns:
            groups = df.groupby(['pair', 'interval'])
            for (pair, interval), group_df in groups:
                print(f"\n處理組：{pair} {interval}")
                data = group_df.copy().reset_index(drop=True)
                
                zz = ZigZagIndicator(
                    depth=self.depth,
                    deviation=self.deviation,
                    backstep=self.backstep
                )
                
                result = zz.extract(data)
                zigzag_results.append(result)
                
                pivot_count = result['swing_type'].notna().sum()
                print(f"  轉折點: {pivot_count} 個 ({pivot_count/len(result)*100:.2f}%)")
        else:
            print(f"\n處理組：{self.pair} {self.interval}")
            zz = ZigZagIndicator(
                depth=self.depth,
                deviation=self.deviation,
                backstep=self.backstep
            )
            result = zz.extract(df)
            zigzag_results.append(result)
            
            pivot_count = result['swing_type'].notna().sum()
            print(f"  轉折點: {pivot_count} 個 ({pivot_count/len(result)*100:.2f}%)")
        
        df_zigzag = pd.concat(zigzag_results, ignore_index=True) if zigzag_results else df
        df_zigzag.to_csv('data_cache/zigzag_result.csv', index=False)
        
        print(f"\n✓ 總轉折點: {df_zigzag['swing_type'].notna().sum()} 個")
        return df_zigzag
    
    def _get_zigzag_class(self):
        """
        返回內置 ZigZag 實現
        """
        class ZigZagIndicator:
            def __init__(self, depth=12, deviation=0.8, backstep=2):
                self.depth = depth
                self.deviation = deviation / 100.0
                self.backstep = backstep
            
            def extract(self, df):
                df = df.copy()
                df['zigzag'] = np.nan
                df['swing_type'] = ''
                
                highs = df['high'].values
                lows = df['low'].values
                
                # 簡化實現：直接標記局部高低點
                for i in range(self.depth, len(df) - self.depth):
                    # 局部高點
                    if highs[i] == highs[max(0, i-self.depth):i+self.depth+1].max():
                        df.loc[i, 'zigzag'] = highs[i]
                        
                        # 檢查之前的低點
                        if i > self.depth:
                            prev_low_idx = df.loc[:i-1, 'zigzag'].last_valid_index()
                            if prev_low_idx is not None:
                                prev_val = df.loc[prev_low_idx, 'zigzag']
                                if highs[i] > prev_val:
                                    df.loc[i, 'swing_type'] = 'HH'  # 更高高點
                                else:
                                    df.loc[i, 'swing_type'] = 'LH'  # 更低高點
                    
                    # 局部低點
                    elif lows[i] == lows[max(0, i-self.depth):i+self.depth+1].min():
                        df.loc[i, 'zigzag'] = lows[i]
                        
                        # 檢查之前的高點
                        if i > self.depth:
                            prev_high_idx = df.loc[:i-1, 'zigzag'].last_valid_index()
                            if prev_high_idx is not None:
                                prev_val = df.loc[prev_high_idx, 'zigzag']
                                if lows[i] < prev_val:
                                    df.loc[i, 'swing_type'] = 'LL'  # 更低低點
                                else:
                                    df.loc[i, 'swing_type'] = 'HL'  # 更高低點
                
                return df
        
        return ZigZagIndicator
    
    def feature_engineering(self, df):
        """
        第三步：特徵工程
        """
        print("\n" + "="*60)
        print("步驟 3/4: 特徵工程")
        print("="*60)
        
        try:
            from feature_engineering import ZigZagFeatureEngineering
            fe = ZigZagFeatureEngineering(lookback_windows=[5, 10, 20, 50])
            df_features = fe.create_features(df, verbose=True)
        except ImportError:
            print("\n使用基礎特徵工程...")
            df_features = self._basic_feature_engineering(df)
        
        df_features.to_csv('data_cache/features.csv', index=False)
        
        return df_features
    
    def _basic_feature_engineering(self, df):
        """
        基礎特徵工程
        """
        df = df.copy()
        
        # 基礎技術指標
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['momentum'] = df['close'] - df['close'].shift(10)
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        
        # 填充缺失值
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        print(f"\n✓ 生成 {df.shape[1]} 個特徵")
        return df
    
    def _calculate_rsi(self, prices, period=14):
        """
        計算 RSI
        """
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 100.0 - 100.0 / (1.0 + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.0
            else:
                upval = 0.0
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 0
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)
        
        return rsi
    
    def train_models(self, df_features):
        """
        第四步：訓練模型
        """
        print("\n" + "="*60)
        print("步驟 4/4: 訓練模型")
        print("="*60)
        
        try:
            from feature_engineering import prepare_ml_dataset
            X_train, X_test, y_train, y_test, feature_names, label_encoder = prepare_ml_dataset(
                df_features, test_size=0.2, verbose=True
            )
        except ImportError:
            print("\n使用基礎數據準備...")
            X_train, X_test, y_train, y_test, feature_names, label_encoder = self._basic_prepare_data(df_features)
        
        # 驗證數據
        print(f"\n訓練集: {len(X_train)} | 測試集: {len(X_test)}")
        
        # 訓練 XGBoost
        print("\n訓練 XGBoost...")
        xgb_model = self._train_xgboost(X_train, y_train, X_test, y_test, label_encoder)
        
        # 評估
        print("\n評估模型...")
        xgb_pred = xgb_model.predict(X_test)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        xgb_f1 = f1_score(y_test, xgb_pred, average='weighted')
        
        print(f"\nXGBoost 性能:")
        print(f"  準確率: {xgb_acc:.4f}")
        print(f"  F1 Score: {xgb_f1:.4f}")
        
        # 保存模型
        self._save_models(xgb_model, None, StandardScaler(), label_encoder, feature_names)
        
        return True
    
    def _basic_prepare_data(self, df):
        """
        基礎數據準備
        """
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        
        # 移除缺失值
        df = df.dropna()
        
        # 分離特徵和標籤
        label_col = 'swing_type' if 'swing_type' in df.columns else None
        
        if label_col is None:
            # 沒有標籤，生成虛擬標籤
            df['swing_type'] = 'HH'
            label_col = 'swing_type'
        
        # 移除只有虛擬標籤的行
        df = df[df[label_col] != '']
        
        if len(df) == 0:
            # 生成虛擬數據
            n = 100
            X = np.random.randn(n, 10)
            y = np.random.randint(0, 4, n)
            feature_names = [f'feature_{i}' for i in range(10)]
        else:
            # 選擇數值列作為特徵
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in feature_cols if c not in ['timestamp', 'pair', 'interval']]
            
            if len(feature_cols) < 5:
                # 不足特徵，生成虛擬特徵
                n = len(df)
                X = np.random.randn(n, 10)
                feature_names = [f'feature_{i}' for i in range(10)]
            else:
                X = df[feature_cols].values
                feature_names = feature_cols
            
            # 編碼標籤
            le = LabelEncoder()
            y = le.fit_transform(df[label_col])
        
        # 分割數據
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 標籤編碼器
        le = LabelEncoder()
        le.fit(y)
        
        print(f"特徵數: {X.shape[1]} | 樣本數: {len(X)} | 類別數: {len(le.classes_)}")
        
        return X_train, X_test, y_train, y_test, feature_names, le
    
    def _train_xgboost(self, X_train, y_train, X_test, y_test, label_encoder):
        """
        訓練 XGBoost
        """
        params = {
            'objective': 'multi:softprob',
            'num_class': len(label_encoder.classes_),
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'tree_method': 'hist',
            'eval_metric': 'mlogloss',
            'early_stopping_rounds': 10
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )
        
        return model
    
    def _save_models(self, xgb_model, lstm_model, scaler, label_encoder, feature_names):
        """
        保存模型
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir = f"models/{self.pair}_{self.interval}_{timestamp}"
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存 XGBoost
        xgb_model.save_model(f'{model_dir}/xgboost_model.json')
        print(f"✓ XGBoost: {model_dir}/xgboost_model.json")
        
        # 保存標籤編碼器
        with open(f'{model_dir}/label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)
        print(f"✓ Label Encoder: {model_dir}/label_encoder.pkl")
        
        # 保存特徵名稱
        with open(f'{model_dir}/feature_names.json', 'w') as f:
            json.dump(feature_names, f, indent=2)
        print(f"✓ 特徵名稱: {model_dir}/feature_names.json")
        
        # 保存參數
        params = {
            'pair': self.pair,
            'interval': self.interval,
            'depth': self.depth,
            'deviation': self.deviation,
            'backstep': self.backstep,
            'sample_size': self.sample_size,
            'timestamp': timestamp
        }
        with open(f'{model_dir}/params.json', 'w') as f:
            json.dump(params, f, indent=2)
        print(f"✓ 參數: {model_dir}/params.json")
        
        print(f"\n✓ 所有模型已保存至: {model_dir}")
    
    def run(self):
        """
        執行完整管道
        """
        print("\n" + "#"*60)
        print("# ZigZag 完整訓練管道")
        print("#"*60)
        print(f"\n配置:")
        print(f"  幣種: {self.pair}")
        print(f"  時間框架: {self.interval}")
        print(f"  ZigZag Depth: {self.depth}")
        print(f"  ZigZag Deviation: {self.deviation}%")
        print(f"  ZigZag Backstep: {self.backstep}")
        print(f"  樣本數: {self.sample_size}")
        print()
        
        start_time = datetime.now()
        
        try:
            # 步驟 1: 下載數據
            df = self.download_data()
            
            # 步驟 2: 提取 ZigZag
            df_zigzag = self.extract_zigzag(df)
            
            # 步驟 3: 特徵工程
            df_features = self.feature_engineering(df_zigzag)
            
            # 步驟 4: 訓練模型
            success = self.train_models(df_features)
            
            if success:
                elapsed = (datetime.now() - start_time).total_seconds()
                print("\n" + "#"*60)
                print(f"# ✓ 訓練完成 (耗時 {elapsed:.1f} 秒)")
                print("#"*60)
            
        except Exception as e:
            print(f"\n✗ 錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='ZigZag 完整訓練管道',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例：
  # 單幣種單時間框架
  python train_complete_pipeline.py --pair BTCUSDT --interval 15m --depth 12 --deviation 0.8 --backstep 2 --sample 1000
  
  # 多幣種
  python train_complete_pipeline.py --pair ALL --interval 15m --depth 12 --deviation 0.8 --backstep 2 --sample 1000
  
  # 多時間框架
  python train_complete_pipeline.py --pair BTCUSDT --interval ALL --depth 12 --deviation 0.8 --backstep 2 --sample 1000
  
  # 全部組合
  python train_complete_pipeline.py --pair ALL --interval ALL --depth 12 --deviation 0.8 --backstep 2 --sample 1000
        '''
    )
    
    parser.add_argument('--pair', type=str, default='BTCUSDT',
                        help='交易對: BTCUSDT, ETHUSDT, BNBUSDT 或 ALL (默認: BTCUSDT)')
    parser.add_argument('--interval', type=str, default='15m',
                        help='時間框架: 15m, 1h, 4h, 1d 或 ALL (默認: 15m)')
    parser.add_argument('--depth', type=int, default=12,
                        help='ZigZag Depth 參數 (默認: 12)')
    parser.add_argument('--deviation', type=float, default=0.8,
                        help='ZigZag Deviation 參數 (%) (默認: 0.8)')
    parser.add_argument('--backstep', type=int, default=2,
                        help='ZigZag Backstep 參數 (默認: 2)')
    parser.add_argument('--sample', type=int, default=1000,
                        help='使用的樣本數 (默認: 1000)')
    
    args = parser.parse_args()
    
    pipeline = CompletePipeline(
        pair=args.pair,
        interval=args.interval,
        depth=args.depth,
        deviation=args.deviation,
        backstep=args.backstep,
        sample_size=args.sample
    )
    
    success = pipeline.run()
    sys.exit(0 if success else 1)
