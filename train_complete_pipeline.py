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
        
        # 支持的幣種
        if self.pair == 'ALL':
            pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        else:
            pairs = [self.pair]
        
        # 支持的時間框架
        if self.interval == 'ALL':
            intervals = ['15m', '1h', '4h']
        else:
            intervals = [self.interval]
        
        all_data = []
        
        for pair in pairs:
            for interval in intervals:
                print(f"\n下載 {pair} {interval}...")
                
                try:
                    url = f"https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data/resolve/main/klines/{pair.replace('USDT', '')}/{pair.split('USDT')[0]}_{interval}.parquet"
                    response = requests.get(url, timeout=30)
                    
                    if response.status_code == 200:
                        df = pd.read_parquet(BytesIO(response.content))
                        df['pair'] = pair
                        df['interval'] = interval
                        all_data.append(df)
                        print(f"✓ 成功下載 {len(df):,} 條記錄")
                    else:
                        print(f"✗ 下載失敗 (狀態碼 {response.status_code})")
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
            raise Exception("沒有成功下載任何數據")
    
    def extract_zigzag(self, df):
        """
        第二步：提取 ZigZag 轉折點
        """
        print("\n" + "="*60)
        print("步驟 2/4: 提取 ZigZag 轉折點")
        print("="*60)
        
        from zigzag_indicator import ZigZagIndicator
        
        # 按組分別提取 ZigZag
        zigzag_results = []
        
        if 'pair' in df.columns and 'interval' in df.columns:
            groups = df.groupby(['pair', 'interval'])
        else:
            groups = [('all', df)]
        
        for group_key, group_df in groups if isinstance(groups, pd.core.groupby.GroupBy) else groups:
            if isinstance(groups, pd.core.groupby.GroupBy):
                print(f"\n處理組：{group_key[0]} {group_key[1]}")
                data = group_df.copy().reset_index(drop=True)
            else:
                data = group_df
            
            zz = ZigZagIndicator(
                depth=self.depth,
                deviation=self.deviation,
                backstep=self.backstep
            )
            
            result = zz.extract(data)
            zigzag_results.append(result)
            
            pivot_count = result['swing_type'].notna().sum()
            print(f"  轉折點: {pivot_count} 個 ({pivot_count/len(result)*100:.2f}%)")
        
        df_zigzag = pd.concat(zigzag_results, ignore_index=True)
        df_zigzag.to_csv('data_cache/zigzag_result.csv', index=False)
        
        print(f"\n✓ 總轉折點: {df_zigzag['swing_type'].notna().sum()} 個")
        return df_zigzag
    
    def feature_engineering(self, df):
        """
        第三步：特徵工程
        """
        print("\n" + "="*60)
        print("步驟 3/4: 特徵工程")
        print("="*60)
        
        from feature_engineering import ZigZagFeatureEngineering
        
        fe = ZigZagFeatureEngineering(lookback_windows=[5, 10, 20, 50])
        df_features = fe.create_features(df, verbose=True)
        
        df_features.to_csv('data_cache/features.csv', index=False)
        
        return df_features
    
    def train_models(self, df_features):
        """
        第四步：訓練模型
        """
        print("\n" + "="*60)
        print("步驟 4/4: 訓練模型")
        print("="*60)
        
        from feature_engineering import prepare_ml_dataset
        
        # 準備數據
        print("\n[1/5] 準備訓練數據...")
        X_train, X_test, y_train, y_test, feature_names, label_encoder = prepare_ml_dataset(
            df_features, test_size=0.2, verbose=True
        )
        
        # 驗證數據完整性
        print("\n[2/5] 驗證數據完整性...")
        pivot_mask = df_features['swing_type'].notna() & (df_features['swing_type'] != '')
        pivot_ratio = (pivot_mask.sum() / len(df_features) * 100)
        print(f"轉折點比例: {pivot_ratio:.3f}%")
        
        if pivot_ratio > 5:
            print("⚠ 警告: 轉折點比例過高，可能存在數據洩漏")
            return False
        
        # 訓練 XGBoost
        print("\n[3/5] 訓練 XGBoost 模型...")
        xgb_model = self._train_xgboost(X_train, y_train, X_test, y_test, label_encoder)
        
        # 訓練 LSTM
        print("\n[4/5] 訓練 LSTM 模型...")
        lstm_model, scaler = self._train_lstm(X_train, y_train, X_test, y_test, label_encoder)
        
        # 評估結果
        print("\n[5/5] 評估模型...")
        self._evaluate_models(
            xgb_model, lstm_model, scaler, label_encoder,
            X_train, y_train, X_test, y_test
        )
        
        # 保存所有模型
        print("\n保存模型...")
        self._save_models(
            xgb_model, lstm_model, scaler, label_encoder, feature_names
        )
        
        return True
    
    def _train_xgboost(self, X_train, y_train, X_test, y_test, label_encoder):
        """
        訓練 XGBoost
        """
        params = {
            'objective': 'multi:softprob',
            'num_class': len(label_encoder.classes_),
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'tree_method': 'hist',
            'eval_metric': 'mlogloss',
            'early_stopping_rounds': 50
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=50
        )
        
        return model
    
    def _train_lstm(self, X_train, y_train, X_test, y_test, label_encoder):
        """
        訓練 LSTM
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 動態設定序列長度
        sequence_length = max(5, int(len(X_train) / 3))
        print(f"序列長度: {sequence_length}")
        
        # 準備序列
        def prepare_sequences(X, y, seq_len):
            if len(X) < seq_len:
                seq_len = max(1, len(X) - 1)
            n_samples = len(X) - seq_len + 1
            if n_samples <= 0:
                return None, None
            
            X_seq = np.zeros((n_samples, seq_len, X.shape[1]))
            y_seq = np.zeros(n_samples, dtype=int)
            
            for i in range(n_samples):
                X_seq[i] = X[i:i+seq_len]
                y_seq[i] = y[i+seq_len-1]
            
            return X_seq, y_seq
        
        X_train_seq, y_train_seq = prepare_sequences(X_train_scaled, y_train, sequence_length)
        X_test_seq, y_test_seq = prepare_sequences(X_test_scaled, y_test, sequence_length)
        
        if X_train_seq is None or X_test_seq is None:
            print("樣本太少，跳過 LSTM 訓練")
            return None, scaler
        
        y_train_cat = to_categorical(y_train_seq, num_classes=len(label_encoder.classes_))
        y_test_cat = to_categorical(y_test_seq, num_classes=len(label_encoder.classes_))
        
        # 構建模型
        model = keras.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=(sequence_length, X_train.shape[1])),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(len(label_encoder.classes_), activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 訓練
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        model.fit(
            X_train_seq, y_train_cat,
            validation_data=(X_test_seq, y_test_cat),
            epochs=100,
            batch_size=max(1, len(X_train_seq) // 2),
            callbacks=[early_stop],
            verbose=1
        )
        
        return model, scaler
    
    def _evaluate_models(self, xgb_model, lstm_model, scaler, label_encoder,
                         X_train, y_train, X_test, y_test):
        """
        評估模型
        """
        print("\n" + "="*60)
        print("模型評估結果")
        print("="*60)
        
        # XGBoost
        xgb_pred = xgb_model.predict(X_test)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        xgb_f1 = f1_score(y_test, xgb_pred, average='weighted')
        
        print(f"\nXGBoost:")
        print(f"  準確率: {xgb_acc:.4f}")
        print(f"  F1 Score: {xgb_f1:.4f}")
        
        # LSTM
        if lstm_model is not None:
            X_test_scaled = scaler.transform(X_test)
            sequence_length = max(5, int(len(X_train) / 3))
            
            def prepare_sequences(X, seq_len):
                if len(X) < seq_len:
                    seq_len = max(1, len(X) - 1)
                n_samples = len(X) - seq_len + 1
                if n_samples <= 0:
                    return None
                X_seq = np.zeros((n_samples, seq_len, X.shape[1]))
                for i in range(n_samples):
                    X_seq[i] = X[i:i+seq_len]
                return X_seq
            
            X_test_seq = prepare_sequences(X_test_scaled, sequence_length)
            
            if X_test_seq is not None:
                lstm_pred_proba = lstm_model.predict(X_test_seq, verbose=0)
                lstm_pred = np.argmax(lstm_pred_proba, axis=1)
                y_test_adj = y_test[-len(lstm_pred):]
                lstm_acc = accuracy_score(y_test_adj, lstm_pred)
                lstm_f1 = f1_score(y_test_adj, lstm_pred, average='weighted')
                
                print(f"\nLSTM:")
                print(f"  準確率: {lstm_acc:.4f}")
                print(f"  F1 Score: {lstm_f1:.4f}")
        
        # 詳細報告
        print("\n" + "-"*60)
        print("分類報告:")
        print(classification_report(
            y_test, xgb_pred,
            target_names=label_encoder.classes_,
            zero_division=0
        ))
    
    def _save_models(self, xgb_model, lstm_model, scaler, label_encoder, feature_names):
        """
        保存所有模型
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir = f"models/{self.pair}_{self.interval}_{timestamp}"
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存 XGBoost
        xgb_model.save_model(f'{model_dir}/xgboost_model.json')
        print(f"✓ XGBoost: {model_dir}/xgboost_model.json")
        
        # 保存 LSTM
        if lstm_model is not None:
            lstm_model.save(f'{model_dir}/lstm_model.h5')
            print(f"✓ LSTM: {model_dir}/lstm_model.h5")
        
        # 保存輔助文件
        with open(f'{model_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✓ Scaler: {model_dir}/scaler.pkl")
        
        with open(f'{model_dir}/label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)
        print(f"✓ Label Encoder: {model_dir}/label_encoder.pkl")
        
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
        
        print(f"\n所有模型已保存至: {model_dir}")
    
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
