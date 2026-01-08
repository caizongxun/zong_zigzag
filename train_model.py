import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.utils import to_categorical

from feature_engineering import ZigZagFeatureEngineering, prepare_ml_dataset

class ZigZagHybridModel:
    """
    混合模型: XGBoost + LSTM
    基於2025年研究最佳實踐:
    1. XGBoost處理表格化特徵 (技術指標, 統計特徵)
    2. LSTM捕捉時間序列模式 (價格動量)
    3. 集成兩個模型的預測
    """
    
    def __init__(self, n_classes: int = 4, sequence_length: int = 30):
        """
        Args:
            n_classes: 類別數量 (HH, HL, LH, LL)
            sequence_length: LSTM輸入序列長度
        """
        self.n_classes = n_classes
        self.sequence_length = sequence_length
        self.xgb_model = None
        self.lstm_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def build_xgboost_model(self, X_train, y_train, X_test, y_test):
        """
        訓練XGBoost模型
        根據研究,XGBoost在金融時間序列分類上表現優異
        """
        print("\n" + "="*60)
        print("訓練XGBoost模型")
        print("="*60)
        
        # XGBoost參數 (針對多類別分類優化)
        params = {
            'objective': 'multi:softprob',
            'num_class': self.n_classes,
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
        
        # 創建模型
        self.xgb_model = xgb.XGBClassifier(**params)
        
        # 訓練
        print("\n正在訓練...")
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=50
        )
        
        # 評估
        train_pred = self.xgb_model.predict(X_train)
        test_pred = self.xgb_model.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        train_f1 = f1_score(y_train, train_pred, average='weighted')
        test_f1 = f1_score(y_test, test_pred, average='weighted')
        
        print(f"\n訓練集準確率: {train_acc:.4f}")
        print(f"測試集準確率: {test_acc:.4f}")
        print(f"訓練集F1: {train_f1:.4f}")
        print(f"測試集F1: {test_f1:.4f}")
        
        return test_pred
    
    def build_lstm_model(self, input_shape):
        """
        建立LSTM模型
        基於2025研究,LSTM適合捕捉時間序列的長期依賴
        """
        model = keras.Sequential([
            # 第一層LSTM
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            
            # 第二層LSTM
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            
            # 全連接層
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            
            # 輸出層
            layers.Dense(self.n_classes, activation='softmax')
        ])
        
        # 編譯
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_lstm_sequences(self, X, y):
        """
        準備LSTM序列資料
        """
        n_samples = len(X) - self.sequence_length + 1
        X_seq = np.zeros((n_samples, self.sequence_length, X.shape[1]))
        y_seq = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            X_seq[i] = X[i:i+self.sequence_length]
            y_seq[i] = y[i+self.sequence_length-1]
        
        return X_seq, y_seq
    
    def train_lstm_model(self, X_train, y_train, X_test, y_test):
        """
        訓練LSTM模型
        """
        print("\n" + "="*60)
        print("訓練LSTM模型")
        print("="*60)
        
        # 標準化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 準備序列
        print(f"\n準備序列資料 (序列長度={self.sequence_length})...")
        X_train_seq, y_train_seq = self.prepare_lstm_sequences(X_train_scaled, y_train)
        X_test_seq, y_test_seq = self.prepare_lstm_sequences(X_test_scaled, y_test)
        
        # One-hot encoding
        y_train_cat = to_categorical(y_train_seq, num_classes=self.n_classes)
        y_test_cat = to_categorical(y_test_seq, num_classes=self.n_classes)
        
        print(f"訓練集形狀: {X_train_seq.shape}")
        print(f"測試集形狀: {X_test_seq.shape}")
        
        # 建立模型
        self.lstm_model = self.build_lstm_model(
            input_shape=(self.sequence_length, X_train.shape[1])
        )
        
        print(f"\n模型結構:")
        self.lstm_model.summary()
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )
        
        # 訓練
        print("\n正在訓練...")
        history = self.lstm_model.fit(
            X_train_seq, y_train_cat,
            validation_data=(X_test_seq, y_test_cat),
            epochs=100,
            batch_size=32,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        # 評估
        train_loss, train_acc = self.lstm_model.evaluate(X_train_seq, y_train_cat, verbose=0)
        test_loss, test_acc = self.lstm_model.evaluate(X_test_seq, y_test_cat, verbose=0)
        
        print(f"\n訓練集準確率: {train_acc:.4f}")
        print(f"測試集準確率: {test_acc:.4f}")
        
        #預測
        test_pred_proba = self.lstm_model.predict(X_test_seq, verbose=0)
        test_pred = np.argmax(test_pred_proba, axis=1)
        
        return test_pred, history
    
    def ensemble_predict(self, X, y=None):
        """
        集成預測: 結合XGBoost和LSTM的預測
        """
        # XGBoost預測機率
        xgb_proba = self.xgb_model.predict_proba(X)
        
        # LSTM預測機率
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self.prepare_lstm_sequences(X_scaled, np.zeros(len(X)))
        lstm_proba = self.lstm_model.predict(X_seq, verbose=0)
        
        # 加權平均 (XGBoost比重0.6, LSTM比重0.4)
        # 根據研究,XGBoost在金融資料上通常較穩定
        ensemble_proba = 0.6 * xgb_proba[self.sequence_length-1:] + 0.4 * lstm_proba
        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        
        return ensemble_pred, ensemble_proba


def main():
    """
    主訓練流程
    """
    print("="*60)
    print("ZigZag Swing Type 預測模型訓練")
    print("="*60)
    print(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 讀取資料
    print("\n[1/6] 讀取ZigZag結果...")
    df = pd.read_csv('zigzag_result.csv')
    print(f"資料筆數: {len(df):,}")
    
    # 2. 特徵工程
    print("\n[2/6] 特徵工程...")
    fe = ZigZagFeatureEngineering(lookback_windows=[5, 10, 20, 50])
    df_features = fe.create_features(df, verbose=True)
    
    # 3. 準備訓練資料
    print("\n[3/6] 準備訓練資料...")
    X_train, X_test, y_train, y_test, feature_names, label_encoder = prepare_ml_dataset(
        df_features, test_size=0.2, verbose=True
    )
    
    # 4. 訓練模型
    print("\n[4/6] 訓練模型...")
    model = ZigZagHybridModel(n_classes=len(label_encoder.classes_), sequence_length=30)
    model.feature_names = feature_names
    
    # 4a. XGBoost
    xgb_pred = model.build_xgboost_model(X_train, y_train, X_test, y_test)
    
    # 4b. LSTM
    lstm_pred, history = model.train_lstm_model(X_train, y_train, X_test, y_test)
    
    # 4c. 集成模型
    print("\n" + "="*60)
    print("集成模型預測")
    print("="*60)
    ensemble_pred, ensemble_proba = model.ensemble_predict(X_test, y_test)
    
    # 調整測試集長度 (因為LSTM需要序列)
    y_test_adj = y_test[model.sequence_length-1:]
    
    # 5. 評估
    print("\n[5/6] 模型評估...")
    
    print("\n" + "="*60)
    print("最終結果比較")
    print("="*60)
    
    # XGBoost結果
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_f1 = f1_score(y_test, xgb_pred, average='weighted')
    print(f"\nXGBoost:")
    print(f"  準確率: {xgb_acc:.4f}")
    print(f"  F1 Score: {xgb_f1:.4f}")
    
    # LSTM結果
    lstm_acc = accuracy_score(y_test_adj, lstm_pred)
    lstm_f1 = f1_score(y_test_adj, lstm_pred, average='weighted')
    print(f"\nLSTM:")
    print(f"  準確率: {lstm_acc:.4f}")
    print(f"  F1 Score: {lstm_f1:.4f}")
    
    # 集成結果
    ensemble_acc = accuracy_score(y_test_adj, ensemble_pred)
    ensemble_f1 = f1_score(y_test_adj, ensemble_pred, average='weighted')
    print(f"\n集成模型:")
    print(f"  準確率: {ensemble_acc:.4f}")
    print(f"  F1 Score: {ensemble_f1:.4f}")
    
    # 詳細分類報告
    print("\n" + "="*60)
    print("集成模型詳細報告")
    print("="*60)
    print("\n分類報告:")
    print(classification_report(
        y_test_adj, ensemble_pred, 
        target_names=label_encoder.classes_
    ))
    
    print("\n混淆矩陣:")
    cm = confusion_matrix(y_test_adj, ensemble_pred)
    cm_df = pd.DataFrame(
        cm, 
        index=label_encoder.classes_, 
        columns=label_encoder.classes_
    )
    print(cm_df)
    
    # 6. 儲存模型
    print("\n[6/6] 儲存模型...")
    
    # 儲存XGBoost
    model.xgb_model.save_model('models/xgboost_model.json')
    print("✓ XGBoost模型: models/xgboost_model.json")
    
    # 儲存LSTM
    model.lstm_model.save('models/lstm_model.h5')
    print("✓ LSTM模型: models/lstm_model.h5")
    
    # 儲存Scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(model.scaler, f)
    print("✓ Scaler: models/scaler.pkl")
    
    # 儲存Label Encoder
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("✓ Label Encoder: models/label_encoder.pkl")
    
    # 儲存特徵名稱
    with open('models/feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)
    print("✓ 特徵名稱: models/feature_names.json")
    
    # 儲存訓練記錄
    training_info = {
        'timestamp': datetime.now().isoformat(),
        'n_classes': len(label_encoder.classes_),
        'classes': label_encoder.classes_.tolist(),
        'n_features': len(feature_names),
        'sequence_length': model.sequence_length,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'metrics': {
            'xgboost': {'accuracy': float(xgb_acc), 'f1_score': float(xgb_f1)},
            'lstm': {'accuracy': float(lstm_acc), 'f1_score': float(lstm_f1)},
            'ensemble': {'accuracy': float(ensemble_acc), 'f1_score': float(ensemble_f1)}
        }
    }
    
    with open('models/training_info.json', 'w') as f:
        json.dump(training_info, f, indent=2)
    print("✓ 訓練記錄: models/training_info.json")
    
    print("\n" + "="*60)
    print("訓練完成")
    print("="*60)
    print(f"結束時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n最佳模型: 集成模型")
    print(f"準確率: {ensemble_acc:.4f}")
    print(f"F1 Score: {ensemble_f1:.4f}")


if __name__ == "__main__":
    import os
    import sys
    
    # 創建models目錄
    if not os.path.exists('models'):
        os.makedirs('models')
        print("創建 models/ 目錄")
    
    try:
        main()
    except FileNotFoundError:
        print("\n錯誤: 找不到 zigzag_result.csv")
        print("請先執行 test_zigzag.py 生成結果檔案")
        sys.exit(1)
    except Exception as e:
        print(f"\n錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
