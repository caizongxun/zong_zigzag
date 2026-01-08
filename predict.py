import pandas as pd
import numpy as np
import pickle
import json
import xgboost as xgb
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import ZigZagFeatureEngineering

class ZigZagPredictor:
    """
    使用訓練好的模型進行預測
    """
    
    def __init__(self, model_dir: str = 'models'):
        """
        Args:
            model_dir: 模型檔案目錄
        """
        self.model_dir = model_dir
        self.load_models()
        
    def load_models(self):
        """
        載入所有模型和配置
        """
        print("正在載入模型...")
        
        # 載入XGBoost
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(f'{self.model_dir}/xgboost_model.json')
        print("✓ XGBoost模型")
        
        # 載入LSTM
        self.lstm_model = keras.models.load_model(f'{self.model_dir}/lstm_model.h5')
        print("✓ LSTM模型")
        
        # 載入Scaler
        with open(f'{self.model_dir}/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        print("✓ Scaler")
        
        # 載入Label Encoder
        with open(f'{self.model_dir}/label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
        print("✓ Label Encoder")
        
        # 載入特徵名稱
        with open(f'{self.model_dir}/feature_names.json', 'r') as f:
            self.feature_names = json.load(f)
        print("✓ 特徵名稱")
        
        # 載入訓練記錄
        with open(f'{self.model_dir}/training_info.json', 'r') as f:
            self.training_info = json.load(f)
        self.sequence_length = self.training_info['sequence_length']
        print("✓ 訓練記錄")
        
        print(f"\n模型資訊:")
        print(f"  訓練時間: {self.training_info['timestamp']}")
        print(f"  類別數: {self.training_info['n_classes']}")
        print(f"  特徵數: {self.training_info['n_features']}")
        print(f"  準確率: {self.training_info['metrics']['ensemble']['accuracy']:.4f}")
    
    def prepare_lstm_sequences(self, X):
        """
        準備LSTM序列
        """
        n_samples = len(X) - self.sequence_length + 1
        X_seq = np.zeros((n_samples, self.sequence_length, X.shape[1]))
        
        for i in range(n_samples):
            X_seq[i] = X[i:i+self.sequence_length]
        
        return X_seq
    
    def predict(self, df: pd.DataFrame, use_ensemble: bool = True) -> pd.DataFrame:
        """
        預測新資料
        
        Args:
            df: 包含OHLCV和ZigZag結果的DataFrame
            use_ensemble: 是否使用集成模型
            
        Returns:
            包含預測結果的DataFrame
        """
        # 特徵工程
        print("\n正在生成特徵...")
        fe = ZigZagFeatureEngineering()
        df_features = fe.create_features(df, verbose=False)
        
        # 提取特徵
        X = df_features[self.feature_names].values
        
        if use_ensemble:
            print("使用集成模型進行預測...")
            
            # XGBoost預測
            xgb_proba = self.xgb_model.predict_proba(X)
            
            # LSTM預測
            X_scaled = self.scaler.transform(X)
            X_seq = self.prepare_lstm_sequences(X_scaled)
            lstm_proba = self.lstm_model.predict(X_seq, verbose=0)
            
            # 集成
            ensemble_proba = 0.6 * xgb_proba[self.sequence_length-1:] + 0.4 * lstm_proba
            predictions = np.argmax(ensemble_proba, axis=1)
            confidence = np.max(ensemble_proba, axis=1)
            
            # 轉換回標籤
            predicted_labels = self.label_encoder.inverse_transform(predictions)
            
            # 創建結果 DataFrame
            result_df = df_features.iloc[self.sequence_length-1:].copy()
            result_df['predicted_swing'] = predicted_labels
            result_df['confidence'] = confidence
            
            # 添加各類別機率
            for i, label in enumerate(self.label_encoder.classes_):
                result_df[f'prob_{label}'] = ensemble_proba[:, i]
        
        else:
            print("使用XGBoost進行預測...")
            predictions = self.xgb_model.predict(X)
            proba = self.xgb_model.predict_proba(X)
            
            predicted_labels = self.label_encoder.inverse_transform(predictions)
            confidence = np.max(proba, axis=1)
            
            result_df = df_features.copy()
            result_df['predicted_swing'] = predicted_labels
            result_df['confidence'] = confidence
            
            for i, label in enumerate(self.label_encoder.classes_):
                result_df[f'prob_{label}'] = proba[:, i]
        
        print(f"\n預測完成: {len(result_df):,} 筆")
        
        return result_df
    
    def predict_next_swing(self, df: pd.DataFrame, top_n: int = 5) -> dict:
        """
        預測下一個可能的swing type
        
        Args:
            df: 最近的K線資料
            top_n: 顯示前 N 個預測結果
            
        Returns:
            預測結果字典
        """
        result_df = self.predict(df, use_ensemble=True)
        
        # 取最後一筆
        last_pred = result_df.iloc[-1]
        
        # 排序機率
        probs = {}
        for label in self.label_encoder.classes_:
            probs[label] = last_pred[f'prob_{label}']
        
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        result = {
            'predicted': last_pred['predicted_swing'],
            'confidence': last_pred['confidence'],
            'top_predictions': sorted_probs,
            'timestamp': last_pred.get('timestamp', 'N/A'),
            'close_price': last_pred['close']
        }
        
        return result


def main():
    """
    示範用法
    """
    import sys
    
    print("="*60)
    print("ZigZag Swing Type 預測")
    print("="*60)
    
    try:
        # 載入模型
        predictor = ZigZagPredictor()
        
        # 讀取資料
        print("\n讀取 zigzag_result.csv...")
        df = pd.read_csv('zigzag_result.csv')
        print(f"資料筆數: {len(df):,}")
        
        # 使用最後 1000 筆進行預測
        df_recent = df.tail(1000).copy()
        
        # 預測
        result_df = predictor.predict(df_recent, use_ensemble=True)
        
        # 顯示結果
        print("\n" + "="*60)
        print("預測結果概覽")
        print("="*60)
        
        print(f"\n預測筆數: {len(result_df):,}")
        print(f"\n預測分佈:")
        print(result_df['predicted_swing'].value_counts())
        
        print(f"\n平均信心度: {result_df['confidence'].mean():.4f}")
        
        # 顯示最近 10 筆預測
        print("\n最近 10 筆預測:")
        display_cols = ['close', 'predicted_swing', 'confidence']
        if 'timestamp' in result_df.columns:
            display_cols.insert(0, 'timestamp')
        print(result_df[display_cols].tail(10).to_string(index=False))
        
        # 預測下一個swing
        print("\n" + "="*60)
        print("下一個Swing預測")
        print("="*60)
        
        next_swing = predictor.predict_next_swing(df_recent)
        print(f"\n當前價格: {next_swing['close_price']:.2f}")
        if next_swing['timestamp'] != 'N/A':
            print(f"時間: {next_swing['timestamp']}")
        print(f"\n預測: {next_swing['predicted']}")
        print(f"信心度: {next_swing['confidence']:.4f}")
        
        print(f"\n所有可能性:")
        for label, prob in next_swing['top_predictions']:
            print(f"  {label}: {prob:.4f} ({prob*100:.1f}%)")
        
        # 儲存預測結果
        output_file = 'prediction_results.csv'
        result_df.to_csv(output_file, index=False)
        print(f"\n預測結果已儲存: {output_file}")
        
        print("\n" + "="*60)
        print("完成")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\n錯誤: 找不到檔案")
        print("請確保:")
        print("1. 已執行 test_zigzag.py 生成 zigzag_result.csv")
        print("2. 已執行 train_model.py 訓練模型")
        sys.exit(1)
    except Exception as e:
        print(f"\n錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
