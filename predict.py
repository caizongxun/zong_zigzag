import pandas as pd
import numpy as np
import pickle
import sys
import argparse
from feature_engineering import ZigZagFeatureEngineer

class ZigZagPredictor:
    """
    ZigZag Swing Type預測器
    加載訓練好的模型並進行預測
    """
    
    def __init__(self, model_path: str = 'zigzag_model.pkl'):
        """
        Args:
            model_path: 模型檔案路徑
        """
        print(f"加載模型: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        
        print(f"模型類型: {self.model_name}")
        print(f"特徵數量: {len(self.feature_names)}")
        print(f"標籤類別: {list(self.label_encoder.classes_)}")
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        對新數據進行預測
        
        Args:
            df: 包含ZigZag計算結果的DataFrame
            
        Returns:
            包含預測結果的DataFrame
        """
        print("\n正在進行預測...")
        
        # 生成特徵
        engineer = ZigZagFeatureEngineer()
        df_features = engineer.create_all_features(df)
        
        # 篩選ZigZag轉折點
        zigzag_points = df_features[df_features['zigzag'].notna()].copy()
        print(f"轉折點數量: {len(zigzag_points)}")
        
        if len(zigzag_points) == 0:
            print("警告: 沒有找到ZigZag轉折點")
            return df_features
        
        # 準備特徵
        X = zigzag_points[self.feature_names].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X)
        
        # 預測
        y_pred_encoded = self.model.predict(X_scaled)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        # 預測機率
        y_pred_proba = self.model.predict_proba(X_scaled)
        
        # 將預測結果加入DataFrame
        zigzag_points['predicted_swing'] = y_pred
        zigzag_points['prediction_confidence'] = np.max(y_pred_proba, axis=1)
        
        # 添加各類別機率
        for i, label in enumerate(self.label_encoder.classes_):
            zigzag_points[f'prob_{label}'] = y_pred_proba[:, i]
        
        # 合併回原始DataFrame
        df_result = df_features.copy()
        for col in ['predicted_swing', 'prediction_confidence'] + \
                   [f'prob_{label}' for label in self.label_encoder.classes_]:
            df_result[col] = np.nan
            df_result.loc[zigzag_points.index, col] = zigzag_points[col]
        
        print("\n預測完成!")
        print(f"預測分佈:")
        print(zigzag_points['predicted_swing'].value_counts())
        
        return df_result
    
    def predict_next_swing(self, df: pd.DataFrame, return_proba: bool = True) -> dict:
        """
        預測下一個Swing Type
        基於最近的市場狀態
        
        Args:
            df: 包含ZigZag計算結果的DataFrame
            return_proba: 是否返回機率
            
        Returns:
            預測結果字典
        """
        # 生成特徵
        engineer = ZigZagFeatureEngineer()
        df_features = engineer.create_all_features(df)
        
        # 取最後一筆數據
        latest_features = df_features.iloc[-1:][self.feature_names].values
        latest_features = np.nan_to_num(latest_features, nan=0.0, posinf=0.0, neginf=0.0)
        latest_features_scaled = self.scaler.transform(latest_features)
        
        # 預測
        y_pred_encoded = self.model.predict(latest_features_scaled)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)[0]
        
        result = {
            'predicted_swing': y_pred,
            'timestamp': df.iloc[-1].get('timestamp', None),
            'current_price': df.iloc[-1]['close']
        }
        
        if return_proba:
            y_pred_proba = self.model.predict_proba(latest_features_scaled)[0]
            result['confidence'] = float(np.max(y_pred_proba))
            result['probabilities'] = {
                label: float(prob) 
                for label, prob in zip(self.label_encoder.classes_, y_pred_proba)
            }
        
        return result


def main():
    """
    主預測流程
    """
    parser = argparse.ArgumentParser(description='ZigZag Swing Type預測')
    parser.add_argument('--input', type=str, default='zigzag_result.csv',
                       help='輸入CSV檔案')
    parser.add_argument('--model', type=str, default='zigzag_model.pkl',
                       help='模型檔案')
    parser.add_argument('--output', type=str, default='zigzag_predictions.csv',
                       help='輸出預測結果檔案')
    parser.add_argument('--next-only', action='store_true',
                       help='只預測下一個Swing')
    args = parser.parse_args()
    
    print("="*60)
    print("ZigZag Swing Type預測")
    print("="*60)
    
    try:
        # 加載預測器
        predictor = ZigZagPredictor(args.model)
        
        # 讀取數據
        print(f"\n讀取數據: {args.input}")
        df = pd.read_csv(args.input)
        print(f"資料筆數: {len(df):,}")
        
        if args.next_only:
            # 只預測下一個
            print("\n預測模式: 下一個Swing")
            result = predictor.predict_next_swing(df)
            
            print("\n" + "="*60)
            print("預測結果")
            print("="*60)
            print(f"預測 Swing Type: {result['predicted_swing']}")
            print(f"信心度: {result['confidence']:.2%}")
            print(f"當前價格: {result['current_price']:.2f}")
            if result.get('timestamp'):
                print(f"時間: {result['timestamp']}")
            
            print("\n各類別機率:")
            for label, prob in sorted(result['probabilities'].items(), 
                                     key=lambda x: x[1], reverse=True):
                print(f"  {label}: {prob:.2%}")
        else:
            # 預測所有轉折點
            print("\n預測模式: 所有轉折點")
            df_result = predictor.predict(df)
            
            # 儲存結果
            print(f"\n儲存預測結果至: {args.output}")
            df_result.to_csv(args.output, index=False)
            
            # 顯示部分結果
            pred_points = df_result[df_result['predicted_swing'].notna()]
            if len(pred_points) > 0:
                print("\n部分預測結果 (最後10個轉折點):")
                display_cols = ['timestamp', 'close', 'swing_type', 'predicted_swing', 
                               'prediction_confidence']
                display_cols = [c for c in display_cols if c in pred_points.columns]
                print(pred_points[display_cols].tail(10).to_string(index=False))
                
                # 計算準確率 (如果有實際標籤)
                if 'swing_type' in pred_points.columns:
                    valid_mask = pred_points['swing_type'].isin(['HH', 'HL', 'LL', 'LH'])
                    if valid_mask.sum() > 0:
                        accuracy = (pred_points[valid_mask]['swing_type'] == 
                                  pred_points[valid_mask]['predicted_swing']).mean()
                        print(f"\n預測準確率: {accuracy:.2%}")
        
        print("\n" + "="*60)
        print("預測完成!")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\n錯誤: 找不到檔案")
        print(str(e))
        sys.exit(1)
    except Exception as e:
        print(f"\n錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
