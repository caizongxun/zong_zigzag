import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings
warnings.filterwarnings('ignore')

class ZigZagModelTrainer:
    """
    ZigZag模型訓練類別
    使用集成學習模型預測 HH/HL/LL/LH
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.feature_names = None
        self.best_model = None
        self.best_model_name = None
        
    def prepare_data(self, X: np.ndarray, y: np.ndarray, feature_names: list, 
                     test_size: float = 0.2, use_smote: bool = True):
        """
        準備訓練和測試數據
        """
        print("\n正在準備數據...")
        
        self.feature_names = feature_names
        
        # 編碼標籤
        y_encoded = self.label_encoder.fit_transform(y)
        print(f"  標籤映射: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        # 分割訓練和測試集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=self.random_state, stratify=y_encoded
        )
        
        print(f"  訓練集: {X_train.shape[0]} 樣本")
        print(f"  測試集: {X_test.shape[0]} 樣本")
        
        # 標準化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # SMOTE過採樣 (處理類別不平衡)
        if use_smote:
            print("  使用SMOTE過採樣...")
            smote = SMOTE(random_state=self.random_state)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
            print(f"  SMOTE後訓練集: {X_train_scaled.shape[0]} 樣本")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_random_forest(self, X_train, y_train, tune: bool = False):
        """
        訓練Random Forest模型
        """
        print("\n訓練 Random Forest...")
        
        if tune:
            print("  執行超參數調整...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestClassifier(random_state=self.random_state)
            grid_search = GridSearchCV(
                rf, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            print(f"  最佳參數: {grid_search.best_params_}")
            model = grid_search.best_estimator_
        else:
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
        
        self.models['RandomForest'] = model
        return model
    
    def train_xgboost(self, X_train, y_train, tune: bool = False):
        """
        訓練XGBoost模型
        """
        print("\n訓練 XGBoost...")
        
        # 計算類別權重
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train
        )
        sample_weights = np.array([class_weights[i] for i in y_train])
        
        if tune:
            print("  執行超參數調整...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9, 1.0]
            }
            
            xgb_model = xgb.XGBClassifier(
                objective='multi:softmax',
                num_class=len(np.unique(y_train)),
                random_state=self.random_state,
                tree_method='hist'
            )
            
            grid_search = GridSearchCV(
                xgb_model, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train, sample_weight=sample_weights)
            
            print(f"  最佳參數: {grid_search.best_params_}")
            model = grid_search.best_estimator_
        else:
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.9,
                objective='multi:softmax',
                num_class=len(np.unique(y_train)),
                random_state=self.random_state,
                tree_method='hist'
            )
            model.fit(X_train, y_train, sample_weight=sample_weights)
        
        self.models['XGBoost'] = model
        return model
    
    def train_lightgbm(self, X_train, y_train, tune: bool = False):
        """
        訓練LightGBM模型
        """
        print("\n訓練 LightGBM...")
        
        if tune:
            print("  執行超參數調整...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 50, 70]
            }
            
            lgb_model = lgb.LGBMClassifier(
                objective='multiclass',
                num_class=len(np.unique(y_train)),
                random_state=self.random_state,
                class_weight='balanced'
            )
            
            grid_search = GridSearchCV(
                lgb_model, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            print(f"  最佳參數: {grid_search.best_params_}")
            model = grid_search.best_estimator_
        else:
            model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=20,
                learning_rate=0.05,
                num_leaves=50,
                objective='multiclass',
                num_class=len(np.unique(y_train)),
                random_state=self.random_state,
                class_weight='balanced',
                verbose=-1
            )
            model.fit(X_train, y_train)
        
        self.models['LightGBM'] = model
        return model
    
    def train_gradient_boosting(self, X_train, y_train):
        """
        訓練Gradient Boosting模型
        """
        print("\n訓練 Gradient Boosting...")
        
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            random_state=self.random_state
        )
        model.fit(X_train, y_train)
        
        self.models['GradientBoosting'] = model
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name: str):
        """
        評估模型
        """
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n{model_name} 評估結果:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score (weighted): {f1:.4f}")
        
        print("\n分類報告:")
        print(classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_,
            digits=4
        ))
        
        return {'accuracy': accuracy, 'f1_score': f1, 'predictions': y_pred}
    
    def plot_feature_importance(self, model, model_name: str, top_n: int = 20):
        """
        繪製特徵重要性
        """
        if not hasattr(model, 'feature_importances_'):
            return
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'{model_name} - Top {top_n} 特徵重要性')
        plt.barh(range(top_n), importances[indices])
        plt.yticks(range(top_n), [self.feature_names[i] for i in indices])
        plt.xlabel('重要性')
        plt.tight_layout()
        
        filename = f'feature_importance_{model_name.lower()}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  特徵重要性圖已儲存: {filename}")
        plt.close()
    
    def plot_confusion_matrix(self, y_test, y_pred, model_name: str):
        """
        繪製混淆矩陣
        """
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        plt.title(f'{model_name} - 混淆矩陣')
        plt.ylabel('實際標籤')
        plt.xlabel('預測標籤')
        plt.tight_layout()
        
        filename = f'confusion_matrix_{model_name.lower()}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  混淆矩陣圖已儲存: {filename}")
        plt.close()
    
    def train_all_models(self, X_train, y_train, tune: bool = False):
        """
        訓練所有模型
        """
        print("\n" + "="*60)
        print("開始訓練所有模型")
        print("="*60)
        
        self.train_random_forest(X_train, y_train, tune=tune)
        self.train_xgboost(X_train, y_train, tune=tune)
        self.train_lightgbm(X_train, y_train, tune=tune)
        self.train_gradient_boosting(X_train, y_train)
        
        print("\n所有模型訓練完成!")
    
    def evaluate_all_models(self, X_test, y_test):
        """
        評估所有模型
        """
        print("\n" + "="*60)
        print("評估所有模型")
        print("="*60)
        
        results = {}
        best_f1 = 0
        
        for name, model in self.models.items():
            result = self.evaluate_model(model, X_test, y_test, name)
            results[name] = result
            
            # 繪製圖表
            self.plot_feature_importance(model, name)
            self.plot_confusion_matrix(y_test, result['predictions'], name)
            
            # 記錄最佳模型
            if result['f1_score'] > best_f1:
                best_f1 = result['f1_score']
                self.best_model = model
                self.best_model_name = name
        
        print("\n" + "="*60)
        print("模型比較")
        print("="*60)
        print(f"{'\u6a21型':<20} {'Accuracy':<12} {'F1-Score':<12}")
        print("-"*44)
        for name, result in results.items():
            marker = " <-- 最佳" if name == self.best_model_name else ""
            print(f"{name:<20} {result['accuracy']:<12.4f} {result['f1_score']:<12.4f}{marker}")
        
        return results
    
    def save_model(self, filename: str = 'zigzag_model.pkl'):
        """
        儲存最佳模型
        """
        if self.best_model is None:
            print("錯誤: 沒有可以儲存的模型")
            return
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n最佳模型 ({self.best_model_name}) 已儲存至: {filename}")
        
        # 儲存配置資訊
        config = {
            'model_name': self.best_model_name,
            'feature_count': len(self.feature_names),
            'label_mapping': dict(zip(self.label_encoder.classes_, 
                                     range(len(self.label_encoder.classes_))))
        }
        
        config_filename = filename.replace('.pkl', '_config.json')
        with open(config_filename, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"模型配置已儲存至: {config_filename}")


def main():
    """
    主訓練流程
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='ZigZag模型訓練')
    parser.add_argument('--input', type=str, default='zigzag_features.csv',
                       help='輸入特徵檔案')
    parser.add_argument('--tune', action='store_true',
                       help='是否執行超參數調整')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='測試集比例')
    parser.add_argument('--no-smote', action='store_true',
                       help='不使用SMOTE過採樣')
    parser.add_argument('--output', type=str, default='zigzag_model.pkl',
                       help='輸出模型檔案')
    args = parser.parse_args()
    
    print("="*60)
    print("ZigZag Swing Type 預測模型訓練")
    print("="*60)
    
    try:
        # 讀取特徵數據
        print(f"\n讀取特徵數據: {args.input}")
        df = pd.read_csv(args.input)
        print(f"總資料筆數: {len(df):,}")
        
        # 準備訓練數據
        from feature_engineering import ZigZagFeatureEngineer
        engineer = ZigZagFeatureEngineer()
        X, y, feature_names = engineer.prepare_training_data(df)
        
        # 創建Trainer
        trainer = ZigZagModelTrainer(random_state=42)
        
        # 準備數據
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            X, y, feature_names, 
            test_size=args.test_size,
            use_smote=not args.no_smote
        )
        
        # 訓練模型
        trainer.train_all_models(X_train, y_train, tune=args.tune)
        
        # 評估模型
        results = trainer.evaluate_all_models(X_test, y_test)
        
        # 儲存模型
        trainer.save_model(args.output)
        
        print("\n" + "="*60)
        print("訓練完成!")
        print("="*60)
        print(f"\n輸出檔案:")
        print(f"  - 模型: {args.output}")
        print(f"  - 配置: {args.output.replace('.pkl', '_config.json')}")
        print(f"  - 特徵重要性圖: feature_importance_*.png")
        print(f"  - 混淆矩陣圖: confusion_matrix_*.png")
        
    except FileNotFoundError:
        print(f"\n錯誤: 找不到 {args.input}")
        print("請先執行 feature_engineering.py 生成特徵數據")
        sys.exit(1)
    except Exception as e:
        print(f"\n錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
