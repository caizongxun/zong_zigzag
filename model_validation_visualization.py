import pandas as pd
import numpy as np
import pickle
import json
import argparse
import glob
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve, auc
)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: matplotlib 和 seaborn 未安裝，將跳過圖表生成")


class ModelValidator:
    """
    模型驗證和可視化工具
    """
    
    def __init__(self, model_dir=None):
        """
        初始化驗證器
        
        參數：
            model_dir (str): 模型目錄路徑。如果為 None，將使用最新的模型
        """
        if model_dir is None:
            # 尋找最新的模型目錄
            model_dirs = glob.glob('models/*')
            if not model_dirs:
                raise FileNotFoundError("未找到模型目錄")
            model_dir = max(model_dirs, key=lambda x: Path(x).stat().st_mtime)
        
        self.model_dir = model_dir
        self.load_model_artifacts()
    
    def load_model_artifacts(self):
        """
        載入模型和相關成品
        """
        print(f"\n從 {self.model_dir} 載入模型...\n")
        
        # 載入 XGBoost 模型
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(f'{self.model_dir}/xgboost_model.json')
        
        # 載入標籤編碼器
        with open(f'{self.model_dir}/label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # 載入特徵名稱
        with open(f'{self.model_dir}/feature_names.json', 'r') as f:
            self.feature_names = json.load(f)
        
        # 載入參數
        with open(f'{self.model_dir}/params.json', 'r') as f:
            self.params = json.load(f)
        
        print(f"✓ 模型載入完成")
        print(f"  交易對: {self.params['pair']}")
        print(f"  時間框架: {self.params['interval']}")
        print(f"  特徵數: {len(self.feature_names)}")
        print(f"  類別數: {len(self.label_encoder.classes_)}")
        print(f"  類別: {', '.join(self.label_encoder.classes_)}")
    
    def validate_on_data(self, df_features):
        """
        在新數據上驗證模型
        
        參數：
            df_features (DataFrame): 包含特徵的數據框
        
        返回：
            dict: 驗證結果
        """
        print("\n" + "="*60)
        print("模型驗證")
        print("="*60)
        
        # 準備數據
        df = df_features.copy()
        df = df.dropna()
        
        # 提取標籤和特徵
        label_col = 'swing_type'
        if label_col not in df.columns:
            print("警告: 未找到標籤列 'swing_type'，無法驗證")
            return None
        
        df = df[df[label_col] != '']
        
        if len(df) == 0:
            print("警告: 沒有有效的標籤數據")
            return None
        
        # 編碼標籤
        y_true = self.label_encoder.transform(df[label_col])
        
        # 提取特徵
        X = df[self.feature_names].values
        
        # 預測
        y_pred = self.xgb_model.predict(X)
        y_pred_proba = self.xgb_model.predict_proba(X)
        
        # 計算指標
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        }
        
        # 打印結果
        print(f"\n總樣本數: {len(df)}")
        print(f"\n性能指標:")
        print(f"  準確率 (Accuracy): {metrics['accuracy']:.4f}")
        print(f"  精確率 (Precision): {metrics['precision']:.4f}")
        print(f"  召回率 (Recall): {metrics['recall']:.4f}")
        print(f"  F1 Score (加權): {metrics['f1']:.4f}")
        print(f"  F1 Score (Macro): {metrics['macro_f1']:.4f}")
        
        # 詳細分類報告
        print(f"\n詳細分類報告:")
        print(classification_report(
            y_true, y_pred,
            target_names=self.label_encoder.classes_,
            zero_division=0
        ))
        
        # 混淆矩陣
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n混淆矩陣:")
        print(cm)
        
        # 添加預測詳情
        df['y_true'] = y_true
        df['y_pred'] = y_pred
        df['y_pred_label'] = self.label_encoder.inverse_transform(y_pred)
        df['prediction_correct'] = (y_true == y_pred)
        df['max_prob'] = np.max(y_pred_proba, axis=1)
        
        metrics['confusion_matrix'] = cm
        metrics['predictions'] = df
        
        return metrics
    
    def generate_visualizations(self, metrics, output_dir='results'):
        """
        生成可視化圖表
        
        參數：
            metrics (dict): 驗證結果
            output_dir (str): 輸出目錄
        """
        if not MATPLOTLIB_AVAILABLE:
            print("跳過圖表生成: matplotlib 未安裝")
            return
        
        print(f"\n生成可視化圖表...")
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 設置風格
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 12)
        
        # 創建圖表
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 混淆矩陣
        ax1 = plt.subplot(2, 3, 1)
        cm = metrics['confusion_matrix']
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_,
            ax=ax1, cbar=True
        )
        ax1.set_title('混淆矩陣', fontsize=12, fontweight='bold')
        ax1.set_ylabel('真實標籤')
        ax1.set_xlabel('預測標籤')
        
        # 2. 性能指標條形圖
        ax2 = plt.subplot(2, 3, 2)
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        metrics_values = [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1']
        ]
        bars = ax2.bar(metrics_names, metrics_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax2.set_ylim([0, 1])
        ax2.set_title('性能指標', fontsize=12, fontweight='bold')
        ax2.set_ylabel('分數')
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # 3. 類別分佈
        ax3 = plt.subplot(2, 3, 3)
        predictions_df = metrics['predictions']
        true_counts = predictions_df['swing_type'].value_counts()
        pred_counts = predictions_df['y_pred_label'].value_counts()
        
        x = np.arange(len(self.label_encoder.classes_))
        width = 0.35
        
        true_vals = [true_counts.get(label, 0) for label in self.label_encoder.classes_]
        pred_vals = [pred_counts.get(label, 0) for label in self.label_encoder.classes_]
        
        ax3.bar(x - width/2, true_vals, width, label='實際', alpha=0.8)
        ax3.bar(x + width/2, pred_vals, width, label='預測', alpha=0.8)
        ax3.set_xlabel('標籤類別')
        ax3.set_ylabel('數量')
        ax3.set_title('標籤類別分佈對比', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(self.label_encoder.classes_)
        ax3.legend()
        
        # 4. 正確率按類別
        ax4 = plt.subplot(2, 3, 4)
        class_accuracy = []
        for label in self.label_encoder.classes_:
            mask = predictions_df['swing_type'] == label
            if mask.sum() > 0:
                acc = predictions_df[mask]['prediction_correct'].mean()
                class_accuracy.append(acc)
            else:
                class_accuracy.append(0)
        
        bars = ax4.bar(self.label_encoder.classes_, class_accuracy, color='skyblue')
        ax4.set_ylim([0, 1])
        ax4.set_title('各類別正確率', fontsize=12, fontweight='bold')
        ax4.set_ylabel('正確率')
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # 5. 預測置信度分佈
        ax5 = plt.subplot(2, 3, 5)
        ax5.hist(predictions_df['max_prob'], bins=50, edgecolor='black', alpha=0.7)
        ax5.axvline(predictions_df['max_prob'].mean(), color='red', linestyle='--', 
                   label=f'平均: {predictions_df["max_prob"].mean():.3f}')
        ax5.set_xlabel('最大預測概率')
        ax5.set_ylabel('頻率')
        ax5.set_title('預測置信度分佈', fontsize=12, fontweight='bold')
        ax5.legend()
        
        # 6. 錯誤分析
        ax6 = plt.subplot(2, 3, 6)
        errors = predictions_df[~predictions_df['prediction_correct']]
        if len(errors) > 0:
            error_types = errors.groupby(['swing_type', 'y_pred_label']).size().reset_index(name='count')
            error_text = "錯誤預測類型 (實際 -> 預測):\n\n"
            for _, row in error_types.iterrows():
                error_text += f"{row['swing_type']} -> {row['y_pred_label']}: {row['count']}\n"
            ax6.text(0.1, 0.5, error_text, fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax6.text(0.5, 0.5, '完美預測！\n沒有錯誤', ha='center', va='center', fontsize=14)
        ax6.axis('off')
        ax6.set_title('錯誤分析', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存圖表
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"{output_dir}/model_validation_{timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 圖表已保存至: {output_path}")
        
        # 也保存為 HTML 交互式版本
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            fig_plotly = make_subplots(
                rows=2, cols=3,
                subplot_titles=('混淆矩陣', '性能指標', '標籤分佈', 
                               '各類別正確率', '預測置信度', '錯誤分析')
            )
            
            # 混淆矩陣熱力圖
            fig_plotly.add_trace(
                go.Heatmap(z=cm, x=self.label_encoder.classes_, y=self.label_encoder.classes_),
                row=1, col=1
            )
            
            # 性能指標
            fig_plotly.add_trace(
                go.Bar(x=metrics_names, y=metrics_values, name='分數'),
                row=1, col=2
            )
            
            # 標籤分佈
            fig_plotly.add_trace(
                go.Bar(x=self.label_encoder.classes_, y=true_vals, name='實際'),
                row=1, col=3
            )
            fig_plotly.add_trace(
                go.Bar(x=self.label_encoder.classes_, y=pred_vals, name='預測'),
                row=1, col=3
            )
            
            # 各類別正確率
            fig_plotly.add_trace(
                go.Bar(x=self.label_encoder.classes_, y=class_accuracy, name='正確率'),
                row=2, col=1
            )
            
            # 預測置信度
            fig_plotly.add_trace(
                go.Histogram(x=predictions_df['max_prob'], name='置信度', nbinsx=50),
                row=2, col=2
            )
            
            fig_plotly.update_layout(height=900, showlegend=True, title_text="模型驗證儀表板")
            
            output_html = f"{output_dir}/model_validation_{timestamp}.html"
            fig_plotly.write_html(output_html)
            print(f"✓ 交互式圖表已保存至: {output_html}")
        except ImportError:
            print("跳過 Plotly 圖表: plotly 未安裝")
        
        plt.close()
    
    def predict_signal(self, df_features, threshold=0.5):
        """
        使用模型生成交易信號
        
        參數：
            df_features (DataFrame): 包含特徵的數據框
            threshold (float): 置信度閾值
        
        返回：
            DataFrame: 包含預測和信號的數據框
        """
        print("\n" + "="*60)
        print("生成交易信號")
        print("="*60)
        
        df = df_features.copy()
        
        # 預測
        X = df[self.feature_names].values
        y_pred = self.xgb_model.predict(X)
        y_pred_proba = self.xgb_model.predict_proba(X)
        max_prob = np.max(y_pred_proba, axis=1)
        
        df['predicted_class'] = self.label_encoder.inverse_transform(y_pred)
        df['confidence'] = max_prob
        df['high_confidence'] = max_prob >= threshold
        
        # 基於預測生成信號
        df['signal'] = 'HOLD'
        df.loc[df['predicted_class'] == 'HH', 'signal'] = 'BUY'  # 更高高點 - 上升趨勢
        df.loc[df['predicted_class'] == 'LL', 'signal'] = 'SELL'  # 更低低點 - 下降趨勢
        df.loc[~df['high_confidence'], 'signal'] = 'HOLD'  # 低信心保持
        
        # 統計
        print(f"\n信號分佈:")
        signal_counts = df['signal'].value_counts()
        for signal, count in signal_counts.items():
            pct = count / len(df) * 100
            print(f"  {signal}: {count:,} ({pct:.2f}%)")
        
        print(f"\n平均置信度: {df['confidence'].mean():.4f}")
        print(f"高信心信號 (>= {threshold}): {df['high_confidence'].sum():,} ({df['high_confidence'].sum()/len(df)*100:.2f}%)")
        
        return df
    
    def generate_report(self, metrics, output_file='results/model_validation_report.txt'):
        """
        生成文字報告
        
        參數：
            metrics (dict): 驗證結果
            output_file (str): 輸出檔案路徑
        """
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("模型驗證報告\n")
            f.write("="*60 + "\n\n")
            
            f.write("模型信息:\n")
            f.write(f"  模型目錄: {self.model_dir}\n")
            f.write(f"  交易對: {self.params['pair']}\n")
            f.write(f"  時間框架: {self.params['interval']}\n")
            f.write(f"  ZigZag Depth: {self.params['depth']}\n")
            f.write(f"  ZigZag Deviation: {self.params['deviation']}%\n")
            f.write(f"  生成時間: {self.params['timestamp']}\n\n")
            
            f.write("性能指標:\n")
            f.write(f"  準確率: {metrics['accuracy']:.4f}\n")
            f.write(f"  精確率: {metrics['precision']:.4f}\n")
            f.write(f"  召回率: {metrics['recall']:.4f}\n")
            f.write(f"  F1 Score (加權): {metrics['f1']:.4f}\n")
            f.write(f"  F1 Score (Macro): {metrics['macro_f1']:.4f}\n\n")
            
            f.write("混淆矩陣:\n")
            cm = metrics['confusion_matrix']
            f.write(f"  {cm}\n\n")
            
            predictions_df = metrics['predictions']
            correct_count = predictions_df['prediction_correct'].sum()
            total_count = len(predictions_df)
            
            f.write(f"預測統計:\n")
            f.write(f"  總樣本數: {total_count}\n")
            f.write(f"  正確預測: {correct_count}\n")
            f.write(f"  錯誤預測: {total_count - correct_count}\n")
            f.write(f"  正確率: {correct_count/total_count:.2%}\n")
        
        print(f"\n✓ 報告已保存至: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='ZigZag 模型驗證和可視化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例：
  # 使用最新模型進行驗證
  python model_validation_visualization.py
  
  # 指定特定模型目錄
  python model_validation_visualization.py --model-dir models/BTCUSDT_15m_20260109_033108
  
  # 生成信號
  python model_validation_visualization.py --signal --threshold 0.7
        '''
    )
    
    parser.add_argument('--model-dir', type=str, default=None,
                       help='模型目錄路徑（預設使用最新模型）')
    parser.add_argument('--data-path', type=str, default='data_cache/features.csv',
                       help='特徵數據路徑')
    parser.add_argument('--signal', action='store_true',
                       help='生成交易信號')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='置信度閾值（預設: 0.5）')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='結果輸出目錄')
    
    args = parser.parse_args()
    
    try:
        # 創建驗證器
        validator = ModelValidator(args.model_dir)
        
        # 載入特徵數據
        print(f"\n載入特徵數據: {args.data_path}")
        df_features = pd.read_csv(args.data_path)
        print(f"✓ 載入 {len(df_features):,} 條記錄")
        
        # 驗證模型
        metrics = validator.validate_on_data(df_features)
        
        if metrics is not None:
            # 生成可視化
            validator.generate_visualizations(metrics, args.output_dir)
            
            # 生成報告
            validator.generate_report(metrics, f'{args.output_dir}/model_validation_report.txt')
            
            # 生成信號（可選）
            if args.signal:
                predictions_df = validator.predict_signal(df_features, args.threshold)
                signal_output = f'{args.output_dir}/trading_signals.csv'
                predictions_df.to_csv(signal_output, index=False)
                print(f"✓ 交易信號已保存至: {signal_output}")
        
        print("\n" + "#"*60)
        print("# 驗證完成")
        print("#"*60)
        
    except Exception as e:
        print(f"\n錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    import sys
    sys.exit(0 if success else 1)
