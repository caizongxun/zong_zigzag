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

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: matplotlib 未安裝")


class TradingSignalVisualizer:
    """
    交易信號可視化工具
    顯示模型預測的 HH/HL/LH/LL 點位是否有效
    """
    
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dirs = glob.glob('models/*')
            if not model_dirs:
                raise FileNotFoundError("未找到模型目錄")
            model_dir = max(model_dirs, key=lambda x: Path(x).stat().st_mtime)
        
        self.model_dir = model_dir
        self.load_model_artifacts()
    
    def load_model_artifacts(self):
        print(f"\n從 {self.model_dir} 載入模型...\n")
        
        # 載入模型
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(f'{self.model_dir}/xgboost_model.json')
        
        # 載入編碼器
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
        
        # 映射關係
        self.swing_type_int_to_str = {0: 'HH', 1: 'HL', 2: 'LH', 3: 'LL'}
        self.swing_type_str_to_int = {'HH': 0, 'HL': 1, 'LH': 2, 'LL': 3}
    
    def validate_predictions(self, df_features, output_dir='results', recent_bars=300):
        """
        驗證預測點位是否有效
        
        參數：
            df_features: 特徵數據框
            output_dir: 輸出目錄
            recent_bars: 需要顯示的最近K棒数
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("驗證預測點位有效性")
        print("="*60)
        
        df = df_features.copy()
        df = df.dropna(subset=['close', 'high', 'low'])
        
        # 確保有 swing_type 列
        if 'swing_type' not in df.columns:
            df['swing_type'] = ''
        
        # 填充 NaN 值
        df['swing_type'] = df['swing_type'].fillna('')
        
        # 生成預測
        X = df[self.feature_names].values
        y_pred = self.xgb_model.predict(X)
        y_pred_proba = self.xgb_model.predict_proba(X)
        
        # 添加預測列
        df['predicted_signal'] = [self.swing_type_int_to_str[i] for i in y_pred]
        df['confidence'] = np.max(y_pred_proba, axis=1)
        
        # 確定是否有轉折點
        df['is_pivot'] = (df['swing_type'] != '').astype(bool)
        
        # 初始化有效性列
        df['high_valid'] = False
        df['low_valid'] = False
        
        # 計算轉折點有效性
        for i in range(1, len(df)):
            if not df.iloc[i]['is_pivot']:
                continue
            
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            current_type = df.iloc[i]['swing_type']
            
            # 找到前一個轉折點
            prev_pivot_idx = None
            for j in range(i-1, -1, -1):
                if df.iloc[j]['is_pivot']:
                    prev_pivot_idx = j
                    break
            
            if prev_pivot_idx is not None:
                prev_high = df.iloc[prev_pivot_idx]['high']
                prev_low = df.iloc[prev_pivot_idx]['low']
                
                # 驗證轉折點有效性
                if current_type == 'HH':
                    df.loc[i, 'high_valid'] = current_high > prev_high
                elif current_type == 'LL':
                    df.loc[i, 'low_valid'] = current_low < prev_low
                elif current_type == 'LH':
                    df.loc[i, 'high_valid'] = current_high < prev_high
                elif current_type == 'HL':
                    df.loc[i, 'low_valid'] = current_low > prev_low
        
        # 綜合判斷
        df['pivot_valid'] = (df['high_valid'] | df['low_valid']) & df['is_pivot']
        
        # 全體統計
        pivot_data = df[df['is_pivot']].copy()
        if len(pivot_data) > 0:
            valid_count = pivot_data['pivot_valid'].sum()
            total_count = len(pivot_data)
            valid_rate = valid_count / total_count if total_count > 0 else 0
            
            print(f"\n全體轉折點統計：")
            print(f"  總轉折點數: {total_count}")
            print(f"  有效轉折點: {valid_count}")
            print(f"  無效轉折點: {total_count - valid_count}")
            print(f"  有效率: {valid_rate:.2%}")
            
            # 按類型統計
            print(f"\n按類型統計：")
            for swing_type in ['HH', 'HL', 'LH', 'LL']:
                type_data = pivot_data[pivot_data['swing_type'] == swing_type]
                if len(type_data) > 0:
                    type_valid = type_data['pivot_valid'].sum()
                    type_total = len(type_data)
                    type_rate = type_valid / type_total if type_total > 0 else 0
                    print(f"  {swing_type}: {type_valid}/{type_total} ({type_rate:.2%})")
        
        # 生成統計圖表
        self._plot_overall_stats(df, output_dir)
        
        # 生成最近 N 根的詳細圖表
        self._plot_recent_detailed(df, output_dir, recent_bars)
        
        # 保存詳細結果
        signal_data = df[df['is_pivot']][[
            'open', 'high', 'low', 'close', 'volume',
            'swing_type', 'predicted_signal', 'confidence',
            'high_valid', 'low_valid', 'pivot_valid'
        ]].copy()
        
        signal_output = f'{output_dir}/pivot_validation_details.csv'
        signal_data.to_csv(signal_output, index=False)
        print(f"\n✓ 詳細結果已保存至: {signal_output}")
        
        return df
    
    def _plot_overall_stats(self, df, output_dir='results'):
        """
        繪製全體統計圖表 (既有圖)
        """
        if not MATPLOTLIB_AVAILABLE:
            return
        
        print(f"\n生成全體統計圖表...")
        
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(18, 12))
        
        # 1. 價格走勢图 + 轉折點
        ax1 = plt.subplot(3, 2, (1, 3))
        
        ax1.plot(range(len(df)), df['close'], color='black', linewidth=1, label='收盤價', alpha=0.7)
        ax1.fill_between(range(len(df)), df['low'], df['high'], alpha=0.2, color='gray')
        
        pivot_data = df[df['is_pivot']].copy()
        
        # HH
        hh_data = pivot_data[pivot_data['swing_type'] == 'HH']
        hh_valid = hh_data[hh_data['pivot_valid']]
        hh_invalid = hh_data[~hh_data['pivot_valid']]
        if len(hh_valid) > 0:
            ax1.scatter(hh_valid.index, hh_valid['high'], marker='^', s=200, color='green', 
                       label='HH (有效)', zorder=5, edgecolors='darkgreen', linewidth=2)
        if len(hh_invalid) > 0:
            ax1.scatter(hh_invalid.index, hh_invalid['high'], marker='v', s=100, color='lightgreen', 
                       label='HH (無效)', zorder=4, edgecolors='darkgreen', linewidth=1, alpha=0.5)
        
        # LL
        ll_data = pivot_data[pivot_data['swing_type'] == 'LL']
        ll_valid = ll_data[ll_data['pivot_valid']]
        ll_invalid = ll_data[~ll_data['pivot_valid']]
        if len(ll_valid) > 0:
            ax1.scatter(ll_valid.index, ll_valid['low'], marker='v', s=200, color='red', 
                       label='LL (有效)', zorder=5, edgecolors='darkred', linewidth=2)
        if len(ll_invalid) > 0:
            ax1.scatter(ll_invalid.index, ll_invalid['low'], marker='^', s=100, color='lightcoral', 
                       label='LL (無效)', zorder=4, edgecolors='darkred', linewidth=1, alpha=0.5)
        
        # LH
        lh_data = pivot_data[pivot_data['swing_type'] == 'LH']
        lh_valid = lh_data[lh_data['pivot_valid']]
        lh_invalid = lh_data[~lh_data['pivot_valid']]
        if len(lh_valid) > 0:
            ax1.scatter(lh_valid.index, lh_valid['high'], marker='v', s=150, color='blue', 
                       label='LH (有效)', zorder=5, edgecolors='darkblue', linewidth=2)
        if len(lh_invalid) > 0:
            ax1.scatter(lh_invalid.index, lh_invalid['high'], marker='^', s=100, color='lightblue', 
                       label='LH (無效)', zorder=4, edgecolors='darkblue', linewidth=1, alpha=0.5)
        
        # HL
        hl_data = pivot_data[pivot_data['swing_type'] == 'HL']
        hl_valid = hl_data[hl_data['pivot_valid']]
        hl_invalid = hl_data[~hl_data['pivot_valid']]
        if len(hl_valid) > 0:
            ax1.scatter(hl_valid.index, hl_valid['low'], marker='^', s=150, color='orange', 
                       label='HL (有效)', zorder=5, edgecolors='darkorange', linewidth=2)
        if len(hl_invalid) > 0:
            ax1.scatter(hl_invalid.index, hl_invalid['low'], marker='v', s=100, color='lightyellow', 
                       label='HL (無效)', zorder=4, edgecolors='darkorange', linewidth=1, alpha=0.5)
        
        ax1.set_title('一整体价格走务 + 轉折點', fontsize=14, fontweight='bold')
        ax1.set_xlabel('時間索引')
        ax1.set_ylabel('價格')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. 有效率統計
        ax2 = plt.subplot(3, 2, 2)
        
        pivot_counts = {
            'HH': len(df[df['swing_type'] == 'HH']),
            'HL': len(df[df['swing_type'] == 'HL']),
            'LH': len(df[df['swing_type'] == 'LH']),
            'LL': len(df[df['swing_type'] == 'LL'])
        }
        
        valid_counts = {
            'HH': len(df[(df['swing_type'] == 'HH') & (df['pivot_valid'])]),
            'HL': len(df[(df['swing_type'] == 'HL') & (df['pivot_valid'])]),
            'LH': len(df[(df['swing_type'] == 'LH') & (df['pivot_valid'])]),
            'LL': len(df[(df['swing_type'] == 'LL') & (df['pivot_valid'])])
        }
        
        types = list(pivot_counts.keys())
        total_vals = list(pivot_counts.values())
        valid_vals = list(valid_counts.values())
        invalid_vals = [total_vals[i] - valid_vals[i] for i in range(len(total_vals))]
        
        x = np.arange(len(types))
        width = 0.35
        
        ax2.bar(x - width/2, valid_vals, width, label='有效', color='green', alpha=0.7)
        ax2.bar(x + width/2, invalid_vals, width, label='無效', color='red', alpha=0.7)
        
        ax2.set_xlabel('轉折點類型')
        ax2.set_ylabel('數量')
        ax2.set_title('轉折點有效性分布', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(types)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加數值標籤
        for i, (v, inv) in enumerate(zip(valid_vals, invalid_vals)):
            if v > 0:
                ax2.text(i - width/2, v, str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')
            if inv > 0:
                ax2.text(i + width/2, inv, str(inv), ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 3. 有效率百分比
        ax3 = plt.subplot(3, 2, 4)
        
        valid_rates = []
        for t in types:
            total = pivot_counts[t]
            valid = valid_counts[t]
            rate = (valid / total * 100) if total > 0 else 0
            valid_rates.append(rate)
        
        colors = ['green' if rate > 70 else 'orange' if rate > 50 else 'red' for rate in valid_rates]
        bars = ax3.bar(types, valid_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax3.set_ylim([0, 100])
        ax3.set_ylabel('有效率 (%)')
        ax3.set_title('各類型轉折點有效率', fontsize=12, fontweight='bold')
        ax3.axhline(y=70, color='green', linestyle='--', linewidth=2, alpha=0.5, label='優秀 (>70%)')
        ax3.axhline(y=50, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='及格 (>50%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 添加百分比標籤
        for bar, rate in zip(bars, valid_rates):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 4. 置信度分布
        ax4 = plt.subplot(3, 2, 5)
        
        valid_confidence = df[df['pivot_valid'] & df['is_pivot']]['confidence']
        invalid_confidence = df[~df['pivot_valid'] & df['is_pivot']]['confidence']
        
        if len(valid_confidence) > 0:
            ax4.hist(valid_confidence, bins=20, alpha=0.6, label='有效預測', color='green', edgecolor='black')
        if len(invalid_confidence) > 0:
            ax4.hist(invalid_confidence, bins=20, alpha=0.6, label='無效預測', color='red', edgecolor='black')
        
        ax4.set_xlabel('預測置信度')
        ax4.set_ylabel('頻率')
        ax4.set_title('有效/無效預測的置信度分布', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. 整體統計
        ax5 = plt.subplot(3, 2, 6)
        ax5.axis('off')
        
        total_pivots = len(df[df['is_pivot']])
        valid_pivots = len(df[df['pivot_valid'] & df['is_pivot']])
        invalid_pivots = total_pivots - valid_pivots
        overall_rate = (valid_pivots / total_pivots * 100) if total_pivots > 0 else 0
        
        stats_text = f"""
轉折點有效性統計

總轉折點數: {total_pivots}
有效轉折點: {valid_pivots}
無效轉折點: {invalid_pivots}

整體有效率: {overall_rate:.2f}%

性能評級:
"""
        
        if overall_rate >= 70:
            stats_text += "優秀 (>70%)"
            color = 'green'
        elif overall_rate >= 60:
            stats_text += "良好 (60-70%)"
            color = 'orange'
        elif overall_rate >= 50:
            stats_text += "一般 (50-60%)"
            color = 'yellow'
        else:
            stats_text += "需要改進 (<50%)"
            color = 'red'
        
        ax5.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3, edgecolor='black', linewidth=2),
                family='monospace', fontweight='bold')
        
        plt.tight_layout()
        
        # 保存圖表
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"{output_dir}/pivot_validation_overall_{timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 整體統計圖表已保存至: {output_path}")
        
        plt.close()
    
    def _plot_recent_detailed(self, df, output_dir='results', recent_bars=300):
        """
        繪製最近 N 根的詳細圖表（每个轉折點有效性詳細信息）
        """
        if not MATPLOTLIB_AVAILABLE:
            return
        
        print(f"\n生成最近 {recent_bars} 根的詳細圖表...")
        
        # 取最近的 N 根
        df_recent = df.tail(recent_bars).copy()
        df_recent['index_in_recent'] = range(len(df_recent))
        
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(20, 12))
        
        # 上方: 價格走勢 + 轉折點
        ax1 = plt.subplot(2, 1, 1)
        
        ax1.plot(df_recent['index_in_recent'], df_recent['close'], color='black', 
                linewidth=1.5, label='收盤價', alpha=0.8)
        ax1.fill_between(df_recent['index_in_recent'], df_recent['low'], df_recent['high'], 
                         alpha=0.2, color='gray')
        
        pivot_data_recent = df_recent[df_recent['is_pivot']].copy()
        
        # 有效 vs 無效轉折點標記
        for swing_type, marker, valid_color, invalid_color in [
            ('HH', '^', 'darkgreen', 'lightgreen'),
            ('LL', 'v', 'darkred', 'lightcoral'),
            ('LH', 'v', 'darkblue', 'lightblue'),
            ('HL', '^', 'darkorange', 'lightyellow')
        ]:
            type_data = pivot_data_recent[pivot_data_recent['swing_type'] == swing_type]
            
            if swing_type in ['HH', 'LH']:
                price_col = 'high'
            else:
                price_col = 'low'
            
            # 有效
            valid_data = type_data[type_data['pivot_valid']]
            if len(valid_data) > 0:
                ax1.scatter(valid_data['index_in_recent'], valid_data[price_col], 
                           marker=marker, s=300, color=valid_color, 
                           zorder=5, edgecolors='black', linewidth=2,
                           label=f'{swing_type} 有效')
            
            # 無效
            invalid_data = type_data[~type_data['pivot_valid']]
            if len(invalid_data) > 0:
                ax1.scatter(invalid_data['index_in_recent'], invalid_data[price_col], 
                           marker=marker, s=150, color=invalid_color, 
                           zorder=4, edgecolors='black', linewidth=1, alpha=0.6,
                           label=f'{swing_type} 無效')
        
        ax1.set_title(f'最近 {recent_bars} 根K棒 - 價格走勢 + 轉折點有效性', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('K棒索引 (从最旧到最新)')
        ax1.set_ylabel('價格')
        ax1.legend(loc='upper left', fontsize=9, ncol=4)
        ax1.grid(True, alpha=0.3)
        
        # 下方: 为每个轉折點的詳細信息
        ax2 = plt.subplot(2, 1, 2)
        ax2.axis('off')
        
        # 制作詳細表格
        pivot_info_list = []
        for idx, (_, row) in enumerate(pivot_data_recent.iterrows()):
            swing_type = row['swing_type']
            predicted = row['predicted_signal']
            confidence = row['confidence']
            is_valid = row['pivot_valid']
            close = row['close']
            high = row['high']
            low = row['low']
            volume = row['volume']
            
            valid_text = '有效' if is_valid else '無效'
            valid_symbol = '✓' if is_valid else '✗'
            match_symbol = '✓' if swing_type == predicted else '✗'
            
            pivot_info_list.append([
                f"#{idx+1}",
                swing_type,
                predicted,
                match_symbol,
                f"{confidence:.2%}",
                valid_text,
                valid_symbol,
                f"{close:.2f}",
                f"{high:.2f}",
                f"{low:.2f}",
                f"{volume:.0f}"
            ])
        
        if pivot_info_list:
            columns = ['顺序', '实际', '预测', '符合', '置信度', 
                     '初验', '有效', '收盤', '高', '低', '成交量']
            
            table = ax2.table(cellText=pivot_info_list, colLabels=columns,
                            cellLoc='center', loc='center',
                            colWidths=[0.08, 0.08, 0.08, 0.08, 0.1, 0.08, 0.08, 0.1, 0.1, 0.1, 0.1])
            
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            
            # 设置索头样式
            for i in range(len(columns)):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # 设置内容样式
            for i, row_data in enumerate(pivot_info_list, start=1):
                for j, cell_value in enumerate(row_data):
                    cell = table[(i, j)]
                    
                    # 根据有效性改变背景色
                    if row_data[5] == '有效':
                        cell.set_facecolor('#d4edda')
                    else:
                        cell.set_facecolor('#f8d7da')
                    
                    # 字体样式
                    if j == 6:  # 有效列
                        cell.set_text_props(weight='bold', fontsize=10)
                        if row_data[5] == '有效':
                            cell.set_text_props(color='green')
                        else:
                            cell.set_text_props(color='red')
        
        plt.tight_layout()
        
        # 保存圖表
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"{output_dir}/pivot_validation_recent_{recent_bars}bars_{timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 最近 {recent_bars} 根詳細圖表已保存至: {output_path}")
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='交易信號可視化 - 驗證轉折點有效性',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例：
  # 使用最新模型驗證
  python trading_signal_visualization.py
  
  # 指定特定模型
  python trading_signal_visualization.py --model-dir models/BTCUSDT_15m_20260109_033108
  
  # 指定數據路徑
  python trading_signal_visualization.py --data-path data_cache/features.csv
  
  # 指定最近 500 根K棒
  python trading_signal_visualization.py --recent-bars 500
        '''
    )
    
    parser.add_argument('--model-dir', type=str, default=None,
                       help='模型目錄路徑')
    parser.add_argument('--data-path', type=str, default='data_cache/features.csv',
                       help='特徵數據路徑')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='結果輸出目錄')
    parser.add_argument('--recent-bars', type=int, default=300,
                       help='最近 N 根K棒')
    
    args = parser.parse_args()
    
    try:
        # 創建可視化器
        visualizer = TradingSignalVisualizer(args.model_dir)
        
        # 載入數據
        print(f"\n載入特徵數據: {args.data_path}")
        df_features = pd.read_csv(args.data_path)
        print(f"✓ 載入 {len(df_features):,} 條記錄")
        
        # 驗證並可視化
        df_results = visualizer.validate_predictions(df_features, args.output_dir, args.recent_bars)
        
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
