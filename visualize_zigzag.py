import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import sys
import argparse

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def parse_arguments():
    """
    解析命令列參數
    """
    parser = argparse.ArgumentParser(
        description='ZigZag結果視覺化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\n範例:
  # 顯示最後300根K棒
  python visualize_zigzag.py
  
  # 顯示最後500根K棒
  python visualize_zigzag.py --bars 500
  
  # 指定輸入檔案
  python visualize_zigzag.py --input my_result.csv --bars 300
  
  # 儲存為圖片
  python visualize_zigzag.py --output zigzag_chart.png
        """
    )
    
    parser.add_argument('--input', type=str, default='zigzag_result.csv',
                        help='輸入CSV檔案 (預設: zigzag_result.csv)')
    parser.add_argument('--bars', type=int, default=300,
                        help='顯示的K棒數量 (預設: 300)')
    parser.add_argument('--output', type=str, default='',
                        help='儲存圖片路徑 (留空則直接顯示)')
    parser.add_argument('--width', type=int, default=16,
                        help='圖表寬度 (英寸, 預設: 16)')
    parser.add_argument('--height', type=int, default=10,
                        help='圖表高度 (英寸, 預設: 10)')
    parser.add_argument('--dpi', type=int, default=100,
                        help='圖片解析度 (預設: 100)')
    
    return parser.parse_args()


def plot_zigzag(df: pd.DataFrame, n_bars: int = 300, figsize: tuple = (16, 10), 
                output_path: str = '', dpi: int = 100):
    """
    繪製ZigZag圖表
    
    Args:
        df: 包含ZigZag資料的DataFrame
        n_bars: 顯示的K棒數量
        figsize: 圖表大小
        output_path: 儲存路徑
        dpi: 圖片解析度
    """
    # 取最後 n_bars 筆資料
    df_plot = df.tail(n_bars).copy()
    df_plot = df_plot.reset_index(drop=True)
    
    # 篩選出ZigZag點
    zigzag_points = df_plot[df_plot['zigzag'].notna()].copy()
    
    print(f"\n準備繪製圖表...")
    print(f"顯示範圍: 最後 {n_bars} 根K棒")
    print(f"ZigZag轉折點: {len(zigzag_points)} 個")
    
    if 'timestamp' in df_plot.columns:
        df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'])
        print(f"時間範圏: {df_plot['timestamp'].min()} 至 {df_plot['timestamp'].max()}")
    
    # 創建圖表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1], 
                                     gridspec_kw={'hspace': 0.3})
    
    # ========== 主圖: K線 + ZigZag ==========
    # 繪製K線
    for idx in range(len(df_plot)):
        color = 'green' if df_plot['close'].iloc[idx] >= df_plot['open'].iloc[idx] else 'red'
        ax1.plot([idx, idx], [df_plot['low'].iloc[idx], df_plot['high'].iloc[idx]], 
                color=color, linewidth=0.5, alpha=0.3)
        ax1.plot([idx, idx], [df_plot['open'].iloc[idx], df_plot['close'].iloc[idx]], 
                color=color, linewidth=2, alpha=0.8)
    
    # 繪製ZigZag線
    if len(zigzag_points) > 0:
        ax1.plot(zigzag_points.index, zigzag_points['zigzag'], 
                color='blue', linewidth=2, marker='o', markersize=6, 
                label='ZigZag', zorder=5)
        
        # 標記HH/HL/LL/LH
        colors = {'HH': 'darkgreen', 'HL': 'lightgreen', 'LH': 'orange', 'LL': 'darkred'}
        for swing_type, color in colors.items():
            points = zigzag_points[zigzag_points['swing_type'] == swing_type]
            if len(points) > 0:
                for idx, row in points.iterrows():
                    # 根據swing type決定標記位置
                    y_offset = 100 if swing_type in ['HH', 'LH'] else -100
                    ax1.annotate(swing_type, 
                               xy=(idx, row['zigzag']),
                               xytext=(0, y_offset),
                               textcoords='offset points',
                               ha='center',
                               fontsize=9,
                               fontweight='bold',
                               color=color,
                               bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor='white', 
                                       edgecolor=color, 
                                       alpha=0.8),
                               arrowprops=dict(arrowstyle='->', 
                                             connectionstyle='arc3,rad=0',
                                             color=color,
                                             lw=1.5))
    
    ax1.set_title(f'BTC 15m ZigZag 分析 (最後{n_bars}根K棒)', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('價格 (USDT)', fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', fontsize=10)
    
    # 設定X軸
    if 'timestamp' in df_plot.columns:
        # 只顯示部分時間標籤
        n_ticks = min(10, len(df_plot))
        tick_indices = [int(i * len(df_plot) / n_ticks) for i in range(n_ticks)]
        ax1.set_xticks(tick_indices)
        ax1.set_xticklabels([df_plot['timestamp'].iloc[i].strftime('%m-%d %H:%M') 
                            for i in tick_indices], rotation=45, ha='right')
    else:
        ax1.set_xlabel('K棒索引', fontsize=12)
    
    # ========== 副圖: Swing Type分佈 ==========
    swing_counts = df_plot['swing_type'].value_counts()
    swing_types = ['HH', 'HL', 'LH', 'LL']
    counts = [swing_counts.get(st, 0) for st in swing_types]
    colors_bar = ['darkgreen', 'lightgreen', 'orange', 'darkred']
    
    bars = ax2.bar(swing_types, counts, color=colors_bar, alpha=0.7, edgecolor='black')
    ax2.set_title('Swing Type 分佈', fontsize=14, fontweight='bold', pad=10)
    ax2.set_ylabel('數量', fontsize=12)
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # 在柱狀圖上顯示數值
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 顯示統計資訊
    total_pivots = len(zigzag_points)
    info_text = f'總轉折點: {total_pivots}'
    ax2.text(0.98, 0.95, info_text, transform=ax2.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 儲存或顯示
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"\n圖表已儲存至: {output_path}")
    else:
        print(f"\n正在顯示圖表... (關閉視窗以結束程式)")
        plt.show()


def main():
    """
    主函數
    """
    args = parse_arguments()
    
    print("="*60)
    print("ZigZag 視覺化工具")
    print("="*60)
    
    try:
        # 讀取CSV檔案
        print(f"\n正在讀取: {args.input}")
        df = pd.read_csv(args.input)
        print(f"資料筆數: {len(df):,}")
        
        # 檢查必要的欄位
        required_cols = ['open', 'high', 'low', 'close', 'zigzag', 'swing_type']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise Exception(f"資料缺少必要欄位: {missing_cols}")
        
        # 繪製圖表
        plot_zigzag(
            df=df,
            n_bars=args.bars,
            figsize=(args.width, args.height),
            output_path=args.output,
            dpi=args.dpi
        )
        
        print("\n" + "="*60)
        print("完成")
        print("="*60)
        
    except FileNotFoundError:
        print(f"\n錯誤: 找不到檔案 '{args.input}'")
        print("請先執行 test_zigzag.py 生成結果檔案")
        sys.exit(1)
    except Exception as e:
        print(f"\n錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
