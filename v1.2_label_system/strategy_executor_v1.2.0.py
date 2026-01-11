#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZigZag 量化系統 v1.2.0 - 策略執行層

實現功能：
1. 入場點位計算 (基於 Fibonacci + 訂單簿)
2. 動態止損/止盈設置
3. 訂單簿深度分析
4. 完整的交易信號輸出
5. 實時監控和調整

作者: ZigZag 開發團隊
日期: 2026-01-11
版本: 1.2.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class OrderBookData:
    """
    訂單簿數據結構
    """
    bids: List[Dict]  # [{"price": float, "quantity": float}, ...]
    asks: List[Dict]  # [{"price": float, "quantity": float}, ...]
    
    @property
    def best_bid(self) -> float:
        return self.bids[0]['price'] if self.bids else 0
    
    @property
    def best_ask(self) -> float:
        return self.asks[0]['price'] if self.asks else 0
    
    @property
    def mid_price(self) -> float:
        if self.best_bid > 0 and self.best_ask > 0:
            return (self.best_bid + self.best_ask) / 2
        return 0


class EntryPointCalculator:
    """
    入場點位計算器
    
    根據 Fibonacci 回調、訂單簿深度、技術指標計算最优入場點
    """
    
    @staticmethod
    def find_high_volume_nodes(ohlcv_data: pd.DataFrame,
                               window: int = 100,
                               bins: int = 20) -> Dict:
        """
        計算過去 N 根 K 線的成交量最集中的價格區間
        
        參數:
            ohlcv_data: OHLCV DataFrame
            window: 計算窗口 (K 線數)
            bins: 价格倓整斎數
        
        返回:
            高成交量區間
        """
        recent_data = ohlcv_data.tail(window)
        
        # 統計价格標箄内的成交量
        high_price = recent_data['high'].max()
        low_price = recent_data['low'].min()
        
        # 創建價格箄位 (bins)
        bins_array = np.linspace(low_price, high_price, bins + 1)
        volume_per_bin = []
        
        for i in range(len(bins_array) - 1):
            bin_low = bins_array[i]
            bin_high = bins_array[i + 1]
            
            # 計算該对收磊价格範围内的成交量
            in_range = (
                (recent_data['high'] >= bin_low) & 
                (recent_data['high'] <= bin_high) |
                (recent_data['low'] >= bin_low) & 
                (recent_data['low'] <= bin_high)
            )
            bin_volume = recent_data[in_range]['volume'].sum()
            
            volume_per_bin.append({
                'price_level': (bin_low + bin_high) / 2,
                'volume': bin_volume,
                'pct': (bin_volume / recent_data['volume'].sum() * 100) if recent_data['volume'].sum() > 0 else 0
            })
        
        # 救筡佋量最大的对
        poc = max(volume_per_bin, key=lambda x: x['volume'])
        
        return {
            'high_volume_nodes': sorted(volume_per_bin, key=lambda x: x['volume'], reverse=True)[:5],
            'poc': poc['price_level'],
            'poc_volume': poc['volume']
        }
    
    @staticmethod
    def calculate_fibonacci_levels(high: float, low: float) -> Dict[float, float]:
        """
        計算斐波那契回調水平
        
        參數:
            high: 過去 N 天的最高價
            low: 過去 N 天的最低價
        
        返回:
            Fibonacci 水平 {level: price}
        """
        amplitude = high - low
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.705, 0.786]
        
        return {
            level: low + amplitude * level 
            for level in fib_levels
        }
    
    @staticmethod
    def find_nearest_support_resistance(current_price: float,
                                       fibonacci_levels: Dict[float, float],
                                       direction: str = 'UP') -> Tuple[float, float]:
        """
        找出最接近當前價格的布林陪是支撐或阻力
        
        參數:
            current_price: 當前價格
            fibonacci_levels: 斐波那契水平字典
            direction: 'UP' (看漲) 或 'DOWN' (看跌)
        
        返回:
            (primary_level, secondary_level)
        """
        if direction == 'UP':
            # 看漲：找下方最接近的支撐
            support_levels = [p for p in fibonacci_levels.values() if p < current_price]
            primary = max(support_levels) if support_levels else current_price * 0.98
            
            secondary_levels = [p for p in fibonacci_levels.values() if p < primary]
            secondary = max(secondary_levels) if secondary_levels else current_price * 0.96
        else:
            # 看跌：找上方最接近的阻力
            resistance_levels = [p for p in fibonacci_levels.values() if p > current_price]
            primary = min(resistance_levels) if resistance_levels else current_price * 1.02
            
            secondary_levels = [p for p in fibonacci_levels.values() if p > primary]
            secondary = min(secondary_levels) if secondary_levels else current_price * 1.04
        
        return primary, secondary
    
    @staticmethod
    def calculate_entry_price(direction: str,
                             current_price: float,
                             fibonacci_levels: Dict[float, float],
                             poc: float,
                             confidence: float,
                             order_book: Optional[OrderBookData] = None) -> float:
        """
        計算最优入場價
        
        參数:
            direction: 'UP' 或 'DOWN'
            current_price: 當前價格
            fibonacci_levels: Fib 水平
            poc: 大成交量最集中的價格
            confidence: 模型置信度 (0-1)
            order_book: 訂單簿 (可選)
        
        返回:
            入場價格
        """
        primary, secondary = EntryPointCalculator.find_nearest_support_resistance(
            current_price, fibonacci_levels, direction
        )
        
        # 根據置信度選拉入場點
        if confidence >= 0.85:
            # 高置信度: 使用主支擲/阻力
            entry = primary
        elif confidence >= 0.75:
            # 中置信度: 折中方案
            entry = (primary + secondary) / 2
        else:
            # 低置信度: 使用 POC 或當前價格的中間
            entry = (poc + current_price) / 2
        
        # 确保方向正確
        if direction == 'UP':
            # 看漲時，入場價应当低于當前價格
            entry = min(entry, current_price * 0.999)
        else:
            # 看跌時，入場價应當高于當前價格
            entry = max(entry, current_price * 1.001)
        
        return entry


class StopLossAndTakeProfitCalculator:
    """
    止損止盈設置計算器
    """
    
    @staticmethod
    def calculate_stop_loss(direction: str,
                           entry_price: float,
                           atr: float,
                           atr_multiplier: float = 1.5) -> float:
        """
        基於 ATR 設置失損
        
        參数:
            direction: 'UP' 或 'DOWN'
            entry_price: 入場價
            atr: 平均真實波幅 (ATR)
            atr_multiplier: ATR 倍數
        
        返回:
            止損價
        """
        if direction == 'UP':
            # 看漲：止損在入場下方
            stop_loss = entry_price - atr * atr_multiplier
        else:
            # 看跌：止損在入場上方
            stop_loss = entry_price + atr * atr_multiplier
        
        return stop_loss
    
    @staticmethod
    def calculate_take_profit(direction: str,
                             entry_price: float,
                             atr: float,
                             risk_reward_ratio: float = 2.0,
                             atr_multiplier: float = 1.5) -> float:
        """
        基於風險/收益比設置止盈
        
        參数:
            direction: 'UP' 或 'DOWN'
            entry_price: 入場價
            atr: ATR
            risk_reward_ratio: 風險/收益比 (e.g., 2.0 = 1:2)
            atr_multiplier: ATR 倍數
        
        返回:
            止盈價
        """
        risk = atr * atr_multiplier
        reward = risk * risk_reward_ratio
        
        if direction == 'UP':
            take_profit = entry_price + reward
        else:
            take_profit = entry_price - reward
        
        return take_profit


class SignalFilterEngine:
    """
    交易信號過濾引擎
    
    遯门低質量的交易。
    """
    
    @staticmethod
    def should_trade(lstm_confidence: float,
                    xgb_approval: bool,
                    current_volatility: float,
                    average_volatility: float,
                    current_volume: float,
                    average_volume: float,
                    current_price: float,
                    entry_price: float,
                    direction: str,
                    **kwargs) -> Tuple[bool, str]:
        """
        可综合過濾是否策略应該執行
        
        參数:
            lstm_confidence: LSTM 置信度
            xgb_approval: XGBoost 應彔
            current_volatility: 當前波動率
            average_volatility: 平均波動率
            current_volume: 當前成交量
            average_volume: 平均成交量
            current_price: 當前價格
            entry_price: 入場價
            direction: 'UP' 或 'DOWN'
        
        返回:
            (should_trade: bool, reason: str)
        """
        # 1. 梨信度過低
        if lstm_confidence < 0.60:
            return False, f"模型置信度不足 ({lstm_confidence:.1%} < 60%)"
        
        # 2. XGBoost 未批准
        if not xgb_approval:
            return False, "元模型不超渊訂先佋幫
        
        # 3. 波動率過高
        volatility_threshold = average_volatility * 2.0  # 2 sigma
        if current_volatility > volatility_threshold:
            return False, f"波動率過高 ({current_volatility:.2f} > {volatility_threshold:.2f})"
        
        # 4. 成交量不足
        if current_volume < average_volume * 0.5:
            return False, f"成交量不足 ({current_volume:.0f} < {average_volume * 0.5:.0f})"
        
        # 5. 價格偏離過遠
        if direction == 'UP' and current_price > entry_price * 1.05:
            return False, f"看漲信號但價格已上漲 ({current_price:.2f} > {entry_price * 1.05:.2f})"
        
        if direction == 'DOWN' and current_price < entry_price * 0.95:
            return False, f"看跌信號但價格已下跌 ({current_price:.2f} < {entry_price * 0.95:.2f})"
        
        return True, "PASS"


class TradingSignalFormatter:
    """
    交易信號格式化器
    
    把模型預測結果格式化成可読的交易信號
    """
    
    @staticmethod
    def format_html_signal(symbol: str,
                          timeframe: str,
                          direction: str,
                          entry_price: float,
                          stop_loss: float,
                          take_profit: float,
                          confidence: float,
                          lstm_confidence: float,
                          xgb_confidence: float,
                          reason: str,
                          timestamp: datetime) -> str:
        """
        產生 HTML 格式的交易信號
        """
        direction_emoji = '' if direction == 'UP' else ''
        risk_reward_ratio = abs(take_profit - entry_price) / abs(entry_price - stop_loss)
        pnl_potential = ((take_profit - entry_price) / entry_price) * 100
        
        html = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; background: #f5f5f5; }}
                .signal-container {{ max-width: 600px; margin: 20px auto; background: white; padding: 20px; border-radius: 8px; }}
                .header {{ text-align: center; border-bottom: 2px solid #2196F3; padding-bottom: 10px; }}
                .title {{ font-size: 24px; font-weight: bold; }}
                .info-row {{ display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #eee; }}
                .label {{ color: #666; font-weight: 500; }}
                .value {{ color: #333; font-weight: bold; }}
                .price {{ color: #2196F3; font-size: 16px; }}
                .up {{ color: #4CAF50; }}
                .down {{ color: #f44336; }}
                .confidence {{ width: 100%; height: 20px; background: #eee; border-radius: 4px; overflow: hidden; margin: 5px 0; }}
                .confidence-bar {{ height: 100%; background: linear-gradient(to right, #f44336, #FFC107, #4CAF50); }}
                .footer {{ text-align: center; padding-top: 10px; font-size: 12px; color: #999; }}
            </style>
        </head>
        <body>
        <div class="signal-container">
            <div class="header">
                <div class="title">{direction_emoji} {symbol} {timeframe}</div>
                <div class="{direction.lower()}">信號: {direction}</div>
            </div>
            
            <div class="info-row">
                <span class="label">兩仪件位：</span>
                <span class="value price">{entry_price:.2f}</span>
            </div>
            
            <div class="info-row">
                <span class="label">止損：</span>
                <span class="value price">{stop_loss:.2f}</span>
            </div>
            
            <div class="info-row">
                <span class="label">止盈：</span>
                <span class="value price">{take_profit:.2f}</span>
            </div>
            
            <div class="info-row">
                <span class="label">風險/收益：</span>
                <span class="value">1 : {risk_reward_ratio:.2f}</span>
            </div>
            
            <div class="info-row">
                <span class="label">準增止盈售希望：</span>
                <span class="value">{pnl_potential:.2f}%</span>
            </div>
            
            <div class="info-row">
                <span class="label">總置信度：</span>
                <span class="value">{confidence:.1%}</span>
            </div>
            
            <div class="confidence">
                <div class="confidence-bar" style="width: {confidence*100}%"></div>
            </div>
            
            <div class="info-row">
                <span class="label">LSTM 模型：</span>
                <span class="value">{lstm_confidence:.1%}</span>
            </div>
            
            <div class="info-row">
                <span class="label">XGBoost 模型：</span>
                <span class="value">{xgb_confidence:.1%}</span>
            </div>
            
            <div class="info-row">
                <span class="label">理由：</span>
                <span class="value">{reason}</span>
            </div>
            
            <div class="footer">
                時間：{timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
        </body>
        </html>
        """
        
        return html
    
    @staticmethod
    def format_text_signal(symbol: str,
                          timeframe: str,
                          direction: str,
                          entry_price: float,
                          stop_loss: float,
                          take_profit: float,
                          confidence: float,
                          lstm_confidence: float,
                          xgb_confidence: float,
                          reason: str,
                          timestamp: datetime) -> str:
        """
        產生文本格式的交易信號
        """
        risk_reward_ratio = abs(take_profit - entry_price) / abs(entry_price - stop_loss)
        pnl_potential = ((take_profit - entry_price) / entry_price) * 100
        
        signal = f"""
=====================================================
交易信號 - {symbol} {timeframe}
=====================================================

方向：        {direction} (看漲 / 看跌)
推薦入場價:  {entry_price:.2f}
止損價:      {stop_loss:.2f}
止盈價:      {take_profit:.2f}

風險:       {abs(entry_price - stop_loss):.2f}
收益:       {abs(take_profit - entry_price):.2f}
風險/收益:  1 : {risk_reward_ratio:.2f}
策略收益希望： {pnl_potential:.2f}%

模型置信度:  {confidence:.1%}
  - LSTM:    {lstm_confidence:.1%}
  - XGBoost: {xgb_confidence:.1%}

理由:        {reason}
時間:        {timestamp.strftime('%Y-%m-%d %H:%M:%S')}

=====================================================
        """
        
        return signal
    
    @staticmethod
    def format_json_signal(symbol: str,
                          timeframe: str,
                          direction: str,
                          entry_price: float,
                          stop_loss: float,
                          take_profit: float,
                          confidence: float,
                          lstm_confidence: float,
                          xgb_confidence: float,
                          reason: str,
                          timestamp: datetime) -> Dict:
        """
        產生 JSON 格式的交易信號
        """
        risk_reward_ratio = abs(take_profit - entry_price) / abs(entry_price - stop_loss)
        pnl_potential = ((take_profit - entry_price) / entry_price) * 100
        
        return {
            'timestamp': timestamp.isoformat(),
            'symbol': symbol,
            'timeframe': timeframe,
            'direction': direction,
            'entry_price': float(entry_price),
            'stop_loss': float(stop_loss),
            'take_profit': float(take_profit),
            'confidence': {
                'total': float(confidence),
                'lstm': float(lstm_confidence),
                'xgb': float(xgb_confidence)
            },
            'position_metrics': {
                'risk': float(abs(entry_price - stop_loss)),
                'reward': float(abs(take_profit - entry_price)),
                'risk_reward_ratio': float(risk_reward_ratio),
                'pnl_potential_pct': float(pnl_potential)
            },
            'reason': reason
        }


class StrategyExecutor:
    """
    完整策略執行系統
    
    整合前面所有素擏，將模型預測轉变為具體交易信號
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.entry_calculator = EntryPointCalculator()
        self.sl_tp_calculator = StopLossAndTakeProfitCalculator()
        self.filter_engine = SignalFilterEngine()
        self.formatter = TradingSignalFormatter()
        self.config = config or {}
        self.open_positions = []
    
    def execute_strategy(self,
                        symbol: str,
                        timeframe: str,
                        model_prediction: Dict,
                        market_data: pd.DataFrame,
                        order_book: Optional[OrderBookData] = None,
                        format_type: str = 'text') -> Dict:
        """
        執行完整的策略流程
        
        參数:
            symbol: 交易对 (e.g., 'BTCUSDT')
            timeframe: 時間框架 (e.g., '15m')
            model_prediction: 模型預測結果字典
            market_data: 市場數據 DataFrame (OHLCV)
            order_book: 訂單簿數據 (可選)
            format_type: 輸出格式 ('text', 'json', 'html')
        
        返回:
            交易信號字典
        """
        try:
            # 1. 提取市場數據
            current_price = market_data['close'].iloc[-1]
            current_high = market_data['high'].tail(100).max()
            current_low = market_data['low'].tail(100).min()
            current_volume = market_data['volume'].iloc[-1]
            average_volume = market_data['volume'].tail(20).mean()
            atr = market_data['atr_14'].iloc[-1] if 'atr_14' in market_data.columns else 100
            current_volatility = market_data['tr'].tail(14).mean() if 'tr' in market_data.columns else atr
            average_volatility = market_data['tr'].tail(50).mean() if 'tr' in market_data.columns else atr
            
            # 2. 計算 Fibonacci 水平
            fibonacci_levels = self.entry_calculator.calculate_fibonacci_levels(
                current_high, current_low
            )
            
            # 3. 計算主成交量最集中位
            volume_analysis = self.entry_calculator.find_high_volume_nodes(
                market_data, window=100, bins=20
            )
            poc = volume_analysis['poc']
            
            # 4. 提取模型預測結果
            direction = 'UP' if model_prediction['lstm_pred_class'][0] == 2 else 'DOWN'
            lstm_confidence = model_prediction['lstm_confidence'][0]
            xgb_confidence = model_prediction['xgb_confidence'][0]
            final_confidence = model_prediction['final_confidence'][0]
            
            # 5. 計算入場點位
            entry_price = self.entry_calculator.calculate_entry_price(
                direction, current_price, fibonacci_levels, poc, final_confidence, order_book
            )
            
            # 6. 計算止損和止盈
            stop_loss = self.sl_tp_calculator.calculate_stop_loss(
                direction, entry_price, atr, atr_multiplier=1.5
            )
            take_profit = self.sl_tp_calculator.calculate_take_profit(
                direction, entry_price, atr, risk_reward_ratio=2.0, atr_multiplier=1.5
            )
            
            # 7. 過濾檢查
            should_trade, filter_reason = self.filter_engine.should_trade(
                lstm_confidence=lstm_confidence,
                xgb_approval=model_prediction['xgb_approval'][0],
                current_volatility=current_volatility,
                average_volatility=average_volatility,
                current_volume=current_volume,
                average_volume=average_volume,
                current_price=current_price,
                entry_price=entry_price,
                direction=direction
            )
            
            if not should_trade:
                return {
                    'status': 'FILTERED_OUT',
                    'reason': filter_reason,
                    'timestamp': datetime.now()
                }
            
            # 8. 格式化輸出
            reason = f"基於 Fibonacci {fibonacci_levels} 加權 + LSTM {direction} 信號"
            timestamp = datetime.now()
            
            if format_type == 'json':
                signal = self.formatter.format_json_signal(
                    symbol, timeframe, direction, entry_price, stop_loss, take_profit,
                    final_confidence, lstm_confidence, xgb_confidence, reason, timestamp
                )
            elif format_type == 'html':
                signal = self.formatter.format_html_signal(
                    symbol, timeframe, direction, entry_price, stop_loss, take_profit,
                    final_confidence, lstm_confidence, xgb_confidence, reason, timestamp
                )
            else:  # text
                signal = self.formatter.format_text_signal(
                    symbol, timeframe, direction, entry_price, stop_loss, take_profit,
                    final_confidence, lstm_confidence, xgb_confidence, reason, timestamp
                )
            
            return {
                'status': 'TRADING_SIGNAL',
                'signal': signal,
                'metadata': {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'direction': direction,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': final_confidence,
                    'timestamp': timestamp
                }
            }
        
        except Exception as e:
            return {
                'status': 'ERROR',
                'error': str(e),
                'timestamp': datetime.now()
            }


if __name__ == "__main__":
    print("ZigZag 量化系統 v1.2.0 - 策略執行層\n")
    print("執行層已準備，待整合不同素擏...")
