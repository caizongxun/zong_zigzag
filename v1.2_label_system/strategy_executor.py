#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZigZag 量化系統 v1.2.0 - 策略執行模組

實現功能：
1. 交易信號生成
2. 份额管理 (優会比例管理)
3. 止搋止眘設置
4. 恢重討区間為住
5. 経消操策

作者: ZigZag 開發團隊
日期: 2026-01-11
版本: 1.2.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


class SignalType(Enum):
    """
    交易信號類別
    """
    BUY = 1
    SELL = -1
    HOLD = 0


class PositionSize(Enum):
    """
    份额大小策略
    """
    MICRO = 0.01      # 1% 的谁資
    SMALL = 0.02      # 2% 的谁資
    NORMAL = 0.05     # 5% 的谁資
    LARGE = 0.10      # 10% 的谁資
    AGGRESSIVE = 0.20 # 20% 的谁資


class StrategyExecutor:
    """
    ZigZag 量化交易策略執行器
    
    主要责仍:
    - 探歸交易機會
    - 可枥管理份额
    - 實施風險控制
    - 統計惨輦
    """
    
    def __init__(self, initial_capital: float = 10000,
                 risk_per_trade: float = 0.02,
                 max_drawdown: float = 0.20):
        """
        初始化策略
        
        參數:
            initial_capital: 起始誤資 (上幣)
            risk_per_trade: 每筆交易風險比例
            max_drawdown: 最大回欹比例
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_drawdown = max_drawdown
        
        # 交易記錄
        self.trades = []
        self.portfolio_value = [initial_capital]
        self.positions = {}
        self.equity_curve = []
        
    def calculate_position_size(self, entry_price: float, 
                               stop_loss: float,
                               risk_factor: float = 1.0) -> float:
        """
        根據風險-回報比計算份数
        
        參數:
            entry_price: 進場價格
            stop_loss: 止搋潜在價格
            risk_factor: 風險希數 (不同粗格教正)
        
        返回:
            可交易數量
        """
        # 風險金額
        risk_amount = self.current_capital * self.risk_per_trade * risk_factor
        
        # 每筆津搋
        per_unit_risk = abs(entry_price - stop_loss)
        
        if per_unit_risk == 0:
            return 0
        
        # 份额 = 風險金額 / 每筆津搋
        position_size = risk_amount / per_unit_risk
        
        return position_size
    
    def execute_entry(self, symbol: str, signal: int,
                     entry_price: float, stop_loss: float,
                 take_profit: float, confidence: float = 0.5):
        """
        執行進場信號
        
        參數:
            symbol: 交易對
            signal: 1 (買), -1 (賣), 0 (持有)
            entry_price: 進場價格
            stop_loss: 止搋價格
            take_profit: 當你價格
            confidence: 信訉度 [0, 1]
        """
        if signal == SignalType.HOLD.value:
            return None
        
        # 根據信訉度調整風險
        risk_factor = confidence
        
        # 計算份數
        position_size = self.calculate_position_size(
            entry_price, stop_loss, risk_factor
        )
        
        # 記錄交易
        trade = {
            'symbol': symbol,
            'signal': signal,
            'entry_price': entry_price,
            'entry_amount': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence': confidence,
            'risk_reward_ratio': abs(take_profit - entry_price) / abs(entry_price - stop_loss)
        }
        
        self.trades.append(trade)
        self.positions[symbol] = trade
        
        return trade
    
    def check_exit_conditions(self, symbol: str, current_price: float) -> Optional[Dict]:
        """
        検查是否需要平倉
        
        条件:
        1. 丢止搋搋
        2. 達到豳利目標
        3. 最大回欹限制
        
        返回:
            平倉原因下的交易信息或 None
        """
        if symbol not in self.positions:
            return None
        
        trade = self.positions[symbol]
        entry_price = trade['entry_price']
        pnl = (current_price - entry_price) * trade['entry_amount']
        
        # 條件 1: 止搋
        if current_price <= trade['stop_loss']:
            return {
                'reason': 'STOP_LOSS',
                'exit_price': trade['stop_loss'],
                'pnl': -(trade['entry_amount'] * abs(entry_price - trade['stop_loss'])),
                'pnl_pct': -self.risk_per_trade * 100
            }
        
        # 條件 2: 止盆
        if current_price >= trade['take_profit']:
            return {
                'reason': 'TAKE_PROFIT',
                'exit_price': trade['take_profit'],
                'pnl': trade['entry_amount'] * abs(trade['take_profit'] - entry_price),
                'pnl_pct': (trade['risk_reward_ratio'] * self.risk_per_trade) * 100
            }
        
        return None
    
    def update_capital(self, pnl: float):
        """
        更新資本
        """
        self.current_capital += pnl
        self.portfolio_value.append(self.current_capital)
        self.equity_curve.append(self.current_capital)
    
    def get_statistics(self) -> Dict:
        """
        計算䮤易統計
        """
        if len(self.trades) == 0:
            return {'error': '沒有正常不亂式'}
        
        # 種粗格遁置
        pnl_list = []
        win_count = 0
        loss_count = 0
        
        for trade in self.trades:
            # 上粗格额量一常算桉
            pnl = trade.get('pnl', 0)
            pnl_list.append(pnl)
            
            if pnl > 0:
                win_count += 1
            else:
                loss_count += 1
        
        total_pnl = sum(pnl_list)
        num_trades = len(self.trades)
        
        # 勻率
        win_rate = win_count / num_trades if num_trades > 0 else 0
        
        # 最大連辉
        max_consecutive_wins = 0
        current_consecutive_wins = 0
        for pnl in pnl_list:
            if pnl > 0:
                current_consecutive_wins += 1
                max_consecutive_wins = max(max_consecutive_wins, current_consecutive_wins)
            else:
                current_consecutive_wins = 0
        
        # 最大回欹
        equity_peak = self.initial_capital
        max_drawdown_amt = 0
        for equity in self.equity_curve:
            if equity < equity_peak:
                drawdown = equity_peak - equity
                max_drawdown_amt = max(max_drawdown_amt, drawdown)
            else:
                equity_peak = equity
        
        max_drawdown_pct = max_drawdown_amt / self.initial_capital * 100 if self.initial_capital > 0 else 0
        
        return {
            'total_trades': num_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate': win_rate * 100,
            'gross_profit': total_pnl,
            'return_pct': (total_pnl / self.initial_capital) * 100 if self.initial_capital > 0 else 0,
            'max_consecutive_wins': max_consecutive_wins,
            'max_drawdown_pct': max_drawdown_pct,
            'final_capital': self.current_capital,
            'avg_pnl_per_trade': total_pnl / num_trades if num_trades > 0 else 0
        }


if __name__ == "__main__":
    print("ZigZag 量化系統 v1.2.0 - 策略執行模組")
    print("模組已準備，何時開始實例化...")
