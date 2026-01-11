#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZigZag 量化系統 v1.2.0 - 模組架構模組
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
import torch
import torch.nn as nn
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')


class LSTMPredictor(nn.Module):
    """
    LSTM 運動語家模型
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, 
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, 512)
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        x = self.fc1(last_hidden)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DualLayerDefenseSystem:
    """
    雙層防御系統
    """
    
    def __init__(self, lstm_model: LSTMPredictor, xgb_model=None):
        self.lstm_model = lstm_model
        self.xgb_model = xgb_model
    
    def generate_signal(self, lstm_pred: int, xgb_pred: int,
                       xgb_proba: np.ndarray) -> Tuple[int, float]:
        """
        綜合檢死兩層的位置
        """
        if lstm_pred == xgb_pred and np.max(xgb_proba) > 0.7:
            confidence = np.max(xgb_proba)
            signal = lstm_pred
        elif lstm_pred != xgb_pred:
            if np.max(xgb_proba) > 0.6:
                confidence = np.max(xgb_proba)
                signal = xgb_pred
            else:
                confidence = 0.5
                signal = 1
        else:
            confidence = np.max(xgb_proba)
            signal = xgb_pred
        
        return signal, confidence


if __name__ == "__main__":
    print("ZigZag 量化系統 v1.2.0 - 模組架構模組")
    print("模組已準備，何時開始訓練...")
