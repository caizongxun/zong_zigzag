#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZigZag v1.2.0 - 完整訓練管道

使用方式：
    python train_complete.py

或在 Colab 中：
    !python /content/zong_zigzag/zigzag_v120_system/train_complete.py
"""

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# 確保模組可導入
sys.path.insert(0, '/content/zong_zigzag')

from zigzag_v120_system.data_loader import HuggingFaceDataLoader, DataProcessor
from zigzag_v120_system.feature_engineering import FeatureBuilder
from zigzag_v120_system.strategy_executor import StrategyExecutor


class ImprovedLSTMPredictor(nn.Module):
    """改進版 LSTM 預測模型"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        x = self.fc1(last_hidden)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def create_sequences(X, y, seq_length):
    """創建時間序列數據"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)


def main():
    print("\n" + "="*70)
    print("ZigZag v1.2.0 量化交易系統 - 完整訓練管道")
    print("="*70)
    
    # ========== 第一步：初始化組件 ==========
    print("\n[步驟 1/7] 初始化核心組件...")
    print("-" * 70)
    
    loader = HuggingFaceDataLoader(use_huggingface=True, local_cache=True)
    processor = DataProcessor()
    feature_builder = FeatureBuilder()
    executor = StrategyExecutor(initial_capital=10000, risk_per_trade=0.02)
    
    print("✓ 數據加載器已初始化")
    print("✓ 數據處理器已初始化")
    print("✓ 特徵構建器已初始化")
    print("✓ 策略執行器已初始化")
    
    # ========== 第二步：加載數據 ==========
    print("\n[步驟 2/7] 加載 BTCUSDT 15分鐘K線數據...")
    print("-" * 70)
    
    df = loader.load_klines(symbol='BTCUSDT', timeframe='15m')
    
    if df.empty:
        print("✗ 數據加載失敗")
        return
    
    print(f"✓ 成功加載 {df.shape[0]} 行數據")
    print(f"✓ 列名: {list(df.columns)[:5]}... (共 {len(df.columns)} 列)")
    
    # ========== 第三步：數據清理 ==========
    print("\n[步驟 3/7] 清理數據...")
    print("-" * 70)
    
    df_clean = processor.clean_data(df, method='forward_fill')
    print(f"✓ 清理完成: {df_clean.shape[0]} 行數據保留")
    
    # ========== 第四步：特徵工程 ==========
    print("\n[步驟 4/7] 構建特徵矩陣...")
    print("-" * 70)
    
    features = feature_builder.build_feature_matrix(df_clean)
    print(f"✓ 特徵構建完成: {features.shape[1]} 個特徵")
    
    # ========== 第五步：準備訓練數據 ==========
    print("\n[步驟 5/7] 準備訓練數據...")
    print("-" * 70)
    
    # 分割訓練/測試
    train_size = int(len(features) * 0.8)
    X_train = features.iloc[:train_size].values
    X_test = features.iloc[train_size:].values
    y_train = df_clean['close'].iloc[:train_size].values
    y_test = df_clean['close'].iloc[train_size:].values
    
    print(f"✓ 訓練數據: {X_train.shape}")
    print(f"✓ 測試數據: {X_test.shape}")
    
    # 創建序列
    sequence_length = 60
    print(f"\n正在創建 {sequence_length} 步長的時間序列...")
    
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)
    
    print(f"✓ 訓練序列: {X_train_seq.shape}")
    print(f"✓ 測試序列: {X_test_seq.shape}")
    
    # 數據標準化（關鍵！）
    print("\n正在標準化數據...")
    scaler = StandardScaler()
    
    X_train_flat = X_train_seq.reshape(-1, X_train_seq.shape[-1])
    X_train_flat_scaled = scaler.fit_transform(X_train_flat)
    X_train_seq_scaled = X_train_flat_scaled.reshape(X_train_seq.shape)
    
    X_test_flat = X_test_seq.reshape(-1, X_test_seq.shape[-1])
    X_test_flat_scaled = scaler.transform(X_test_flat)
    X_test_seq_scaled = X_test_flat_scaled.reshape(X_test_seq.shape)
    
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train_seq.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test_seq.reshape(-1, 1)).flatten()
    
    print("✓ 數據標準化完成")
    print(f"  - X 均值: {X_train_seq_scaled.mean():.6f}")
    print(f"  - X 方差: {X_train_seq_scaled.std():.6f}")
    
    # 轉換為 PyTorch 張量
    X_train_tensor = torch.FloatTensor(X_train_seq_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_seq_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    # 建立 DataLoader
    batch_size = 64
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"✓ DataLoader 已準備: batch_size={batch_size}")
    
    # ========== 第六步：模型訓練 ==========
    print("\n[步驟 6/7] 訓練 LSTM 模型...")
    print("-" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"設備: {device}")
    
    # 初始化模型
    input_dim = X_train_seq_scaled.shape[2]
    lstm_model = ImprovedLSTMPredictor(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2
    ).to(device)
    
    print(f"✓ LSTM 模型已初始化 (輸入維度: {input_dim})")
    
    # 損失函數和優化器
    criterion = torch.nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(lstm_model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=False
    )
    
    print("✓ 優化器: AdamW")
    print("✓ 損失函數: HuberLoss")
    
    # 訓練
    num_epochs = 5
    print(f"\n正在訓練 {num_epochs} 個 epoch...")
    print("-" * 70)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 訓練模式
        lstm_model.train()
        epoch_loss = 0
        batch_count = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            outputs = lstm_model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            
            if torch.isnan(loss):
                print(f"警告: NaN loss 在 epoch {epoch+1}")
                continue
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_train_loss = epoch_loss / batch_count
        train_losses.append(avg_train_loss)
        
        # 驗證
        lstm_model.eval()
        with torch.no_grad():
            val_outputs = lstm_model(X_test_tensor.to(device))
            val_loss = criterion(val_outputs.squeeze(), y_test_tensor.to(device)).item()
        
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")
    
    print("-" * 70)
    print("✓ 訓練完成！")
    
    # ========== 第七步：模型評估與保存 ==========
    print("\n[步驟 7/7] 模型評估與保存...")
    print("-" * 70)
    
    lstm_model.eval()
    with torch.no_grad():
        train_pred = lstm_model(X_train_tensor.to(device)).squeeze()
        test_pred = lstm_model(X_test_tensor.to(device)).squeeze()
        
        train_mse = torch.mean((train_pred - y_train_tensor.to(device))**2).item()
        test_mse = torch.mean((test_pred - y_test_tensor.to(device))**2).item()
    
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    
    print(f"\n訓練集 RMSE: {train_rmse:.6f}")
    print(f"測試集 RMSE:  {test_rmse:.6f}")
    print(f"\n相對誤差: {(test_rmse/train_rmse - 1)*100:.2f}%")
    
    # 保存模型
    model_path = '/content/zong_zigzag/zigzag_v120_system/models/lstm_model_complete.pth'
    torch.save({
        'model_state_dict': lstm_model.state_dict(),
        'scaler': scaler,
        'y_scaler': y_scaler,
        'input_dim': input_dim,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse
    }, model_path)
    
    print(f"\n✓ 模型已保存: {model_path}")
    
    print("\n" + "="*70)
    print("訓練管道完成！")
    print("="*70)
    print(f"\n總結:")
    print(f"  - 訓練樣本: {X_train_seq.shape[0]:,} 個")
    print(f"  - 測試樣本: {X_test_seq.shape[0]:,} 個")
    print(f"  - 序列長度: {sequence_length}")
    print(f"  - 特徵維度: {input_dim}")
    print(f"  - 訓練 RMSE: {train_rmse:.6f}")
    print(f"  - 測試 RMSE:  {test_rmse:.6f}")
    print(f"  - 模型位置: models/lstm_model_complete.pth")
    print("\n下一步: 使用訓練好的模型進行推理和交易信號生成")
    print("="*70 + "\n")
    
    return lstm_model, scaler, y_scaler, X_test_seq_scaled, y_test_scaled


if __name__ == '__main__':
    main()
