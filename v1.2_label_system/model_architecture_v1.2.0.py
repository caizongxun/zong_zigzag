#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZigZag 量化系統 v1.2.0 - 模型架構模塊

實現功能：
1. LSTM 初級模型 (包含 Attention 機制)
2. XGBoost 元標記模型
3. 自定義損失函數
4. 模型訓練和推理

作者: ZigZag 開發團隊
日期: 2026-01-11
版本: 1.2.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score, 
    precision_score, recall_score, roc_auc_score, roc_curve
)
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class AttentionLayer(nn.Module):
    """
    Attention 機制層
    
    計算時間步的權重，讓模型賽中波動輸入的重要部份
    """
    
    def __init__(self, hidden_size: int):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        參數:
            lstm_output: (batch_size, seq_length, hidden_size)
        
        返回:
            context: (batch_size, hidden_size) - 加權後的上下文向量
            weights: (batch_size, seq_length) - Attention 權重
        """
        # 計算 attention scores
        scores = self.attention(lstm_output)  # (batch_size, seq_length, 1)
        scores = scores.squeeze(-1)  # (batch_size, seq_length)
        
        # 應用 softmax 獲得權重
        weights = torch.softmax(scores, dim=1)  # (batch_size, seq_length)
        
        # 加權求和獲得上下文
        context = torch.bmm(weights.unsqueeze(1), lstm_output)  # (batch_size, 1, hidden_size)
        context = context.squeeze(1)  # (batch_size, hidden_size)
        
        return context, weights


class LSTMPrimaryModel(nn.Module):
    """
    LSTM 初級模型
    
    架構:
        Input -> Embedding -> LSTM1 -> Attention -> LSTM2 -> GlobalAvgPooling -> Dense -> Output
    """
    
    def __init__(self, 
                 n_features: int,
                 lstm_hidden_1: int = 128,
                 lstm_hidden_2: int = 64,
                 dense_hidden: int = 32,
                 dropout: float = 0.3,
                 n_classes: int = 3,
                 device: str = 'cpu'):
        super(LSTMPrimaryModel, self).__init__()
        
        self.n_features = n_features
        self.lstm_hidden_1 = lstm_hidden_1
        self.lstm_hidden_2 = lstm_hidden_2
        self.device = device
        
        # LSTM 層
        self.lstm1 = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_hidden_1,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        # Attention 機制
        self.attention = AttentionLayer(lstm_hidden_1)
        
        # 第二層 LSTM
        self.lstm2 = nn.LSTM(
            input_size=lstm_hidden_1,
            hidden_size=lstm_hidden_2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        # 全域平均池
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全連接層
        self.fc1 = nn.Linear(lstm_hidden_2, dense_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # 輸出層
        self.fc_out = nn.Linear(dense_hidden, n_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        參數:
            x: (batch_size, seq_length, n_features)
        
        返回:
            output: (batch_size, n_classes) - 標一化概率
        """
        # LSTM1
        lstm1_out, (h1, c1) = self.lstm1(x)
        
        # Attention
        context, _ = self.attention(lstm1_out)
        context = context.unsqueeze(1)  # (batch_size, 1, lstm_hidden_1)
        
        # LSTM2 使用 context 作為龍侀狀態
        lstm2_out, _ = self.lstm2(lstm1_out)
        
        # 全域平均池
        pooled = self.global_avg_pool(lstm2_out.transpose(1, 2))  # (batch_size, lstm_hidden_2, 1)
        pooled = pooled.squeeze(-1)  # (batch_size, lstm_hidden_2)
        
        # 全連接層
        fc1_out = self.fc1(pooled)
        fc1_out = self.relu(fc1_out)
        fc1_out = self.dropout(fc1_out)
        
        # 輸出
        logits = self.fc_out(fc1_out)
        output = self.softmax(logits)
        
        return output
    
    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        預測 (頻率)
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
        return output.cpu().numpy()


class MetaFeatureExtractor:
    """
    元批特徵提取器
    
    從 LSTM 輸出提取特徵供 XGBoost 使用
    """
    
    @staticmethod
    def extract_from_predictions(lstm_probs: np.ndarray,
                                market_features: pd.DataFrame) -> pd.DataFrame:
        """
        提取元批特徵
        
        參數:
            lstm_probs: LSTM 輸出 (shape: (n_samples, 3))
            market_features: 市場特徵 DataFrame
        
        返回:
            元批特徵 DataFrame
        """
        meta_features = pd.DataFrame(index=market_features.index)
        
        # LSTM 預測特徵
        meta_features['lstm_pred_down'] = lstm_probs[:, 0]
        meta_features['lstm_pred_flat'] = lstm_probs[:, 1]
        meta_features['lstm_pred_up'] = lstm_probs[:, 2]
        
        # 最高概率
        meta_features['lstm_pred_prob_max'] = np.max(lstm_probs, axis=1)
        
        # 預測類別
        meta_features['lstm_pred_class'] = np.argmax(lstm_probs, axis=1)
        
        # 不確定性 (entropy)
        entropy = -np.sum(lstm_probs * np.log(lstm_probs + 1e-8), axis=1)
        meta_features['lstm_pred_entropy'] = entropy
        
        # 非最大概率之和
        meta_features['lstm_pred_second_prob'] = np.sort(lstm_probs, axis=1)[:, -2]
        
        # 概率号的間隔
        meta_features['lstm_pred_prob_gap'] = (
            meta_features['lstm_pred_prob_max'] - 
            meta_features['lstm_pred_second_prob']
        )
        
        # 加入市場特徵
        if 'atr_14' in market_features.columns:
            meta_features['market_volatility'] = market_features['atr_14']
        
        if 'volume_ratio' in market_features.columns:
            meta_features['market_volume_ratio'] = market_features['volume_ratio']
        
        if 'volume' in market_features.columns:
            meta_features['market_volume'] = market_features['volume']
        
        # 填充缺失值
        meta_features = meta_features.fillna(meta_features.mean())
        
        return meta_features


class XGBoostMetaModel:
    """
    XGBoost 元模型
    
    判斷 LSTM 預測是否可信 (是否應該執行)
    """
    
    def __init__(self, params: Optional[Dict] = None):
        if params is None:
            self.params = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 1.0,
                'reg_lambda': 1.0,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'random_state': 42,
                'verbosity': 0
            }
        else:
            self.params = params
        
        self.model = None
        self.feature_names = None
    
    def train(self, 
              X_train: pd.DataFrame, 
              y_train: np.ndarray,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[np.ndarray] = None,
              early_stopping_rounds: int = 20) -> Dict:
        """
        訓練 XGBoost 模型
        
        參數:
            X_train: 訓練特徵
            y_train: 訓練標籤 (0 或 1)
            X_val: 驗證特徵 (可選)
            y_val: 驗證標籤 (可選)
            early_stopping_rounds: 提前停止室數
        
        返回:
            訓練索引
        """
        self.feature_names = X_train.columns.tolist()
        
        # 計算類別權重
        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        self.params['scale_pos_weight'] = scale_pos_weight
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            self.model = xgb.XGBClassifier(**self.params)
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )
        else:
            self.model = xgb.XGBClassifier(**self.params)
            self.model.fit(X_train, y_train, verbose=False)
        
        return {
            'status': 'trained',
            'n_features': len(self.feature_names),
            'scale_pos_weight': scale_pos_weight
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        預測（類別）
        """
        if self.model is None:
            raise ValueError("模型未訓練")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        預測（概率）
        """
        if self.model is None:
            raise ValueError("模型未訓練")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, top_n: int = 20) -> Dict:
        """
        獲取特徵重要性
        """
        if self.model is None:
            raise ValueError("模型未訓練")
        
        importances = self.model.feature_importances_
        top_indices = np.argsort(importances)[-top_n:][::-1]
        
        return {
            feature: importances[idx] 
            for feature, idx in zip(
                [self.feature_names[i] for i in top_indices],
                top_indices
            )
        }


class DualLayerMetaLabelingSystem:
    """
    雙層元標記系統
    
    第一層: LSTM 預測方向 (DOWN/FLAT/UP)
    第二層: XGBoost 判斷是否有效
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.lstm_model = None
        self.xgb_model = None
        self.meta_extractor = MetaFeatureExtractor()
        self.scaler = None
        self.history = {}
    
    def build_lstm_model(self, 
                        n_features: int,
                        seq_length: int,
                        lstm_hidden_1: int = 128,
                        lstm_hidden_2: int = 64,
                        dense_hidden: int = 32,
                        dropout: float = 0.3,
                        n_classes: int = 3) -> LSTMPrimaryModel:
        """
        構建 LSTM 模型
        """
        self.lstm_model = LSTMPrimaryModel(
            n_features=n_features,
            lstm_hidden_1=lstm_hidden_1,
            lstm_hidden_2=lstm_hidden_2,
            dense_hidden=dense_hidden,
            dropout=dropout,
            n_classes=n_classes,
            device=str(self.device)
        ).to(self.device)
        
        return self.lstm_model
    
    def train_lstm(self,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_val: Optional[np.ndarray] = None,
                   y_val: Optional[np.ndarray] = None,
                   epochs: int = 100,
                   batch_size: int = 64,
                   learning_rate: float = 0.001,
                   early_stopping_patience: int = 10) -> Dict:
        """
        訓練 LSTM 模型
        
        參數:
            X_train: (n_samples, seq_length, n_features)
            y_train: (n_samples,)
            X_val: 驗證數據 (可選)
            y_val: 驗證標籤 (可選)
        """
        if self.lstm_model is None:
            raise ValueError("請先按低 build_lstm_model")
        
        # 伊儫化數據
        X_train_tensor = torch.from_numpy(X_train).float().to(self.device)
        y_train_tensor = torch.from_numpy(y_train).long().to(self.device)
        
        # 訓練數據載入器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 損失函數和設定器
        # 計算類別權重
        class_weights = []
        for cls in range(3):
            count = (y_train == cls).sum()
            weight = len(y_train) / (3 * count) if count > 0 else 1.0
            class_weights.append(weight)
        
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(self.lstm_model.parameters(), lr=learning_rate)
        
        # 訓練誀
        best_val_loss = float('inf')
        patience_counter = 0
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_f1': [],
            'val_f1': []
        }
        
        for epoch in range(epochs):
            # 訓練
            self.lstm_model.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.lstm_model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                train_targets.extend(y_batch.cpu().numpy())
            
            train_loss /= len(train_loader)
            train_f1 = f1_score(train_targets, train_preds, average='weighted', zero_division=0)
            
            history['train_loss'].append(train_loss)
            history['train_f1'].append(train_f1)
            
            # 驗證
            if X_val is not None and y_val is not None:
                self.lstm_model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.from_numpy(X_val).float().to(self.device)
                    y_val_tensor = torch.from_numpy(y_val).long().to(self.device)
                    
                    val_outputs = self.lstm_model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                    
                    val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
                    val_f1 = f1_score(y_val, val_preds, average='weighted', zero_division=0)
                    
                    history['val_loss'].append(val_loss)
                    history['val_f1'].append(val_f1)
                    
                    # 提前停止
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            print(f"[LSTM] Epoch {epoch+1}: 提前停止")
                            break
            
            if (epoch + 1) % 10 == 0:
                print(f"[LSTM] Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, F1: {train_f1:.4f}")
        
        self.history = history
        return history
    
    def predict_with_confidence(self, 
                               X: np.ndarray,
                               meta_features: pd.DataFrame,
                               confidence_threshold: float = 0.75) -> Dict:
        """
        綜合預測 (LSTM + XGBoost)
        
        參數:
            X: (n_samples, seq_length, n_features)
            meta_features: 市場特徵
            confidence_threshold: 置信度閾值
        
        返回:
            預測結果字典
        """
        if self.lstm_model is None or self.xgb_model is None:
            raise ValueError("模型未完成訓練")
        
        # 第一層: LSTM 預測
        self.lstm_model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            lstm_probs = self.lstm_model(X_tensor).cpu().numpy()
        
        lstm_pred_class = np.argmax(lstm_probs, axis=1)
        lstm_confidence = np.max(lstm_probs, axis=1)
        
        # 提取元批特徵
        meta_X = self.meta_extractor.extract_from_predictions(lstm_probs, meta_features)
        
        # 第二層: XGBoost 判斷
        xgb_pred = self.xgb_model.predict(meta_X)  # 0 或 1
        xgb_proba = self.xgb_model.predict_proba(meta_X)[:, 1]
        
        # 綜合決策
        final_confidence = lstm_confidence * xgb_proba
        should_trade = final_confidence > confidence_threshold
        
        return {
            'lstm_pred_class': lstm_pred_class,
            'lstm_confidence': lstm_confidence,
            'lstm_probs': lstm_probs,
            'xgb_approval': xgb_pred,
            'xgb_confidence': xgb_proba,
            'final_confidence': final_confidence,
            'should_trade': should_trade,
            'class_names': ['DOWN', 'FLAT', 'UP']
        }
    
    def save_models(self, model_dir: str, version: str = '1.2.0'):
        """
        保存模型
        """
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        # LSTM 模型
        lstm_path = os.path.join(model_dir, f"lstm_model_v{version}.pt")
        torch.save(self.lstm_model.state_dict(), lstm_path)
        
        # XGBoost 模型
        xgb_path = os.path.join(model_dir, f"xgb_model_v{version}.json")
        self.xgb_model.model.save_model(xgb_path)
        
        # 元数據
        metadata = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'lstm_architecture': {
                'n_features': self.lstm_model.n_features,
                'lstm_hidden_1': self.lstm_model.lstm_hidden_1,
                'lstm_hidden_2': self.lstm_model.lstm_hidden_2,
            },
            'xgb_feature_names': self.xgb_model.feature_names,
            'history': self.history
        }
        
        metadata_path = os.path.join(model_dir, f"metadata_v{version}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"[DualLayerMetaLabelingSystem] 模型已保存到 {model_dir}")
    
    def load_models(self, model_dir: str, version: str = '1.2.0'):
        """
        事先載入模型
        """
        import os
        
        lstm_path = os.path.join(model_dir, f"lstm_model_v{version}.pt")
        xgb_path = os.path.join(model_dir, f"xgb_model_v{version}.json")
        
        if self.lstm_model is not None:
            self.lstm_model.load_state_dict(torch.load(lstm_path, map_location=self.device))
        
        if self.xgb_model is not None:
            self.xgb_model.model = xgb.XGBClassifier()
            self.xgb_model.model.load_model(xgb_path)
        
        print(f"[DualLayerMetaLabelingSystem] 模型已從 {model_dir} 載入")


if __name__ == "__main__":
    print("ZigZag 量化系統 v1.2.0 - 模型架構模塊\n")
    print("結構已準備，待數據加載並進行訓練...")
