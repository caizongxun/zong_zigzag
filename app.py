#!/usr/bin/env python3
"""
ZigZag 實時預測 API 伺服端
Flask 應用，提供實時市場數據和模型預測
支援全部 38 個幣種
使用前一根完成的 K 棒進行預測
顯示無信號、HH、HL、LL、LH 的概率分佈

啟動方法：
  python app.py

API 端點：
  GET /api/pairs - 獲取全部可用幣種
  GET /api/available-models - 獲取可用模型
  POST /api/load-model - 加載模型
  GET /api/latest - 獲取最新預測
  GET /api/history - 獲取歷史數據
  POST /api/predict - 觸發預測
  GET /api/config - 獲取配置信息
  GET /api/signal-definitions - 獲取信號定義
  GET /api/all-coins-status - 獲取所有幣種最新狀態
  POST /api/predict-all-coins - 針對所有幣種觸發預測
  WebSocket /ws - 實時推送信號
"""

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import pandas as pd
import numpy as np
import pickle
import json as json_lib
import glob
from datetime import datetime, timedelta
from pathlib import Path
import threading
import time
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("警告: yfinance 未安裝")

try:
    from binance.client import Client
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    print("警告: python-binance 未安裝")

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# 信號定義
SIGNAL_DEFINITIONS = {
    'HOLD': "無信號 - 不符合任何趨勢特徵",
    'HH': "Higher High - 上升趨勢延續，新高",
    'HL': "High Low - 上升轉下降，頭部反轉",
    'LH': "Low High - 下降轉上升，底部反轉",
    'LL': "Lower Low - 下降趨勢延續，新低"
}

# 38 個支持的幣種
AVAILABLE_PAIRS = [
    'AAVEUSDT', 'ADAUSDT', 'ALGOUSDT', 'ARBUSDT', 'ATOMUSDT',
    'AVAXUSDT', 'BALUSDT', 'BATUSDT', 'BCHUSDT', 'BNBUSDT',
    'BTCUSDT', 'COMPUSDT', 'CRVUSDT', 'DOGEUSDT', 'DOTUSDT',
    'ENJUSDT', 'ENSUSDT', 'ETCUSDT', 'ETHUSDT', 'FILUSDT',
    'GALAUSDT', 'GRTUSDT', 'IMXUSDT', 'KAVAUSDT', 'LINKUSDT',
    'LTCUSDT', 'MANAUSDT', 'MATICUSDT', 'MKRUSDT', 'NEARUSDT',
    'OPUSDT', 'SANDUSDT', 'SNXUSDT', 'SOLUSDT', 'SPELLUSDT',
    'UNIUSDT', 'XRPUSDT', 'ZRXUSDT'
]

AVAILABLE_INTERVALS = ['15m', '1h']

# 其餘原有程式碼保持不變 ...
# 為節省篇幅，此處省略未修改部分

# 在原有 API 之後新增以下兩個端點：

@app.route('/api/all-coins-status', methods=['GET'])
def get_all_coins_status():
    """獲取所有幣種的最新狀態
    僅使用目前已存在的 latest_prediction 作為示例，如果需要真正針對每個幣種預測，
    建議搭配 /api/predict-all-coins 使用。
    """
    interval = request.args.get('interval', '15m')
    result = {}

    for pair in AVAILABLE_PAIRS:
        # 目前為了避免大量即時請求 Binance，先返回占位資料
        result[pair] = {
            'pair': pair,
            'main_signal': 'HOLD',
            'confidence': 0.0,
            'ohlcv': None,
            'timestamp': datetime.now().isoformat()
        }

    # 如果已有最新單幣種預測，將其覆蓋對應幣種
    if predictor_service.latest_prediction is not None:
        lp = predictor_service.latest_prediction
        pair = lp.get('pair')
        if pair in result:
            result[pair] = lp

    return jsonify({
        'status': 'success',
        'data': result
    })


@app.route('/api/predict-all-coins', methods=['POST'])
def predict_all_coins():
    """針對所有幣種觸發一次預測
    注意: 這個操作可能較慢，視模型載入與資料來源而定。
    """
    interval = request.json.get('interval', '15m')
    results = {}

    for pair in AVAILABLE_PAIRS:
        try:
            # 每個幣種先嘗試載入模型
            predictor_service.load_model(pair, interval)
            result = predictor_service.predict(pair, interval)
            if result.get('status') == 'success':
                results[pair] = result['data']
            else:
                results[pair] = {
                    'pair': pair,
                    'main_signal': 'HOLD',
                    'confidence': 0.0,
                    'ohlcv': None,
                    'timestamp': datetime.now().isoformat(),
                    'error': result.get('message', '未知錯誤')
                }
        except Exception as e:
            results[pair] = {
                'pair': pair,
                'main_signal': 'HOLD',
                'confidence': 0.0,
                'ohlcv': None,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

    return jsonify({
        'status': 'success',
        'data': results
    })
