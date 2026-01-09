#!/usr/bin/env python3
"""
ZigZag 實時預測 API 後端
Flask 應用，提供實時市場數據和模型預測

啟動方法：
  python app.py

API 端點：
  GET /api/latest - 獲取最新預測
  GET /api/history - 獲取歷史數據
  GET /api/config - 獲取配置信息
"""

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
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


class RealTimePredictorService:
    """
    實時預測服務
    負責加載模型、獲取實時數據、進行預測
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.label_encoder = None
        self.params = None
        self.scaler = StandardScaler()
        self.latest_prediction = None
        self.latest_ohlcv = None
        self.history = []
        self.update_thread = None
        self.is_running = False
        self.binance_client = None
        
        # 初始化 Binance 客戶端
        if BINANCE_AVAILABLE:
            try:
                self.binance_client = Client()
                print("✓ Binance 客戶端已初始化")
            except:
                print("⚠ Binance 客戶端初始化失敖")
        
        self.load_model()
    
    def load_model(self):
        """
        加載最新的模型
        """
        try:
            model_dirs = glob.glob('models/*')
            if not model_dirs:
                print("未找到模型目錄")
                return False
            
            model_dir = max(model_dirs, key=lambda x: Path(x).stat().st_mtime)
            print(f"加載模型: {model_dir}")
            
            # 优先尝试 joblib 格式
            if Path(f'{model_dir}/xgboost_model.joblib').exists():
                print("  使用 joblib 格式...")
                self.model = joblib.load(f'{model_dir}/xgboost_model.joblib')
            # 此二尝试 JSON 格式
            elif Path(f'{model_dir}/xgboost_model.json').exists():
                print("  使用 JSON 格式...")
                self.model = xgb.XGBClassifier()
                self.model.load_model(f'{model_dir}/xgboost_model.json')
            else:
                print("未找到模型檔案")
                return False
            
            # 加載標籤編碼器
            with open(f'{model_dir}/label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            # 加載特徵名稱
            with open(f'{model_dir}/feature_names.json', 'r') as f:
                self.feature_names = json_lib.load(f)
            
            # 加載參數
            with open(f'{model_dir}/params.json', 'r') as f:
                self.params = json_lib.load(f)
            
            print(f"模型加載成功")
            print(f"  交易對: {self.params.get('pair', 'N/A')}")
            print(f"  時間框架: {self.params.get('interval', 'N/A')}")
            print(f"  特徵數: {len(self.feature_names)}")
            
            return True
        except Exception as e:
            print(f"模型加載失敖: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_realtime_data_binance(self, pair='BTCUSDT', interval='15m'):
        """
        從 Binance 獲取實時數據 (API 官方)
        """
        try:
            if not self.binance_client:
                return None
            
            # 映射時間框架
            interval_map = {
                '15m': Client.KLINE_INTERVAL_15MINUTE,
                '1h': Client.KLINE_INTERVAL_1HOUR,
                '4h': Client.KLINE_INTERVAL_4HOUR,
                '1d': Client.KLINE_INTERVAL_1DAY
            }
            
            binance_interval = interval_map.get(interval, Client.KLINE_INTERVAL_15MINUTE)
            
            # 獲取最近 200 根 K 線 (最大上限)
            klines = self.binance_client.get_klines(
                symbol=pair,
                interval=binance_interval,
                limit=200
            )
            
            if not klines:
                print(f"煙 Binance 未獲取資料: {pair} {interval}")
                return None
            
            # 轉換為 DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # 粗低篇正式轉換
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # 仅保留需要的欄
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            return df
        
        except Exception as e:
            print(f"獲取 Binance 數據失敕: {str(e)}")
            return None
    
    def get_realtime_data_yfinance(self, pair='BTCUSDT', interval='15m'):
        """
        從 yfinance 獲取實時數據 (備用方案)
        """
        try:
            # 符號轉換
            if pair == 'BTCUSDT':
                symbol = 'BTC-USD'
            else:
                symbol = pair.replace('USDT', '-USD')
            
            # 映射時間框架
            interval_map = {
                '15m': '15m',
                '1h': '1h',
                '4h': '1h',  # yfinance 最小粒度
                '1d': '1d'
            }
            yf_interval = interval_map.get(interval, '15m')
            
            # 獲取数據
            df = yf.download(symbol, period='60d', interval=yf_interval, progress=False)
            
            if df.empty:
                print(f"未能从yfinance獲取 {pair} 的數據")
                return None
            
            # 重命名列
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df.index.name = 'timestamp'
            df = df.reset_index()
            
            return df
        
        except Exception as e:
            print(f"獲取 yfinance 數據失敕: {str(e)}")
            return None
    
    def get_realtime_data(self, pair='BTCUSDT', interval='15m'):
        """
        獲取實時數據
        選項: Binance (API) → yfinance (備用)
        """
        # 優先使用 Binance API
        if BINANCE_AVAILABLE:
            df = self.get_realtime_data_binance(pair, interval)
            if df is not None:
                return df
        
        # 詰依 yfinance
        if YFINANCE_AVAILABLE:
            return self.get_realtime_data_yfinance(pair, interval)
        
        return None
    
    def extract_zigzag_features(self, df):
        """
        提取 ZigZag 特徵
        基於最新的 K 棒數據
        """
        try:
            df = df.copy()
            
            # 基礎技術指標
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(20).std()
            df['momentum'] = df['close'] - df['close'].shift(10)
            df['rsi'] = self._calculate_rsi(df['close'], 14)
            df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
            
            # 添加移動平均
            df['sma_5'] = df['close'].rolling(5).mean()
            df['sma_10'] = df['close'].rolling(10).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            
            # 添加高低點距離
            df['high_low_range'] = df['high'] - df['low']
            df['close_to_high'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10)
            df['close_to_low'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
            
            # 填充缺失值
            df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
            
            return df
        
        except Exception as e:
            print(f"特徵提取失敕: {str(e)}")
            return None
    
    def _calculate_rsi(self, prices, period=14):
        """
        計算 RSI
        """
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 100.0 - 100.0 / (1.0 + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.0
            else:
                upval = 0.0
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 0
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)
        
        return rsi
    
    def predict(self, pair='BTCUSDT', interval='15m'):
        """
        進行預測
        """
        try:
            if self.model is None:
                return {
                    'status': 'error',
                    'message': '模型未加載'
                }
            
            # 獲取數據
            df = self.get_realtime_data(pair, interval)
            if df is None:
                return {
                    'status': 'error',
                    'message': '無法獲取市場數據 - 請検查網路連接'
                }
            
            # 提取特徵
            df_features = self.extract_zigzag_features(df)
            if df_features is None:
                return {
                    'status': 'error',
                    'message': '特徵提取失敕'
                }
            
            # 獲取最新的 K 棒
            latest_row = df_features.iloc[-1]
            self.latest_ohlcv = {
                'timestamp': str(latest_row['timestamp']),
                'open': float(latest_row['open']),
                'high': float(latest_row['high']),
                'low': float(latest_row['low']),
                'close': float(latest_row['close']),
                'volume': float(latest_row['volume']) if pd.notna(latest_row['volume']) else 0
            }
            
            # 準備特徵向量
            feature_cols = [col for col in self.feature_names if col in df_features.columns]
            X = df_features.iloc[-1:][feature_cols].values
            
            if X.shape[1] != len(self.feature_names):
                print(f"特徵數量不匹配: {X.shape[1]} vs {len(self.feature_names)}")
                # 填充缺失的特徵
                X_padded = np.zeros((1, len(self.feature_names)))
                for i, feat in enumerate(self.feature_names):
                    if feat in feature_cols:
                        idx = feature_cols.index(feat)
                        X_padded[0, i] = X[0, idx]
                X = X_padded
            
            # 預測
            y_pred = self.model.predict(X)[0]
            y_pred_proba = self.model.predict_proba(X)[0]
            
            # 將預測轉換為標籤
            pred_label = self.label_encoder.inverse_transform([y_pred])[0]
            confidence = float(np.max(y_pred_proba))
            
            # 判斷是否應該 HOLD
            if confidence < 0.6:
                signal = 'HOLD'
            else:
                signal = pred_label
            
            self.latest_prediction = {
                'timestamp': datetime.now().isoformat(),
                'signal': signal,
                'predicted_type': pred_label,
                'confidence': confidence,
                'ohlcv': self.latest_ohlcv,
                'all_probabilities': {
                    label: float(prob) 
                    for label, prob in zip(self.label_encoder.classes_, y_pred_proba)
                }
            }
            
            # 添加到歷史
            self.history.append(self.latest_prediction.copy())
            # 保留最近 1000 個預測
            if len(self.history) > 1000:
                self.history = self.history[-1000:]
            
            return {
                'status': 'success',
                'data': self.latest_prediction
            }
        
        except Exception as e:
            print(f"預測失敕: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def start_auto_update(self, pair='BTCUSDT', interval='15m', update_interval=60):
        """
        開始自動更新線程
        """
        if self.is_running:
            return
        
        self.is_running = True
        self.update_thread = threading.Thread(
            target=self._auto_update_loop,
            args=(pair, interval, update_interval),
            daemon=True
        )
        self.update_thread.start()
        print(f"自動更新已啟動，更新間隔: {update_interval} 秒")
    
    def _auto_update_loop(self, pair, interval, update_interval):
        """
        自動更新循環
        """
        while self.is_running:
            try:
                self.predict(pair, interval)
                time.sleep(update_interval)
            except Exception as e:
                print(f"自動更新出錯: {str(e)}")
                time.sleep(update_interval)
    
    def stop_auto_update(self):
        """
        停止自動更新
        """
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        print("自動更新已停止")


# 初始化服務
predictor_service = RealTimePredictorService()


# API 端點
@app.route('/', methods=['GET'])
def index():
    """主頁面"""
    return render_template('index.html')


@app.route('/api/latest', methods=['GET'])
def get_latest():
    """獲取最新預測"""
    if predictor_service.latest_prediction is None:
        return jsonify({
            'status': 'error',
            'message': '暫無預測數據，請稍候'
        })
    
    return jsonify({
        'status': 'success',
        'data': predictor_service.latest_prediction
    })


@app.route('/api/history', methods=['GET'])
def get_history():
    """獲取歷史預測"""
    limit = int(request.args.get('limit', 100))
    return jsonify({
        'status': 'success',
        'data': predictor_service.history[-limit:]
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """觸發預測"""
    pair = request.json.get('pair', 'BTCUSDT')
    interval = request.json.get('interval', '15m')
    
    result = predictor_service.predict(pair, interval)
    return jsonify(result)


@app.route('/api/config', methods=['GET'])
def get_config():
    """獲取配置信息"""
    if predictor_service.params:
        return jsonify({
            'status': 'success',
            'data': {
                'pair': predictor_service.params.get('pair', 'BTCUSDT'),
                'interval': predictor_service.params.get('interval', '15m'),
                'depth': predictor_service.params.get('depth', 12),
                'deviation': predictor_service.params.get('deviation', 0.8),
                'num_features': len(predictor_service.feature_names) if predictor_service.feature_names else 0
            }
        })
    else:
        return jsonify({
            'status': 'error',
            'message': '模型未加載'
        })


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康檢查"""
    return jsonify({
        'status': 'ok',
        'model_loaded': predictor_service.model is not None,
        'latest_prediction': predictor_service.latest_prediction is not None,
        'binance_available': BINANCE_AVAILABLE and predictor_service.binance_client is not None,
        'yfinance_available': YFINANCE_AVAILABLE
    })


if __name__ == '__main__':
    # 啟動自動更新
    predictor_service.start_auto_update(
        pair='BTCUSDT',
        interval='15m',
        update_interval=60  # 每 60 秒更新一次
    )
    
    # 啟動 Flask 應用
    print("\n" + "="*60)
    print("ZigZag 實時預測服務已啟動")
    print("="*60)
    print("訪問: http://localhost:5000")
    print("API: http://localhost:5000/api/latest")
    print("模式Binance API 優先, yfinance 備用")
    print("="*60 + "\n")
    
    try:
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\n關閉應用...")
        predictor_service.stop_auto_update()
