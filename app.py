#!/usr/bin/env python3
"""
ZigZag 實時預測 API 伺服端
Flask 應用，提供實時市場數據和模型預測
支援光端選擇幣種和時間框架
支援每秒自動更新，使用前一根完成的K棒進行預測

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

# 信號定義 - 明確標示每個代碼的含義
SIGNAL_DEFINITIONS = {
    0: "HH (Higher High) - 上升趨勢延續，新高",
    1: "HL (High Low) - 上升轉下降，頭部反轉信號",
    2: "LH (Low High) - 下降轉上升，底部反轉信號",
    3: "LL (Lower Low) - 下降趨勢延續，新低"
}

# 22 個支持的幣種
AVAILABLE_PAIRS = [
    'AAVEUSDT', 'ADAUSDT', 'ALGOUSDT', 'ARBUSDT', 'ATOMUSDT',
    'AVAXUSDT', 'BCHUSDT', 'BNBUSDT', 'BTCUSDT', 'DOGEUSDT',
    'DOTUSDT', 'ETCUSDT', 'ETHUSDT', 'FILUSDT', 'LINKUSDT',
    'LTCUSDT', 'MATICUSDT', 'NEARUSDT', 'OPUSDT', 'SOLUSDT',
    'UNIUSDT', 'XRPUSDT'
]

AVAILABLE_INTERVALS = ['15m', '1h']

def convert_to_python_types(obj):
    """
    遞歸會診 NumPy 類形離义轉換為 Python 粗來類形
    目程是 JSON 序列化
    """
    if isinstance(obj, dict):
        return {convert_to_python_types(k): convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


class RealTimePredictorService:
    """
    實時預測服務
    負責加載模型、獲取實時數據、進行預測
    每秒自動更新，使用前一根完成的K棒
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.label_encoder = None
        self.params = None
        self.scaler = StandardScaler()
        self.latest_prediction = None
        self.previous_ohlcv = None  # 前一根K棒
        self.current_ohlcv = None   # 當前K棒
        self.history = []
        self.update_thread = None
        self.is_running = False
        self.binance_client = None
        self.current_pair = 'BTCUSDT'
        self.current_interval = '15m'
        self.last_candle_time = None  # 追蹤上一根K棒的時間
        
        # 初始化 Binance 客戶端
        if BINANCE_AVAILABLE:
            try:
                self.binance_client = Client()
                print("Binance 客戶端已初始化")
            except:
                print("Binance 客戶端初始化失敗")
        
        self.load_model()
    
    def get_available_models(self):
        """
        獲取所有可用模型
        返回格式: {pair: {interval: model_path}}
        """
        models = {}
        
        # 掃描 models 目錄
        pair_dirs = glob.glob('models/*')
        for pair_dir in pair_dirs:
            pair_name = Path(pair_dir).name
            interval_dirs = glob.glob(f'{pair_dir}/*')
            
            models[pair_name] = {}
            for interval_dir in interval_dirs:
                interval_name = Path(interval_dir).name
                model_dirs = glob.glob(f'{interval_dir}/model_*')
                
                if model_dirs:
                    # 取最新的模型
                    latest_model = max(model_dirs, key=lambda x: Path(x).stat().st_mtime)
                    models[pair_name][interval_name] = latest_model
        
        return models
    
    def load_model(self, pair='BTCUSDT', interval='15m'):
        """
        加載指定幣種和時間框架的模型
        """
        try:
            # 這一實数据幫下模型目錄
            model_path = f'models/{pair}/{interval}'
            
            if not Path(model_path).exists():
                print(f"模型目錄不存在: {model_path}")
                return False
            
            # 取最新的模型
            model_dirs = glob.glob(f'{model_path}/model_*')
            if not model_dirs:
                print(f"何找到模型檔案: {model_path}")
                return False
            
            model_dir = max(model_dirs, key=lambda x: Path(x).stat().st_mtime)
            print(f"加載模型: {model_dir}")
            
            # 儯先嘗試 joblib 格式
            if Path(f'{model_dir}/xgboost_model.joblib').exists():
                print("  使用 joblib 格式...")
                self.model = joblib.load(f'{model_dir}/xgboost_model.joblib')
            # 其次嘗試 JSON 格式
            elif Path(f'{model_dir}/xgboost_model.json').exists():
                print("  使用 JSON 格式...")
                self.model = xgb.XGBClassifier()
                self.model.load_model(f'{model_dir}/xgboost_model.json')
            else:
                print("何找到模型檔案")
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
            
            self.current_pair = pair
            self.current_interval = interval
            
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
                print(f"Binance 未獲取數據: {pair} {interval}")
                return None
            
            # 轉換為 DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # 粗略篇正式轉換
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # 僅保留需要的欄
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            return df
        
        except Exception as e:
            print(f"獲取 Binance 數據失敖: {str(e)}")
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
                '4h': '1h',
                '1d': '1d'
            }
            yf_interval = interval_map.get(interval, '15m')
            
            # 獲取數據
            df = yf.download(symbol, period='60d', interval=yf_interval, progress=False)
            
            if df.empty:
                print(f"未能從 yfinance 獲取 {pair} 的數據")
                return None
            
            # 重命名欄
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df.index.name = 'timestamp'
            df = df.reset_index()
            
            return df
        
        except Exception as e:
            print(f"獲取 yfinance 數據失敖: {str(e)}")
            return None
    
    def get_realtime_data(self, pair='BTCUSDT', interval='15m'):
        """
        獲取實時數據
        選項: Binance (API) -> yfinance (備用)
        """
        # 儯先使用 Binance API
        if BINANCE_AVAILABLE:
            df = self.get_realtime_data_binance(pair, interval)
            if df is not None:
                return df
        
        # 撤伊 yfinance
        if YFINANCE_AVAILABLE:
            return self.get_realtime_data_yfinance(pair, interval)
        
        return None
    
    def _zigzag_points(self, prices, depth=12, deviation=0.8):
        """
        简單 ZigZag 轉折點計算
        返回轉折點位置的布林陳列
        """
        if len(prices) < depth * 2:
            return np.zeros(len(prices))
        
        # 計算 ZigZag
        zz = np.zeros(len(prices))
        trend = 0  # 1: 上下, -1: 下下
        pivot_idx = 0
        pivot_price = prices.iloc[0]
        
        for i in range(depth, len(prices) - depth):
            current = prices.iloc[i]
            high_window = prices.iloc[i:i+depth].max()
            low_window = prices.iloc[i:i+depth].min()
            
            # 計算偽差百分比
            if pivot_price > 0:
                deviation_pct = abs(current - pivot_price) / pivot_price * 100
            else:
                deviation_pct = 0
            
            # 判斷趨勢變化
            if trend == 0:  # 初始化
                if current > pivot_price:
                    trend = 1
                else:
                    trend = -1
            elif trend == 1 and current < pivot_price * (1 - deviation / 100):
                # 上下轉下下
                trend = -1
                zz[pivot_idx] = 1  # 標記高點
                pivot_idx = i
                pivot_price = current
            elif trend == -1 and current > pivot_price * (1 + deviation / 100):
                # 下下轉上下
                trend = 1
                zz[pivot_idx] = -1  # 標記低點
                pivot_idx = i
                pivot_price = current
        
        return zz
    
    def extract_zigzag_features(self, df):
        """
        提取 ZigZag 特徵
        基於最新的 K 棊數據
        
        為保證輸出特徵數量為 11：
        - open
        - high
        - low
        - close
        - volume
        - zigzag (轉折點指標)
        - returns
        - volatility
        - momentum
        - rsi
        - macd
        """
        try:
            df = df.copy()
            
            # 提取 ZigZag 轉折點特徵
            # 使用訓練時的參數
            depth = self.params.get('depth', 12)
            deviation = self.params.get('deviation', 0.8)
            df['zigzag'] = self._zigzag_points(df['close'], depth=depth, deviation=deviation)
            
            # 基礎技術指標
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(20).std()
            df['momentum'] = df['close'] - df['close'].shift(10)
            df['rsi'] = self._calculate_rsi(df['close'], 14)
            df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
            
            # 填充缺失值
            df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
            
            return df
        
        except Exception as e:
            print(f"特徵提取失敖: {str(e)}")
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
    
    def _get_signal_code_and_description(self, prediction_value):
        """
        將預測值映射到信號代碼和描述
        0: HH, 1: HL, 2: LH, 3: LL
        """
        try:
            signal_code = int(prediction_value)
            if signal_code not in SIGNAL_DEFINITIONS:
                signal_code = 0
        except:
            signal_code = 0
        
        return signal_code, SIGNAL_DEFINITIONS[signal_code]
    
    def predict(self, pair='BTCUSDT', interval='15m', use_previous_candle=True):
        """
        進行預測
        
        參數:
            pair: 交易對
            interval: 時間框架
            use_previous_candle: 是否使用前一根完成的K棒進行預測
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
                    'message': '無法獲取市場數據 - 請檢查網路連接'
                }
            
            # 提取特徵
            df_features = self.extract_zigzag_features(df)
            if df_features is None:
                return {
                    'status': 'error',
                    'message': '特徵提取失敖'
                }
            
            # 決定使用哪一根K棒
            if use_previous_candle and len(df_features) >= 2:
                # 使用前一根完成的K棒
                analysis_row = df_features.iloc[-2]  # 倒數第二個
                candle_type = "前一根"
            else:
                # 使用當前K棒
                analysis_row = df_features.iloc[-1]  # 最後一個
                candle_type = "當前"
            
            # 保存OHLCV數據
            ohlcv_data = {
                'timestamp': str(analysis_row['timestamp']),
                'open': float(analysis_row['open']),
                'high': float(analysis_row['high']),
                'low': float(analysis_row['low']),
                'close': float(analysis_row['close']),
                'volume': float(analysis_row['volume']) if pd.notna(analysis_row['volume']) else 0
            }
            
            if use_previous_candle:
                self.previous_ohlcv = ohlcv_data
            else:
                self.current_ohlcv = ohlcv_data
            
            # 準備特徵橢量 - 按照模型訓練時的順序
            feature_cols = [col for col in self.feature_names if col in df_features.columns]
            
            # 剪槍記風黱: 有五個特徵沒有對應的欄
            missing_features = [f for f in self.feature_names if f not in df_features.columns]
            if missing_features:
                print(f"警告: 特徵 {missing_features} 不存在整個數據褒藤中")
            
            # 按照預測標整順溏提取
            X_raw = df_features.iloc[-1:][feature_cols].values
            
            # 如果特徵數量不灆，沒有適當的特徵就填 0
            if X_raw.shape[1] != len(self.feature_names):
                print(f"特徵數量不匹配: {X_raw.shape[1]} vs {len(self.feature_names)}")
                print(f"  存在的特徵: {feature_cols}")
                print(f"  標浪之特徵: {self.feature_names}")
                
                # 建策整個整物特徵橢量，沒有的就填 0
                X_padded = np.zeros((1, len(self.feature_names)))
                for i, feat in enumerate(self.feature_names):
                    if feat in feature_cols:
                        idx = feature_cols.index(feat)
                        X_padded[0, i] = X_raw[0, idx]
                X = X_padded
            else:
                X = X_raw
            
            # 預測
            y_pred = self.model.predict(X)[0]
            y_pred_proba = self.model.predict_proba(X)[0]
            
            # 將預測轉換為標籤
            pred_label = self.label_encoder.inverse_transform([y_pred])[0]
            confidence = float(np.max(y_pred_proba))
            
            # 獲取信號代碼和描述
            signal_code, signal_description = self._get_signal_code_and_description(pred_label)
            
            # 判斷是否應該 HOLD
            if confidence < 0.6:
                signal = 'HOLD'
                signal_code = -1
            else:
                signal = f"{signal_code}_{['HH', 'HL', 'LH', 'LL'][signal_code]}"
            
            # 將 label_encoder.classes_ 轉換為 Python 類形
            class_labels = [str(label) for label in self.label_encoder.classes_]
            
            self.latest_prediction = {
                'timestamp': datetime.now().isoformat(),
                'pair': pair,
                'interval': interval,
                'signal_code': signal_code,
                'signal_name': ['HH', 'HL', 'LH', 'LL'][signal_code] if signal_code >= 0 else 'HOLD',
                'signal_description': signal_description if signal_code >= 0 else 'Hold Position',
                'predicted_type': str(pred_label),
                'confidence': confidence,
                'ohlcv': ohlcv_data,
                'candle_type': candle_type,
                'based_on': 'previous_candle' if use_previous_candle else 'current_candle',
                'all_probabilities': {
                    class_labels[idx]: float(prob) 
                    for idx, prob in enumerate(y_pred_proba)
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
            print(f"預測失敖: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def start_auto_update(self, pair='BTCUSDT', interval='15m', update_interval=1):
        """
        開始自動更新線程
        默認每1秒更新一次
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
        自動更新趨環 - 每秒執行一次，使用前一根K棒
        """
        while self.is_running:
            try:
                self.predict(pair, interval, use_previous_candle=True)
                
                # 通過 WebSocket 推送最新信號到所有連接的客戶端
                if self.latest_prediction:
                    try:
                        socketio.emit('prediction_update', {
                            'status': 'success',
                            'data': convert_to_python_types(self.latest_prediction)
                        }, to=None)
                    except Exception as emit_err:
                        print(f"WebSocket 推送出錯: {str(emit_err)}")
                
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


@app.route('/api/pairs', methods=['GET'])
def get_pairs():
    """獲取全部可用幣種"""
    return jsonify({
        'status': 'success',
        'pairs': AVAILABLE_PAIRS,
        'intervals': AVAILABLE_INTERVALS
    })


@app.route('/api/available-models', methods=['GET'])
def get_available_models():
    """獲取可用模型清單"""
    available_models = predictor_service.get_available_models()
    return jsonify({
        'status': 'success',
        'models': available_models
    })


@app.route('/api/load-model', methods=['POST'])
def load_model():
    """加載指定模型"""
    pair = request.json.get('pair', 'BTCUSDT')
    interval = request.json.get('interval', '15m')
    
    success = predictor_service.load_model(pair, interval)
    
    if success:
        # 停止之前的自動更新
        predictor_service.stop_auto_update()
        # 開始新的自動更新 - 每1秒更新一次
        predictor_service.start_auto_update(pair, interval, update_interval=1)
        
        return jsonify({
            'status': 'success',
            'message': f'模型已加載: {pair} {interval}',
            'config': {
                'pair': pair,
                'interval': interval,
                'depth': predictor_service.params.get('depth'),
                'deviation': predictor_service.params.get('deviation'),
                'update_interval': 1
            }
        })
    else:
        return jsonify({
            'status': 'error',
            'message': f'模型加載失敗: {pair} {interval}'
        })


@app.route('/api/latest', methods=['GET'])
def get_latest():
    """獲取最新預測"""
    if predictor_service.latest_prediction is None:
        return jsonify({
            'status': 'error',
            'message': '暫無預測數據，請稍後'
        })
    
    # 轉換為 Python 類形
    response_data = convert_to_python_types(predictor_service.latest_prediction)
    return jsonify({
        'status': 'success',
        'data': response_data
    })


@app.route('/api/history', methods=['GET'])
def get_history():
    """獲取歷史預測"""
    limit = int(request.args.get('limit', 100))
    # 轉換為 Python 類形
    history_data = convert_to_python_types(predictor_service.history[-limit:])
    return jsonify({
        'status': 'success',
        'data': history_data
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """觸發預測"""
    pair = request.json.get('pair', 'BTCUSDT')
    interval = request.json.get('interval', '15m')
    use_previous = request.json.get('use_previous_candle', True)
    
    # 檢查模型是否匹配
    if predictor_service.current_pair != pair or predictor_service.current_interval != interval:
        # 需要加載新模型
        predictor_service.load_model(pair, interval)
    
    result = predictor_service.predict(pair, interval, use_previous_candle=use_previous)
    # 轉換為 Python 類形
    result = convert_to_python_types(result)
    return jsonify(result)


@app.route('/api/config', methods=['GET'])
def get_config():
    """獲取配置信息"""
    if predictor_service.params:
        return jsonify({
            'status': 'success',
            'data': {
                'pair': predictor_service.current_pair,
                'interval': predictor_service.current_interval,
                'depth': predictor_service.params.get('depth', 12),
                'deviation': predictor_service.params.get('deviation', 0.8),
                'num_features': len(predictor_service.feature_names) if predictor_service.feature_names else 0,
                'auto_update_enabled': predictor_service.is_running,
                'update_interval': 1
            }
        })
    else:
        return jsonify({
            'status': 'error',
            'message': '模型未加載'
        })


@app.route('/api/signal-definitions', methods=['GET'])
def get_signal_definitions():
    """獲取信號定義"""
    return jsonify({
        'status': 'success',
        'definitions': SIGNAL_DEFINITIONS
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'model_loaded': predictor_service.model is not None,
        'latest_prediction': predictor_service.latest_prediction is not None,
        'binance_available': BINANCE_AVAILABLE and predictor_service.binance_client is not None,
        'yfinance_available': YFINANCE_AVAILABLE,
        'current_pair': predictor_service.current_pair,
        'current_interval': predictor_service.current_interval,
        'auto_update_enabled': predictor_service.is_running
    })


# WebSocket 事件
@socketio.on('connect')
def handle_connect():
    print('客戶端已連接')
    emit('connection_response', {'data': 'Connected to prediction server'})


@socketio.on('disconnect')
def handle_disconnect():
    print('客戶端已斷開')


@socketio.on('subscribe')
def handle_subscribe(data):
    pair = data.get('pair', 'BTCUSDT')
    interval = data.get('interval', '15m')
    join_room(f'{pair}_{interval}')
    emit('subscribed', {'pair': pair, 'interval': interval})


if __name__ == '__main__':
    # 開始自動更新 - 每1秒更新一次
    predictor_service.start_auto_update(
        pair='BTCUSDT',
        interval='15m',
        update_interval=1
    )
    
    # 啟動 Flask 應用
    print("\n" + "="*60)
    print("ZigZag 實時預測服務已啟動")
    print("="*60)
    print("訪問: http://localhost:5000")
    print("API: http://localhost:5000/api/latest")
    print("WebSocket: ws://localhost:5000/socket.io")
    print("特徵: 每秒自動更新，使用前一根K棒進行預測")
    print("信號: 0=HH, 1=HL, 2=LH, 3=LL")
    print("模式: Binance API 優先, yfinance 備用")
    print("="*60 + "\n")
    
    try:
        socketio.run(app, debug=False, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n關閉應用...")
        predictor_service.stop_auto_update()
