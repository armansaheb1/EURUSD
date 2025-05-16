import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import time
import datetime

# پارامترها
SYMBOL = "EURUSD"
TIMEFRAME_1M = mt5.TIMEFRAME_M1
TIMEFRAME_5M = mt5.TIMEFRAME_M5
WINDOW_SIZE = 30
ATR_PERIOD = 14
ATR_MULTIPLIER = 1.5
MODEL_PATH = "scalp_model.h5"
LOG_FILE = "trade_log.csv"

# اتصال به متاتریدر
if not mt5.initialize():
    raise RuntimeError("MT5 initialization failed")

# بارگذاری مدل
model = load_model(MODEL_PATH)

# محاسبه ATR
def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

# استخراج ویژگی‌ها
def extract_features(df):
    df['ema_10'] = df['close'].ewm(span=10).mean()
    df['rsi_14'] = compute_rsi(df['close'], 14)
    df['atr'] = calculate_atr(df, ATR_PERIOD)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    return df

def compute_rsi(series, period):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain = pd.Series(gain).rolling(window=period).mean()
    loss = pd.Series(loss).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

# دریافت داده از MT5
def get_price_data(symbol, timeframe, n):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# بررسی پوزیشن باز
def has_open_position(symbol):
    positions = mt5.positions_get(symbol=symbol)
    return positions is not None and len(positions) > 0

# انجام معامله
def open_trade(symbol, direction, sl_points, tp_points, lot):
    price = mt5.symbol_info_tick(symbol).ask if direction == "buy" else mt5.symbol_info_tick(symbol).bid
    deviation = 10

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": price - sl_points if direction == "buy" else price + sl_points,
        "tp": price + tp_points if direction == "buy" else price - tp_points,
        "deviation": deviation,
        "magic": 123456,
        "comment": "LSTM scalp trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    print(f"[{datetime.datetime.now()}] Trade result:", result)
    return result

# ثبت در لاگ
def log_trade(signal, sl, tp, result):
    log_data = {
        "time": [datetime.datetime.now()],
        "signal": [signal],
        "sl": [sl],
        "tp": [tp],
        "result": [result.retcode if result else None]
    }
    df = pd.DataFrame(log_data)
    df.to_csv(LOG_FILE, mode='a', header=not pd.io.common.file_exists(LOG_FILE), index=False)

# حلقه اصلی اجرا
def run_live():
    account_info = mt5.account_info()
    balance = account_info.balance
    lot = round(balance * 0.001, 2)

    df_1m = get_price_data(SYMBOL, TIMEFRAME_1M, WINDOW_SIZE + ATR_PERIOD)
    df_5m = get_price_data(SYMBOL, TIMEFRAME_5M, (WINDOW_SIZE + ATR_PERIOD) // 5 + 1)

    df_1m = extract_features(df_1m)
    df_5m = extract_features(df_5m)
    df_5m_aligned = df_5m.reindex(df_1m.index).fillna(method='ffill')

    combined = pd.concat([df_1m, df_5m_aligned.add_suffix("_5m")], axis=1)
    features = combined[-WINDOW_SIZE:].drop(columns=["time", "time_5m"], errors="ignore")

    # نرمال‌سازی
    scaler = MinMaxScaler()
    input_scaled = scaler.fit_transform(features)
    X_input = np.expand_dims(input_scaled, axis=0)

    # پیش‌بینی
    pred_direction, pred_sl_norm, pred_tp_norm = model.predict(X_input)[0]
    predicted_class = np.argmax(pred_direction)
    direction = "buy" if predicted_class == 2 else "sell" if predicted_class == 0 else "hold"

    # مقیاس‌دهی TP و SL
    last_atr = df_1m['atr'].iloc[-1]
    sl_pips = max(5, pred_sl_norm * ATR_MULTIPLIER * last_atr)
    tp_pips = max(5, pred_tp_norm * ATR_MULTIPLIER * last_atr)

    # بررسی و ارسال سفارش
    if direction != "hold" and not has_open_position(SYMBOL):
        result = open_trade(SYMBOL, direction, sl_pips, tp_pips, lot)
        log_trade(direction, sl_pips, tp_pips, result)
    else:
        print(f"[{datetime.datetime.now()}] No trade executed. Signal: {direction}")

# اجرای متناوب
if __name__ == "__main__":
    while True:
        try:
            run_live()
            time.sleep(60)  # اجرا در هر 1 دقیقه
        except Exception as e:
            print("Error:", e)
            time.sleep(60)
