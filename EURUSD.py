# اسکالپ کامل با CNN + BiLSTM + Attention + ترکیب تایم‌فریم‌ها و ویژگی‌ها + فعال‌سازی GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # استفاده از GPU اول اگر موجود باشد

import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import ta
import time
import pytz
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv1D, Bidirectional, LSTM, Dense, Attention, Dropout, Concatenate, LayerNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import joblib

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU فعال است")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ هیچ GPUیی شناسایی نشد، استفاده از CPU")

### 1. بارگذاری و ساخت ویژگی‌ها از CSV

def load_and_create_features(file_1m, file_5m):
    df_1m = pd.read_csv(file_1m)
    df_5m = pd.read_csv(file_5m)

    for df in [df_1m, df_5m]:
        df['EMA'] = ta.trend.ema_indicator(df['close'], window=14).fillna(0)
        df['RSI'] = ta.momentum.rsi(df['close'], window=14).fillna(0)
        df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14).fillna(0)
        macd = ta.trend.macd_diff(df['close'])
        df['MACD_hist'] = macd.fillna(0)
        df['BB_width'] = (ta.volatility.BollingerBands(df['close']).bollinger_hband() - ta.volatility.BollingerBands(df['close']).bollinger_lband()).fillna(0)
        df['VWAP'] = ((df['high'] + df['low'] + df['close']) / 3).fillna(0)

        df['candle_body'] = abs(df['close'] - df['open'])
        df['candle_range'] = df['high'] - df['low']
        df['marubozu'] = ((df['high'] == df['close']) & (df['low'] == df['open'])).astype(int)
        df['doji'] = (df['candle_body'] / df['candle_range'] < 0.1).astype(int)

    df = df_1m.copy()
    df5_aligned = df_5m.reindex(df.index).fillna(method='ffill')

    combined = pd.concat([df, df5_aligned.add_suffix('_5m')], axis=1).dropna()
    return combined

### 2. آماده‌سازی داده‌ها برای آموزش

def prepare_data(df, lookback=30):
    features = df.drop(columns=['time', 'symbol', 'direction', 'TP', 'SL'], errors='ignore').values
    labels = df[['direction', 'TP', 'SL']].values

    X, y = [], []
    for i in range(lookback, len(df)):
        X.append(features[i - lookback:i])
        y.append(labels[i])
    return np.array(X), np.array(y)

### 3. ساخت مدل CNN → BiLSTM → Attention

def build_model(input_shape):
    inp = Input(shape=input_shape)
    x = Conv1D(64, kernel_size=3, activation='relu')(inp)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Attention()([x, x])
    x = LayerNormalization()(x)
    x = tf.keras.layers.Flatten()(x)

    direction_output = Dense(1, activation='sigmoid', name='direction')(x)
    tp_output = Dense(1, activation='linear', name='tp')(x)
    sl_output = Dense(1, activation='linear', name='sl')(x)

    model = Model(inputs=inp, outputs=[direction_output, tp_output, sl_output])
    model.compile(optimizer='adam',
                  loss={'direction': 'binary_crossentropy', 'tp': 'mse', 'sl': 'mse'},
                  loss_weights={'direction': 1.0, 'tp': 0.7, 'sl': 0.3},
                  metrics={'direction': 'accuracy'})
    return model

### 4. اتصال به متاتریدر۵ برای داده زنده و ارسال سفارش

def connect_mt5():
    if not mt5.initialize():
        raise Exception("MT5 init failed")

def get_live_data(symbol, timeframe, n=50):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    return pd.DataFrame(rates)

def place_order(symbol, direction, tp, sl, volume):
    price = mt5.symbol_info_tick(symbol).ask if direction == 1 else mt5.symbol_info_tick(symbol).bid
    order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
    sl_price = price - sl if direction == 1 else price + sl
    tp_price = price + tp if direction == 1 else price - tp

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": 10,
        "magic": 1001,
        "comment": "scalp_ai",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    return mt5.order_send(request)

def get_lot_size():
    account_info = mt5.account_info()
    balance = account_info.balance
    lot = balance * 0.002  # یک‌هزارم
    return round(lot, 2)
### 5. اجرای زنده سیستم

def run_live_trading(model, symbol):
    connect_mt5()
    lot = get_lot_size()
    while True:
        df = get_live_data(symbol, mt5.TIMEFRAME_M1, n=100)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df['symbol'] = symbol

        # افزودن ویژگی‌ها دقیقاً مشابه مرحله آموزش
        df_feats = load_and_create_features(df, df)
        X_live, _ = prepare_data(df_feats)
        X_live = X_live[-1:]

        direction, tp, sl = model.predict(X_live)
        direction = int(direction[0][0] > 0.5)
        tp_val = float(tp[0][0])
        sl_val = float(sl[0][0])

        if not any([p.ticket for p in mt5.positions_get(symbol=symbol)]):
            res = place_order(symbol, direction, tp_val, sl_val, lot)
            print("ORDER SENT", res)
        time.sleep(60)

### 6. اجرای آموزش

def train_and_run():
    df = load_and_create_features("m1.csv", "m5.csv")
    X, y = prepare_data(df)

    model = build_model(X.shape[1:])
    early_stop = EarlyStopping(patience=5, monitor='val_direction_accuracy', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(patience=3, factor=0.5)

    model.fit(X, {"direction": y[:, 0], "tp": y[:, 1], "sl": y[:, 2]},
              validation_split=0.1,
              epochs=30,
              batch_size=64,
              callbacks=[early_stop, reduce_lr])

    model.save("scalp_model.h5")
    run_live_trading(model, "EURUSD")

if __name__ == '__main__':
    train_and_run()
