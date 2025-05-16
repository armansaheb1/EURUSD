import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import tensorflow as tf
from keras.models import load_model
import ta
from datetime import datetime
import time


LOOKBACK = 30
SYMBOL = "EURUSD-VIP"
TIMEFRAME = mt5.TIMEFRAME_M1
VOLUME = 0.01
MODEL_PATH = "scalp_model.h5"
def get_lot_size():
    account_info = mt5.account_info()
    balance = account_info.balance
    lot = balance * 0.002  # یک‌هزارم
    return round(lot, 2)

def connect_mt5():
    if not mt5.initialize():
        raise Exception("❌ اتصال به متاتریدر5 برقرار نشد")
    
    print("✅ اتصال به متاتریدر5 برقرار شد")





def load_and_create_features(df_1m, df_5m):
    def add_indicators(df):
        df['EMA']       = ta.trend.ema_indicator(df['close'], window=14).fillna(0)
        df['RSI']       = ta.momentum.rsi(df['close'], window=14).fillna(0)
        df['ATR']       = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14).fillna(0)
        df['MACD_hist'] = ta.trend.macd_diff(df['close']).fillna(0)
        bb = ta.volatility.BollingerBands(df['close'])
        df['BB_width']  = (bb.bollinger_hband() - bb.bollinger_lband()).fillna(0)
        df['VWAP']      = ((df['high'] + df['low'] + df['close']) / 3).fillna(0)
        df['candle_body']  = abs(df['close'] - df['open'])
        df['candle_range'] = df['high'] - df['low']
        df['marubozu']     = ((df['high'] == df['close']) & (df['low'] == df['open'])).astype(int)
        df['doji']         = (df['candle_body'] / df['candle_range'] < 0.1).astype(int)
        return df

    df1 = add_indicators(df_1m.copy())
    df5 = add_indicators(df_5m.copy())

    df5_aligned = df5.reindex(df1.index).ffill().add_suffix("_5m")
    combined = pd.concat([df1, df5_aligned], axis=1).fillna(0)

    # اضافه کردن 3 ستون dummy که مدل آموزش دیده انتظارشان را دارد
    combined['direction'] = 0
    combined['TP']        = 0
    combined['SL']        = 0

    input_columns = [
        # 1m
        'open','high','low','close','tick_volume','volume','spread',
        'EMA','RSI','ATR','MACD_hist','BB_width','VWAP',
        'candle_body','candle_range','marubozu','doji',
        # 5m
        'open_5m','high_5m','low_5m','close_5m','tick_volume_5m','volume_5m','spread_5m',
        'EMA_5m','RSI_5m','ATR_5m','MACD_hist_5m','BB_width_5m','VWAP_5m',
        'candle_body_5m','candle_range_5m','marubozu_5m','doji_5m',
        # dummy
        'direction','TP','SL'
    ]
    # حتماً همه ستون‌ها
    for col in input_columns:
        if col not in combined.columns:
            combined[col] = 0
    combined = combined[input_columns]
    assert combined.shape[1] == 37, f"❌ تعداد ستون {combined.shape[1]} است، باید 37 باشد"
    return combined

def get_last_data():
    # دریافت نرخ‌ها
    r1 = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, LOOKBACK + 100)
    r5 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, LOOKBACK + 50)
    df1 = pd.DataFrame(r1)[['time','open','high','low','close','tick_volume']].copy()
    df5 = pd.DataFrame(r5)[['time','open','high','low','close','tick_volume']].copy()

    df1['time'] = pd.to_datetime(df1['time'], unit='s')
    df5['time'] = pd.to_datetime(df5['time'], unit='s')
    df1.set_index('time', inplace=True)
    df5.set_index('time', inplace=True)

    df = load_and_create_features(df1, df5)
    df = df[-LOOKBACK:]
    if len(df) < LOOKBACK:
        raise Exception("❌ داده کافی برای پیش‌بینی وجود ندارد")

    X = df.select_dtypes(include=[np.number]).values.reshape(1, LOOKBACK, -1)
    atr = df['ATR'].iloc[-1]
    return X, atr

def has_open_position():
    pos = mt5.positions_get(symbol=SYMBOL)
    return pos is not None and len(pos) > 0

def place_order(direction, tp_value, sl_value, max_retries=3):
    VOLUME = get_lot_size()
    tick = mt5.symbol_info_tick(SYMBOL)
    info = mt5.symbol_info(SYMBOL)
    price = tick.ask if direction == 1 else tick.bid

    point = info.point                 # مثلاً 0.00001
    min_dist = info.trade_stops_level * point  # معمولاً 0

    # حداقل فاصله (مثلاً 1 پیپ = 10 * point)
    pip_dist = 10 * point

    # raw distances (از مدل/ATR)
    raw_sl = sl_value
    raw_tp = tp_value

    # فاصله‌ی نهایی = raw + حداقل از سرور و اجباری 1 پیپ
    dist_sl = max(raw_sl, 0.00030) 
    dist_tp = max(raw_tp, 0.00030) 

    # محاسبه قیمت SL/TP
    if direction == 1:  # BUY
        sl_price = price - dist_sl
        tp_price = price + dist_tp
    else:              # SELL
        sl_price = price + dist_sl
        tp_price = price - dist_tp

    print(f"🔍 Using dist_sl={dist_sl:.6f}, dist_tp={dist_tp:.6f} (min 1 pip = {pip_dist:.6f})")

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": VOLUME,
        "type": mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": 10,
        "magic": 1001,
        "comment": "scalp_ai",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    for attempt in range(1, max_retries + 1):
        res = mt5.order_send(request)
        code = res.retcode
        print(f"[{datetime.now()}] Attempt {attempt}: retcode={code}")
        if code in (mt5.TRADE_RETCODE_DONE, 10009):
            print("✅ سفارش موفق!")
            return res
        time.sleep(0.5)

    print(f"❌ بعد از {max_retries} تلاش، سفارش شکست خورد (retcode={code})")
    return res

def run_trader():
    connect_mt5()
    model = load_model(MODEL_PATH, compile=False)

    if has_open_position():
        print("ℹ️ پوزیشن باز موجود است، هیچ کاری انجام نمی‌شود.")
        return

    X, atr = get_last_data()
    # ==== اینجا مهم است: =====
    direction_pred, tp_pred, sl_pred = model.predict(X)
    direction = int(direction_pred[0,0] > 0.5)
    tp = float(tp_pred[0,0] * atr)
    sl = float(sl_pred[0,0] * atr)
    print(f"📈 سیگنال: {'Buy' if direction else 'Sell'} | TP: {tp:.5f} | SL: {sl:.5f}")

    place_order(direction, tp, sl)

if __name__ == "__main__":
    run_trader()
