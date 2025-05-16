import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import time
import os

# تنظیمات اولیه
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M1
SEQ_LEN = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# کلاس مدل (همانند train.py)
class DirectionPredictor(nn.Module):
    def __init__(self, input_size):
        super(DirectionPredictor, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(128, 1)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.fc(context)

# بارگذاری مدل
model = DirectionPredictor(input_size=10).to(DEVICE)
model.load_state_dict(torch.load("model.pt", map_location=DEVICE))
model.eval()

# شروع متاتریدر
if not mt5.initialize():
    raise RuntimeError("MT5 initialization failed")

# بررسی باز بودن معامله
def trade_is_open():
    positions = mt5.positions_get(symbol=SYMBOL)
    return len(positions) > 0

# دریافت داده‌های زنده
def get_live_data(n=SEQ_LEN):
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, n)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# پیش‌پردازش داده‌ها
def preprocess(df, scaler=None):
    df['returns'] = df['close'].pct_change().fillna(0)
    df['hl'] = df['high'] - df['low']
    df['oc'] = df['close'] - df['open']
    features = df[['open', 'high', 'low', 'close', 'tick_volume', 'volume', 'spread', 'returns', 'hl', 'oc']]
    if scaler is None:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    else:
        features = scaler.transform(features)
    return features, scaler

# اجرای پیش‌بینی
def predict_direction():
    df = get_live_data(SEQ_LEN + 1)
    if df.shape[0] < SEQ_LEN + 1:
        return None
    X, _ = preprocess(df)
    x_seq = torch.tensor(X[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x_seq)
        pred = torch.argmax(logits, dim=1).item()
    return pred  # 0: Sell, 1: Buy

# ارسال سفارش
def place_order(direction):
    price = mt5.symbol_info_tick(SYMBOL).ask if direction == 1 else mt5.symbol_info_tick(SYMBOL).bid
    volume = mt5.account_info().balance * 0.001 / price  # 0.1% سرمایه
    order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": round(volume, 2),
        "type": order_type,
        "price": price,
        "deviation": 10,
        "magic": 123456,
        "comment": "AutoTrade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }
    result = mt5.order_send(request)
    return result

# حلقه اصلی
print("Started auto trader...")
scaler = None
while True:
    try:
        if not trade_is_open():
            df = get_live_data(SEQ_LEN + 1)
            if df.shape[0] < SEQ_LEN + 1:
                time.sleep(10)
                continue
            X, scaler = preprocess(df, scaler)
            direction = predict_direction()
            if direction is not None:
                result = place_order(direction)
                print(f"{datetime.now()} - Signal: {'Buy' if direction==1 else 'Sell'} - Order: {result.retcode}")
        time.sleep(60)  # هر دقیقه یکبار بررسی
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(60)
