import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import MetaTrader5 as mt5
import talib
import time

# ===========================
# Section 1: Model Definition
# ===========================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x).squeeze(-1), dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)

class CNNBiLSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.cnn = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bilstm = nn.LSTM(64, hidden_size, batch_first=True, bidirectional=True)
        self.attn = Attention(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.cnn(x))
        x = x.permute(0, 2, 1)
        x, _ = self.bilstm(x)
        x = self.attn(x)
        return self.fc(x)

# ===========================
# Section 2: Training Setup
# ===========================
SEQ_LEN = 60
device = torch.device("cpu")

def add_features(df):
    df['return'] = df['close'].pct_change()
    df['future_return'] = df['close'].shift(-5) / df['close'] - 1
    df['target'] = np.where(df['future_return'] > 0, 1, 0)
    df['EMA_10'] = talib.EMA(df['close'], timeperiod=10)
    df['RSI_14'] = talib.RSI(df['close'], timeperiod=14)
    macd, signal, _ = talib.MACD(df['close'])
    df['MACD_hist'] = macd - signal
    upper, middle, lower = talib.BBANDS(df['close'])
    df['BB_width'] = upper - lower
    df['VWAP'] = (df['close'] * df['tick_volume']).cumsum() / df['tick_volume'].replace(0, np.nan).cumsum()
    df.dropna(inplace=True)
    return df

def prepare_sequences(df, feature_cols):
    scaler = StandardScaler()
    features = scaler.fit_transform(df[feature_cols])
    X, y = [], []
    for i in range(SEQ_LEN, len(df)):
        X.append(features[i-SEQ_LEN:i])
        y.append(df['target'].iloc[i])
    return np.array(X), np.array(y), scaler

def train_model(csv_file, model_path="best_model.pth", scaler_path="scaler.pkl"):
    df = pd.read_csv(csv_file)
    df = add_features(df)
    feature_cols = ['open','high','low','close','spread','tick_volume',
                    'EMA_10','RSI_14','MACD_hist','BB_width','VWAP']
    
    X, y, scaler = prepare_sequences(df, feature_cols)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = CNNBiLSTMAttention(input_size=X.shape[2]).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0], device=device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)

    best_val_acc = 0
    for epoch in range(15):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = torch.argmax(model(X_val), dim=1)
            val_acc = (val_pred == y_val).float().mean().item()
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Val Acc: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), model_path)
                joblib.dump(scaler, scaler_path)
                np.save("features.npy", np.array(feature_cols))

    return model, scaler, feature_cols

# ===========================
# Section 3: Trader
# ===========================
SYMBOL = "EURUSD-VIP"
VOLUME = 0.01

def get_data(symbol, n=500, tf=mt5.TIMEFRAME_M1):
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, n)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={'time':'datetime'}, inplace=True)
    return df

def make_trade(model, scaler, feature_cols):
    df = get_data(SYMBOL, n=500)
    df = add_features(df)
    df = df[-(SEQ_LEN+1):]
    if len(df) < SEQ_LEN+1:
        return
    features = scaler.transform(df[feature_cols])
    X_live = torch.tensor(features[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred = torch.argmax(model(X_live), dim=1).item()

    tick = mt5.symbol_info_tick(SYMBOL)
    price = tick.ask if pred == 1 else tick.bid
    atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
    atr = max(atr, 0.0003)

    if pred == 1:
        sl, tp = price - atr * 1.5, price + atr * 3
        order_type = mt5.ORDER_TYPE_BUY
    else:
        sl, tp = price + atr * 1.5, price - atr * 3
        order_type = mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": VOLUME,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": 123456,
        "comment": "auto-trade",
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"✅ Trade sent: {'BUY' if pred==1 else 'SELL'} at {price:.5f}")
    else:
        print("❌ Order failed:", result.comment)

def run_trader_loop(model, scaler, feature_cols):
    if not mt5.initialize():
        print("MT5 init failed")
        return
    while True:
        pos = mt5.positions_get(symbol=SYMBOL)
        if pos and len(pos) > 0:
            print("⏳ Position already open. Waiting...")
        else:
            make_trade(model, scaler, feature_cols)
        time.sleep(60)

# ===========================
# Main Logic
# ===========================
if __name__ == "__main__":
    model_path = "best_model.pth"
    scaler_path = "scaler.pkl"
    features_path = "features.npy"

    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path):
        print("📦 Loading model & scaler...")
        dummy_input = torch.zeros((1, SEQ_LEN, len(np.load(features_path))))
        model = CNNBiLSTMAttention(input_size=dummy_input.shape[2]).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        scaler = joblib.load(scaler_path)
        feature_cols = np.load(features_path).tolist()
    else:
        print("🔧 No model found. Training...")
        model, scaler, feature_cols = train_model("m1.csv", model_path, scaler_path)

    run_trader_loop(model, scaler, feature_cols)
