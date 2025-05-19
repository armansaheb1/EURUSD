import os
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ⚙️ تنظیمات
SEQ_LEN = 30
SYMBOL = "EURUSD-VIP"
VOLUME = 0.01
MAGIC_NUMBER = 123456
MODEL_PATH = "best_model.pth"
SCALER_PATH = "scaler.pkl"
FEATURE_PATH = "feature_order.npy"
CSV_FILE = "m1.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📌 کلاس‌های مدل
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

# 📊 افزودن ویژگی‌ها
def add_features(df):
    import talib
    df['EMA_10'] = talib.EMA(df['close'], timeperiod=10)
    df['RSI_14'] = talib.RSI(df['close'], timeperiod=14)
    macd, signal, _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD_hist'] = macd - signal
    upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
    df['BB_width'] = upper - lower
    df['VWAP'] = (df['close'] * df['tick_volume']).cumsum() / df['tick_volume'].replace(0, np.nan).cumsum()
    df.dropna(inplace=True)
    return df

# ⏱ ترکیب تایم‌فریم‌ها
def merge_timeframes(df1m):
    df1m = df1m.drop(columns=["volume"], errors="ignore")
    df1m = df1m.dropna(subset=["datetime"])
    df1m.set_index("datetime", inplace=True)

    agg = {
        "open": "first", "high": "max", "low": "min",
        "close": "last", "tick_volume": "sum", "spread": "mean"
    }
    df5m = df1m.resample("5min").agg(agg).dropna()
    df5m.columns = [f"{col}_5m" for col in df5m.columns]

    df = df1m.join(df5m, how="inner")
    df.reset_index(inplace=True)
    return df

# 📁 پردازش داده CSV برای آموزش
def load_and_process_csv(csv_file):
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df = merge_timeframes(df)
    df = add_features(df)
    df = df.drop(columns=["date", "time", "volume"], errors="ignore")

    df['return'] = df['close'].pct_change().shift(-1)
    df['target'] = (df['return'] > 0).astype(int)
    df.dropna(inplace=True)
    return df

# 🧪 آماده‌سازی داده دنباله‌ای
def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

# 🏋️ آموزش مدل
def train_model(csv_file, model_path, scaler_path):
    df = load_and_process_csv(csv_file)

    y = df["target"]
    X = df.drop(columns=["target", "return", "datetime"], errors="ignore")
    feature_cols = X.columns
    joblib.dump(feature_cols.tolist(), FEATURE_PATH)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, scaler_path)

    X_seq, y_seq = create_sequences(X_scaled, y.values, SEQ_LEN)
    X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

    model = CNNBiLSTMAttention(X_seq.shape[2]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 11):
        model.train()
        for i in range(0, len(X_train), 64):
            x_batch = torch.tensor(X_train[i:i+64], dtype=torch.float32).to(device)
            y_batch = torch.tensor(y_train[i:i+64], dtype=torch.long).to(device)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

        # اعتبارسنجی
        model.eval()
        val_x = torch.tensor(X_val, dtype=torch.float32).to(device)
        val_y = torch.tensor(y_val, dtype=torch.long).to(device)
        with torch.no_grad():
            val_out = model(val_x)
            acc = (val_out.argmax(dim=1) == val_y).float().mean().item()
        print(f"Epoch {epoch}, Val Acc: {acc:.4f}")

    torch.save(model.state_dict(), model_path)
    return model, scaler, feature_cols

# 🤖 پیش‌بینی روی آخرین داده CSV
def predict_live(model, scaler, feature_cols):
    df = load_and_process_csv(CSV_FILE)
    df = df[feature_cols]
    df_scaled = scaler.transform(df)

    X_live, _ = create_sequences(df_scaled, np.zeros(len(df_scaled)), SEQ_LEN)
    X_live = torch.tensor(X_live[-1:], dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        output = model(X_live)
        pred = torch.argmax(output, dim=1).item()
    return pred

# ▶️ اجرای نهایی
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print("🔧 No model found. Training...")
        model, scaler, feature_cols = train_model(CSV_FILE, MODEL_PATH, SCALER_PATH)
    else:
        print("✅ Model exists. Loading...")
        feature_cols = joblib.load(FEATURE_PATH)
        scaler = joblib.load(SCALER_PATH)
        model = CNNBiLSTMAttention(len(feature_cols)).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # پیش‌بینی آخرین جهت
    pred = predict_live(model, scaler, feature_cols)
    print(f"📈 Market Direction: {'BUY' if pred == 1 else 'SELL'}")
