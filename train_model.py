import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import talib
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Technical Indicators ---
def add_features(df):
    df['EMA_10'] = talib.EMA(df['close'], timeperiod=10)
    df['RSI_14'] = talib.RSI(df['close'], timeperiod=14)
    macd, signal, _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD_hist'] = macd - signal
    upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
    df['BB_width'] = upper - lower
    df['VWAP'] = (df['close'] * df['tick_volume']).cumsum() / df['tick_volume'].replace(0, np.nan).cumsum()
    return df.dropna()

# --- Multi-timeframe merge ---
def merge_timeframes(df1m):
    df1m['datetime'] = pd.to_datetime(df1m['date'] + ' ' + df1m['time'])
    df1m.set_index('datetime', inplace=True)
    df5m = df1m.resample('5T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'tick_volume': 'sum',
        'volume': 'sum',
        'spread': 'mean'
    }).dropna().rename(columns={c: f'{c}_5m' for c in df1m.columns})
    df = df1m.join(df5m, how='inner')
    df.reset_index(drop=False, inplace=True)
    return df

# --- Labeling ---
def generate_labels(df, future_steps=5, threshold=0.0003):
    future_returns = df['close'].shift(-future_steps) - df['close']
    df['direction'] = (future_returns > threshold).astype(int)
    return df.dropna()

# --- Load Data ---
def load_data(file_path, seq_len=60):
    df = pd.read_csv(file_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = merge_timeframes(df)
    df = add_features(df)
    df = generate_labels(df)
    
    features = df.drop(columns=['date', 'time', 'datetime', 'direction'])
    labels = df['direction'].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(features)

    X_seq = []
    y_seq = []
    for i in range(seq_len, len(X_scaled)):
        X_seq.append(X_scaled[i-seq_len:i])
        y_seq.append(labels[i])
    
    X_seq = torch.tensor(np.array(X_seq), dtype=torch.float32)
    y_seq = torch.tensor(np.array(y_seq), dtype=torch.long)
    return X_seq, y_seq, scaler

# --- Attention ---
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x).squeeze(-1), dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)

# --- Model ---
class CNNBiLSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.cnn = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bilstm = nn.LSTM(64, hidden_size, batch_first=True, bidirectional=True)
        self.attn = Attention(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)              # (B, F, T)
        x = torch.relu(self.cnn(x))         # (B, C, T)
        x = x.permute(0, 2, 1)              # (B, T, C)
        x, _ = self.bilstm(x)               # (B, T, H*2)
        x = self.attn(x)                    # (B, H*2)
        return self.fc(x)

# --- Training ---
def train(file='m1.csv', seq_len=60, batch_size=128, epochs=30, val_ratio=0.1):
    X, y, _ = load_data(file, seq_len)
    dataset = TensorDataset(X, y)
    val_len = int(len(dataset) * val_ratio)
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_len, val_len])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    model = CNNBiLSTMAttention(X.shape[2]).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).to(device))  # Weight "buy" class more
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    best_f1 = 0
    patience = 5
    counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device)
                pred = model(xb)
                all_preds.extend(pred.argmax(1).cpu().numpy())
                all_labels.extend(yb.numpy())

        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        avg_loss = total_loss / len(train_dl.dataset)

        print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_loss:.4f} | Acc: {acc:.4f} - Prec: {prec:.4f} - Rec: {rec:.4f} - F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping.")
                break

if __name__ == "__main__":
    train()
