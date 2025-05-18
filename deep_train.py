import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset, DataLoader, random_split
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange
from sklearn.metrics import classification_report
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau

warnings.filterwarnings('ignore')

# 1. کلاس Dataset
class ForexDataset(Dataset):
    def __init__(self, features, targets, seq_length=60):
        self.features = features
        self.targets = targets
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        x = self.features[idx:idx+self.seq_length]
        y = self.targets[idx+self.seq_length-1]
        return torch.FloatTensor(x), torch.FloatTensor([y])

# 2. مدل پیشرفته
class ForexModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 128, batch_first=True, bidirectional=True, dropout=0.2)
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        attn_weights = self.attention(x)
        x = torch.sum(x * attn_weights, dim=1)
        return self.classifier(x)

# 3. پیش‌پردازش داده‌ها
def preprocess_data(file_path):
    # بارگیری داده‌ها
    df = pd.read_csv(file_path)
    
    # تحلیل اولیه
    print("\nتحلیل داده‌های اولیه:")
    print(f"تعداد نمونه‌ها: {len(df)}")
    print("مقدارهای گمشده:")
    print(df.isnull().sum())
    
    # تبدیل تاریخ و زمان
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%m/%d/%Y %I:%M:%S %p')
    df = df.dropna(subset=['datetime'])
    df = df.sort_values('datetime')
    
    # محاسبه ویژگی‌های اصلی
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    df['momentum'] = df['close'] - df['close'].shift(5)
    
    # اندیکاتورهای تکنیکال
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    df['macd'] = MACD(df['close'], window_slow=26, window_fast=12).macd()
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    
    # ویژگی‌های ترکیبی
    df['rsi_macd'] = df['rsi'] * df['macd']
    df['vol_return'] = df['volatility'] * df['returns'].abs()
    
    # هدف‌گذاری (پیش‌بینی 5 کندل آینده)
    df['target'] = (df['close'].shift(-5) > df['close']).astype(int)
    
    # حذف مقادیر نامعتبر
    df = df.dropna()
    
    # انتخاب ویژگی‌های نهایی
    features = df[['returns', 'rsi', 'macd', 'volatility', 'momentum', 'atr', 'rsi_macd', 'vol_return']]
    targets = df['target'].values
    
    # نرمال‌سازی
    scaler = RobustScaler()
    features = scaler.fit_transform(features)
    
    print(f"\nتعداد نمونه‌های نهایی: {len(features)}")
    print(f"تعداد ویژگی‌ها: {features.shape[1]}")
    return features, targets, scaler

# 4. آموزش و ارزیابی
def train_and_evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nاستفاده از دستگاه: {device}")
    
    # پیش‌پردازش داده‌ها
    features, targets, scaler = preprocess_data('m1.csv')
    
    # ایجاد Dataset
    dataset = ForexDataset(features, targets)
    
    # تقسیم داده
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # DataLoaderها
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # ایجاد مدل و انتقال به دستگاه مناسب
    model = ForexModel(input_size=features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5], device=device))  # انتقال وزن به دستگاه
    
    # آموزش
    best_acc = 0
    for epoch in range(100):
        model.train()
        train_loss, correct = 0, 0
        
        for x, y in train_loader:
            # انتقال داده به دستگاه مناسب
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == y).sum().item()
        
        # ارزیابی
        model.eval()
        test_correct = 0
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for x, y in test_loader:
                # انتقال داده به دستگاه مناسب
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                test_correct += ((torch.sigmoid(outputs) > 0.5).float().eq(y).sum().item())
                y_true.extend(y.cpu().numpy())  # انتقال به CPU برای گزارش
                y_pred.extend((torch.sigmoid(outputs) > 0.5).float().cpu().numpy())
        
        train_acc = correct / len(train_dataset)
        test_acc = test_correct / len(test_dataset)
        scheduler.step(test_acc)
        
        if epoch % 1 == 0:
            print(f"\nEpoch {epoch}:")
            print(f"Train Loss: {train_loss/len(train_loader):.4f}")
            print(f"Train Accuracy: {train_acc:.2%}")
            print(f"Test Accuracy: {test_acc:.2%}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_forex_model.pth')
            print(f"\nمدل بهبود یافت! دقت تست جدید: {best_acc:.2%}")
            
            # گزارش طبقه‌بندی
            print("\nگزارش طبقه‌بندی:")
            print(classification_report(y_true, y_pred, target_names=['Down', 'Up']))
        
        # Early Stopping
        if epoch > 20 and test_acc < 0.52:
            print("\nتوقف زودهنگام به دلیل عملکرد ضعیف")
            break
    
    print(f"\nبهترین دقت تست: {best_acc:.2%}")

if __name__ == "__main__":
    train_and_evaluate()