import os
import glob
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

class EnhancedFeatureTransformer:
    def __init__(self, window_sizes=(5, 10, 20, 30)):
        self.window_sizes = window_sizes
        self.earth_radius = 6371000

    def transform(self, df):
        df = df.copy()
        
        if {'timestamp', 'latitude', 'longitude'}.issubset(df.columns):
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            df['time_diff'] = df['timestamp'].diff().fillna(1).clip(lower=0.1)
            
            df['lat_rad'] = np.radians(df['latitude'])
            df['lon_rad'] = np.radians(df['longitude'])
            df['dlat'] = df['lat_rad'].diff().fillna(0)
            df['dlon'] = df['lon_rad'].diff().fillna(0)
            
            a = (np.sin(df['dlat']/2)**2 + 
                 np.cos(df['lat_rad'])*np.cos(df['lat_rad'].shift(1).fillna(df['lat_rad']))*
                 np.sin(df['dlon']/2)**2)
            c = 2*np.arctan2(np.sqrt(a.clip(0,1)), np.sqrt((1-a).clip(0,1)))
            df['distance'] = self.earth_radius * c
        
        if 'distance' in df.columns:
            df['velocity'] = df['distance'] / df['time_diff']
            df['accel'] = df['velocity'].diff().fillna(0) / df['time_diff']
            df['jerk'] = df['accel'].diff().fillna(0) / df['time_diff']
            df['abs_accel'] = df['accel'].abs()

        if 'throttle' in df.columns:
            df['throttle_squared'] = df['throttle'] ** 2
            df['throttle_cubed'] = df['throttle'] ** 3
            df['throttle_change'] = df['throttle'].diff().fillna(0)
            df['throttle_change_rate'] = df['throttle_change'] / df['time_diff']
            
            if 'velocity' in df.columns:
                df['throttle_velocity'] = df['throttle'] * df['velocity']
                df['throttle_accel'] = df['throttle'] * df['accel']

        if {'velocity', 'throttle'}.issubset(df.columns):
            df['kinetic_energy'] = df['velocity'] ** 2
            df['power_estimate'] = df['throttle'] * df['velocity']
            df['energy_change'] = df['kinetic_energy'].diff().fillna(0)
            df['power_change'] = df['power_estimate'].diff().fillna(0)
            
            df['power_to_accel_ratio'] = np.where(df['accel'].abs() > 1e-6, 
                                                 df['power_estimate'] / df['accel'].abs(), 
                                                 0)
            
            df['efficiency'] = np.where(df['throttle'] > 1e-6,
                                       df['velocity'] / df['throttle'],
                                       0)

        for col in df.columns:
            if df[col].dtype.kind in 'if':
                for window in self.window_sizes:
                    if col in ['velocity', 'accel', 'throttle', 'power_estimate']:
                        df[f'{col}_ma_{window}'] = df[col].rolling(window, min_periods=1, center=False).mean()
                        df[f'{col}_std_{window}'] = df[col].rolling(window, min_periods=1, center=False).std().fillna(0)
                        df[f'{col}_min_{window}'] = df[col].rolling(window, min_periods=1, center=False).min()
                        df[f'{col}_max_{window}'] = df[col].rolling(window, min_periods=1, center=False).max()
                    elif col not in ['timestamp', 'latitude', 'longitude', 'flight_id']:
                        df[f'{col}_ma_{window}'] = df[col].rolling(window, min_periods=1, center=False).mean()

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)
        
        drop_cols = ['latitude', 'longitude', 'timestamp', 'flight_id'] 
        return df[[c for c in df.columns if c not in drop_cols]].select_dtypes(include=[np.number])

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y=None, seq_len=32):
        self.X = X.astype(np.float32)
        self.seq_len = seq_len
        self.y = y.astype(np.float32) if y is not None else None
        
    def __len__(self):
        return len(self.X) - self.seq_len + 1
    
    def __getitem__(self, idx):
        X_seq = self.X[idx:idx+self.seq_len]
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        
        if self.y is None:
            return X_tensor
            
        y_value = self.y[idx + self.seq_len - 1]
        y_tensor = torch.tensor(y_value, dtype=torch.float32)
        
        return X_tensor, y_tensor

class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size=192, num_layers=2, dropout=0.3, bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_out_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_out_size,
            num_heads=8,
            dropout=dropout/2,
            batch_first=True
        )
        
        self.layer_norm1 = nn.LayerNorm(lstm_out_size)
        self.layer_norm2 = nn.LayerNorm(lstm_out_size)
        
        self.fc1 = nn.Linear(lstm_out_size, lstm_out_size)
        self.fc2 = nn.Linear(lstm_out_size, lstm_out_size//2)
        self.fc3 = nn.Linear(lstm_out_size//2, lstm_out_size//4)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout/2)
        self.dropout3 = nn.Dropout(dropout/4)
        
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        
        self.out = nn.Linear(lstm_out_size//4, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        lstm_out = self.layer_norm1(lstm_out + attn_out)
        
        mean_pool = torch.mean(lstm_out, dim=1)
        max_pool, _ = torch.max(lstm_out, dim=1)
        seq_rep = (mean_pool + max_pool) / 2
        
        normalized = self.layer_norm2(seq_rep)
        
        fc1_out = self.fc1(normalized)
        fc1_out = self.dropout1(self.gelu(fc1_out))
        
        fc2_out = self.fc2(fc1_out)
        fc2_out = self.dropout2(self.gelu(fc2_out))
        
        fc3_out = self.fc3(fc2_out)
        fc3_out = self.dropout3(self.gelu(fc3_out))
        
        return self.out(fc3_out).squeeze(-1)

def load_data_splits(base_dir='BED', transformer=None, val_size=0.2, test_size=0.1, random_state=42):
    if transformer is None:
        transformer = EnhancedFeatureTransformer()
    
    all_dfs = []
    all_ys = []
    
    train_files = sorted(glob.glob(os.path.join(base_dir, 'train', '*.csv')))
    for f in train_files:
        df = pd.read_csv(f)
        if 'battery_current' in df.columns:
            all_ys.append(df['battery_current'].values)
            df = df.drop(columns=['battery_current'])
        all_dfs.append(df)
    
    val_files = sorted(glob.glob(os.path.join(base_dir, 'val', '*.csv')))
    for f in val_files:
        df = pd.read_csv(f)
        if 'battery_current' in df.columns:
            all_ys.append(df['battery_current'].values)
            df = df.drop(columns=['battery_current'])
        all_dfs.append(df)
    
    all_data = pd.concat(all_dfs, ignore_index=True)
    all_targets = np.concatenate(all_ys) if all_ys else None
    
    features = transformer.transform(all_data)
    
    n_samples = len(features)
    indices = np.arange(n_samples)
    np.random.seed(random_state)
    np.random.shuffle(indices)
    
    test_idx = int(n_samples * (1 - test_size))
    val_idx = int(test_idx * (1 - val_size))
    
    train_indices = indices[:val_idx]
    val_indices = indices[val_idx:test_idx]
    test_indices = indices[test_idx:]
    
    X_train = features.iloc[train_indices].values
    y_train = all_targets[train_indices] if all_targets is not None else None
    
    X_val = features.iloc[val_indices].values
    y_val = all_targets[val_indices] if all_targets is not None else None
    
    test_files = sorted(glob.glob(os.path.join(base_dir, 'pred', '*.csv')))
    test_dfs = []
    for f in test_files:
        df = pd.read_csv(f)
        test_dfs.append(df)
    
    test_data = pd.concat(test_dfs, ignore_index=True)
    X_test = transformer.transform(test_data).values
    
    test_gt_files = sorted(glob.glob(os.path.join(base_dir, 'test', '*.csv')))
    if test_gt_files:
        y_test = np.concatenate([
            pd.read_csv(f)['battery_current'].values
            for f in test_gt_files
        ])
    else:
        y_test = None
    
    return X_train, y_train, X_val, y_val, X_test, y_test

class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, decay_factor=0.95, min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.decay_factor = decay_factor
        self.min_lr = min_lr
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        else:
            decay = max(self.decay_factor ** (self.last_epoch - self.warmup_steps), self.min_lr / self.base_lrs[0])
            return [max(base_lr * decay, self.min_lr) for base_lr in self.base_lrs]

def train_model(model, train_dl, val_dl, device, epochs, patience, lr, weight_decay, seq_len, clip_value=0.5, warmup_steps=3):
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = WarmupScheduler(optimizer, warmup_steps=warmup_steps)
    
    # Training loop
    best_val_loss = float('inf')
    no_improvement = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rmse': [],
        'val_r2': [],
        'lr': []
    }
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_losses = []
        train_pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{epochs} [Train]")
        
        for Xb, yb in train_pbar:
            Xb, yb = Xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            outputs = model(Xb)
            loss = nn.MSELoss()(outputs, yb)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            
            optimizer.step()
            scheduler.step()
            
            train_losses.append(loss.item())
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_lr()[0]:.7f}'})
        
        # Validation phase
        model.eval()
        val_losses = []
        val_preds = []
        val_trues = []
        
        with torch.no_grad():
            for Xb, yb in val_dl:
                Xb, yb = Xb.to(device), yb.to(device)
                outputs = model(Xb)
                loss = nn.MSELoss()(outputs, yb)
                val_losses.append(loss.item())
                val_preds.append(outputs.cpu().numpy())
                val_trues.append(yb.cpu().numpy())
        
        # Calculate metrics
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        val_preds = np.concatenate(val_preds)
        val_trues = np.concatenate(val_trues)
        val_rmse = math.sqrt(mean_squared_error(val_trues, val_preds))
        val_r2 = r2_score(val_trues, val_preds)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_rmse'].append(val_rmse)
        history['val_r2'].append(val_r2)
        history['lr'].append(scheduler.get_lr()[0])
        
        # Log metrics
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.6f}")
        print(f"Val Loss: {avg_val_loss:.6f}")
        print(f"Val RMSE: {val_rmse:.6f}")
        print(f"Val R²: {val_r2:.4f}")
        print(f"LR: {scheduler.get_lr()[0]:.7f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement = 0
            torch.save(model.state_dict(), 'best.pth')
            print(f"New best model saved! (Val Loss: {best_val_loss:.6f})")
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
    
    return model, history

def predict_and_evaluate(model, test_dl, device, y_test=None, seq_len=32):
    model.eval()
    test_preds = []
    
    test_pbar = tqdm(test_dl, desc="Predicting")
    
    with torch.no_grad():
        for X_batch in test_pbar:
            if isinstance(X_batch, tuple):
                X_batch = X_batch[0]
                
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            test_preds.extend(outputs.cpu().numpy())
    
    y_pred = np.array(test_preds)
    
    if y_test is not None:
        y_test_aligned = y_test[seq_len-1:seq_len-1+len(y_pred)]
        
        rmse = math.sqrt(mean_squared_error(y_test_aligned, y_pred))
        r2 = r2_score(y_test_aligned, y_pred)
        
        print(f"Test Results → RMSE: {rmse:.6f} | R²: {r2:.4f}")
        return y_pred, rmse, r2
    
    return y_pred

def visualize_results(history, y_test=None, y_pred=None):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(history['val_r2'], label='Validation R²', marker='o')
    plt.title('Validation R² Score')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(history['val_rmse'], label='Validation RMSE', marker='o')
    plt.title('Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(history['lr'], label='Learning Rate', marker='.')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    if y_test is not None and y_pred is not None:
        plt.figure(figsize=(15, 6))
        plt.plot(y_test[:len(y_pred)], label='Actual', alpha=0.7)
        plt.plot(y_pred, label='Predicted', alpha=0.7)
        plt.title('Battery Current: Actual vs Predicted')
        plt.xlabel('Time Step')
        plt.ylabel('Battery Current')
        plt.legend()
        plt.grid(True)
        plt.savefig('predictions.png')
    
    plt.close('all')

def main():
    base_dir = 'BED'
    seq_len = 64
    batch_size = 128
    hidden_size = 192
    num_layers = 3
    dropout = 0.4
    epochs = 5
    patience = 5
    learning_rate = 1e-4
    weight_decay = 5e-5
    bidirectional = True
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading and transforming data...")
    transformer = EnhancedFeatureTransformer()
    X_train, y_train, X_val, y_val, X_test, y_test = load_data_splits(
        base_dir=base_dir, 
        transformer=transformer,
        val_size=0.2,
        test_size=0.1,
        random_state=42
    )
    
    print(f"Data shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape if y_test is not None else None}")
    
    print("Standardizing features...")
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("Creating data loaders...")
    train_ds = TimeSeriesDataset(X_train_scaled, y_train, seq_len=seq_len)
    val_ds = TimeSeriesDataset(X_val_scaled, y_val, seq_len=seq_len)
    test_ds = TimeSeriesDataset(X_test_scaled, None, seq_len=seq_len)
    
    train_dl = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True
    )
    
    val_dl = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True
    )
    
    test_dl = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True
    )
    
    model_type = "LSTM"
    input_size = X_train_scaled.shape[1]
    
    print(f"Creating {model_type} model with attention...")
    model = LSTMWithAttention(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    print("Training model...")
    model, history = train_model(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        device=device,
        epochs=epochs,
        patience=patience,
        lr=learning_rate,
        weight_decay=weight_decay,
        seq_len=seq_len,
        clip_value=0.5,
        warmup_steps=3
    )
    
    print("Making predictions...")
    if y_test is not None:
        y_pred, test_rmse, test_r2 = predict_and_evaluate(model, test_dl, device, y_test, seq_len)
    else:
        y_pred = predict_and_evaluate(model, test_dl, device, None, seq_len)
    
    np.save('predictions.npy', y_pred)
    
    print("Visualizing results...")
    visualize_results(history, y_test, y_pred)
    
    print("Done!")

if __name__ == "__main__":
    main() 
