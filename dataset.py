import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class EnhancedFeatureExtractor:
    def __init__(self, window_sizes=(5, 10, 20)):
        self.window_sizes = window_sizes
        
    def compute_features(self, df):
        df = df.copy()
        
        # Time-based features
        df['time_diff'] = df['timestamp'].diff().fillna(0)
        
        # Motion features
        if {'pitch', 'roll', 'yaw'}.issubset(df.columns):
            df['total_rotation'] = np.sqrt(df['pitch']**2 + df['roll']**2 + df['yaw']**2)
            df['rotation_change'] = df['total_rotation'].diff().fillna(0)
        
        # Power-related features
        if 'throttle' in df.columns:
            df['throttle_squared'] = df['throttle']**2
            df['throttle_change'] = df['throttle'].diff().fillna(0)
            df['throttle_acceleration'] = df['throttle_change'].diff().fillna(0)
        
        # Position and movement features
        if {'latitude', 'longitude', 'altitude'}.issubset(df.columns):
            # Calculate velocity and acceleration
            df['vertical_speed'] = df['altitude'].diff().fillna(0) / df['time_diff']
            df['vertical_accel'] = df['vertical_speed'].diff().fillna(0) / df['time_diff']
            
            # Approximate horizontal movement
            df['lat_change'] = df['latitude'].diff().fillna(0)
            df['lon_change'] = df['longitude'].diff().fillna(0)
            df['horizontal_movement'] = np.sqrt(df['lat_change']**2 + df['lon_change']**2)
            df['horizontal_speed'] = df['horizontal_movement'] / df['time_diff']
            
            # Combined movement features
            df['total_speed'] = np.sqrt(df['vertical_speed']**2 + df['horizontal_speed']**2)
            
        # Rolling statistics for key features
        for col in ['throttle', 'total_rotation', 'altitude']:
            if col in df.columns:
                for window in self.window_sizes:
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window, min_periods=1).mean()
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(window, min_periods=1).std()
                    df[f'{col}_rolling_max_{window}'] = df[col].rolling(window, min_periods=1).max()
        
        # Drop unnecessary columns and handle missing values
        cols_to_drop = ['timestamp', 'latitude', 'longitude']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        
        return df.fillna(0)

class EnhancedBatteryDataset(Dataset):
    def __init__(self, features, targets=None, seq_length=32):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.features) - self.seq_length + 1
    
    def __getitem__(self, idx):
        feature_seq = self.features[idx:idx + self.seq_length]
        if self.targets is not None:
            target = self.targets[idx + self.seq_length - 1]
            return feature_seq, target
        return feature_seq

def prepare_enhanced_data(data_dir, seq_length=32, batch_size=64):
    # Initialize feature extractor
    feature_extractor = EnhancedFeatureExtractor()
    
    # Load and process data
    def load_and_process(split):
        files = sorted(glob.glob(os.path.join(data_dir, split, '*.csv')))
        if not files:
            raise ValueError(f"No CSV files found in {os.path.join(data_dir, split)}")
        
        all_features = []
        all_targets = []
        
        for file in files:
            df = pd.read_csv(file)
            if 'battery_current' not in df.columns:
                raise ValueError(f"File {file} missing 'battery_current' column")
            
            # Extract target and compute features
            targets = df['battery_current'].values
            features = feature_extractor.compute_features(df)
            
            all_features.append(features)
            all_targets.append(targets)
        
        return pd.concat(all_features, ignore_index=True), np.concatenate(all_targets)
    
    # Load all splits
    X_train, y_train = load_and_process('train')
    X_val, y_val = load_and_process('val')
    X_test, y_test = load_and_process('test')
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Create datasets
    train_dataset = EnhancedBatteryDataset(X_train_scaled, y_train, seq_length)
    val_dataset = EnhancedBatteryDataset(X_val_scaled, y_val, seq_length)
    test_dataset = EnhancedBatteryDataset(X_test_scaled, y_test, seq_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader, X_train.shape[1] 