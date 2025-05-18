import torch
import torch.nn as nn

class EnhancedBatteryPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=192, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Feature extraction layers with batch normalization
        self.feature_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout/2)
        )
        
        # Bidirectional LSTM with residual connections
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Position-wise feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size * 2)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size * 2)
        self.norm2 = nn.LayerNorm(hidden_size * 2)
        
        # Output layers with skip connections and batch normalization
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'norm' not in name:
                if len(param.shape) > 1:
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Reshape for batch normalization
        x_reshaped = x.view(batch_size * seq_len, -1)
        
        # Feature extraction with proper reshaping
        features = self.feature_net(x_reshaped)
        features = features.view(batch_size, seq_len, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)
        
        # Multi-head self-attention with residual connection
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        lstm_out = self.norm1(lstm_out + attn_out)
        
        # Position-wise FFN with residual connection
        ffn_out = self.ffn(lstm_out)
        combined = self.norm2(lstm_out + ffn_out)
        
        # Global pooling: combine mean and max pooling
        mean_pool = torch.mean(combined, dim=1)
        max_pool, _ = torch.max(combined, dim=1)
        seq_repr = torch.cat([mean_pool, max_pool], dim=1)
        
        # Final prediction
        return self.output_net(seq_repr)

    def compute_physics_loss(self, pred_current, throttle, voltage):
        # Optional: Add physics-based constraints
        predicted_power = pred_current * voltage
        expected_power = self.power_constraint(throttle)
        return nn.MSELoss()(predicted_power, expected_power) 