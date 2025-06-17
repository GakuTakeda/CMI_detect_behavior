import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam                
from torch.optim.lr_scheduler import ReduceLROnPlateau  
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data.dataloader import default_collate

def feature_eng(df: pd.DataFrame) -> pd.DataFrame:
    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    df['rot_angle'] = 2 * np.arccos(df['rot_w'].clip(-1, 1))
    df['acc_mag_jerk'] = df.groupby('sequence_id')['acc_mag'].diff().fillna(0)
    df['rot_angle_vel'] = df.groupby('sequence_id')['rot_angle'].diff().fillna(0)

    insert_cols = ['acc_mag', 'rot_angle', 'acc_mag_jerk', 'rot_angle_vel']
    cols = list(df.columns)

    for i, col in enumerate(cols):
        if col.startswith('thm_') or col.startswith('tof_'):
            insert_index = i
            break

    cols_wo_insert = [c for c in cols if c not in insert_cols]

    new_order = cols_wo_insert[:insert_index] + insert_cols + cols_wo_insert[insert_index:]
    df = df[new_order]

    return df


class TimeseriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)   # shape: (N, L, C)
        self.y = torch.as_tensor(y, dtype=torch.long)      # 1-D クラスラベル

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ResidualBlock(nn.Module):
    """Conv1D → BN → ReLU → Conv1D → BN → add & ReLU"""
    def __init__(self, ch_in, ch_out, k=3, p_drop=0.1):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(ch_in, ch_out, k, padding=pad)
        self.bn1   = nn.BatchNorm1d(ch_out)
        self.conv2 = nn.Conv1d(ch_out, ch_out, k, padding=pad)
        self.bn2   = nn.BatchNorm1d(ch_out)
        self.act   = nn.ReLU()
        self.dropout = nn.Dropout(p_drop)
        # 1×1 conv for channel-mismatch shortcut
        self.shortcut = nn.Conv1d(ch_in, ch_out, 1) if ch_in != ch_out else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.act(out)
        return out

def train(model, loader_tr, loader_val, epochs=50, lr=1e-3,
          weight_decay=1e-4, patience=5, device='cuda'):

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5, verbose=True)

    best_val = np.inf
    wait = 0

    for epoch in range(1, epochs + 1):
        # -------- train --------
        model.train()
        loss_tr = 0
        for X, y in loader_tr:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_tr += loss.item() * len(X)

        # -------- validate --------
        model.eval()
        loss_val = 0
        with torch.no_grad():
            for X, y in loader_val:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                loss = criterion(logits, y)
                loss_val += loss.item() * len(X)

        loss_tr /= len(loader_tr.dataset)
        loss_val /= len(loader_val.dataset)
        scheduler.step(loss_val)

        print(f'epoch {epoch:2d}: train {loss_tr:.4f}  val {loss_val:.4f}')

        # Early-Stopping check
        if loss_val < best_val:
            best_val = loss_val
            wait = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            wait += 1
            if wait >= patience:
                print('Early stopping triggered')
                break

time_sum          = lambda x: x.sum(dim=2)            # (N, C, L) → (N, C)
squeeze_last_axis = lambda x: x.squeeze(-1)           # (… ,1) → (…)
expand_last_axis  = lambda x: x.unsqueeze(-1)         # (… )   → (…,1)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)           # GlobalAveragePooling1D
        self.fc1  = nn.Linear(channels, channels // reduction)
        self.fc2  = nn.Linear(channels // reduction, channels)

    def forward(self, x):                             # x: (N, C, L)
        z = self.pool(x).squeeze(-1)                  # (N, C)
        z = F.relu(self.fc1(z))
        z = torch.sigmoid(self.fc2(z))                # (N, C)
        z = z.unsqueeze(-1)                           # (N, C,1)
        return x * z                                  # channel-wise 補正

class ResidualSEBlock(nn.Module):
    """
    2×Conv1D + BN + ReLU → SE → Add → ReLU → MaxPool → Dropout
    """
    def __init__(self, in_ch, out_ch,
                 k=3, pool_size=2, drop=0.3, wd=1e-4):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=pad,
                               bias=False, padding_mode='zeros')
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=pad,
                               bias=False, padding_mode='zeros')
        self.bn2   = nn.BatchNorm1d(out_ch)

        # Shortcut のチャネル数調整
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1, bias=False) \
                        if in_ch != out_ch else nn.Identity()
        self.bn_sc = nn.BatchNorm1d(out_ch) if in_ch != out_ch else nn.Identity()

        self.se     = SEBlock(out_ch)
        self.pool   = nn.MaxPool1d(pool_size)
        self.drop   = nn.Dropout(drop)
        self.weight_decay = wd   # ⇒ Optimizer で weight_decay に設定

    def forward(self, x):                    # x: (N, C_in, L)
        residual = self.bn_sc(self.shortcut(x))
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = F.relu(out + residual)
        out = self.pool(out)
        out = self.drop(out)
        return out                           # (N, C_out, L')
class EnhancedSEBlock(nn.Module):
    """
    An enhanced Squeeze-and-Excitation block that uses both average and max pooling,
    inspired by the reference implementation.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction, bias=False),
            nn.SiLU(inplace=True),  # Using SiLU (swish) as in TF reference
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        avg_y = self.avg_pool(x).view(b, c)
        max_y = self.max_pool(x).view(b, c)
        y = torch.cat([avg_y, max_y], dim=1)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)

class MultiScaleConv1d(nn.Module):
    """Multi-scale temporal convolution block"""
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.convs = nn.ModuleList()
        for ks in kernel_sizes:
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, ks, padding=ks//2, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
    def forward(self, x):
        outputs = [conv(x) for conv in self.convs]
        return torch.cat(outputs, dim=1)
class MetaFeatureExtractor(nn.Module):
    """Extract statistical meta-features from input sequence"""
    def forward(self, x):
        # x shape: (B, L, C)
        mean = torch.mean(x, dim=1)
        std = torch.std(x, dim=1)
        max_val, _ = torch.max(x, dim=1)
        min_val, _ = torch.min(x, dim=1)
        
        # Calculate slope: (last - first) / seq_len
        seq_len = x.size(1)
        if seq_len > 1:
            slope = (x[:, -1, :] - x[:, 0, :]) / (seq_len - 1)
        else:
            slope = torch.zeros_like(x[:, 0, :])
        
        return torch.cat([mean, std, max_val, min_val, slope], dim=1)

class AttentionLayer(nn.Module):
    """
    inputs: (N, T, D)
    出力  : (N, D)   — 時系列方向に重み付けした文脈ベクトル
    """
    def __init__(self, d_model):
        super().__init__()
        self.score_fc = nn.Linear(d_model, 1)   # Dense(1, tanh)

    def forward(self, x):                       # x: (N, T, D)
        # (N, T, 1) → squeeze → softmax over T
        score = torch.tanh(self.score_fc(x)).squeeze(-1)  # (N, T)
        weights = F.softmax(score, dim=1).unsqueeze(-1)   # (N, T,1)
        context = (x * weights).sum(dim=1)                # (N, D)
        return context

class CNN_BiLSTM_Att(nn.Module):
    def __init__(self, n_classes, n_features, lstm_units=128):
        super().__init__()
        # Conv1d は (N,C,L) なので転置の方針は前回と同じ
        self.block1 = ResidualSEBlock(n_features, 64, k=5)
        self.block2 = ResidualSEBlock(64, 128, k=3)

        self.bilstm = nn.LSTM(
            input_size=128, hidden_size=lstm_units,
            batch_first=True, bidirectional=True
        )
        self.att = AttentionLayer(2 * lstm_units)   # 双方向 ⇒ 2×units
        self.fc  = nn.Linear(2 * lstm_units, n_classes)

    def forward(self, x):        # x: (N, L, C)
        x = x.transpose(1, 2)    # → (N, C, L)
        x = self.block1(x)
        x = self.block2(x)       # (N, 128, L')
        x = x.transpose(1, 2)    # → (N, L', 128)

        x, _ = self.bilstm(x)    # (N, L', 2*units)
        x = self.att(x)          # (N, 2*units)
        return self.fc(x)

def preprocess_sequence(
        df_seq: pd.DataFrame,
        feature_cols: list[str],
        scaler: StandardScaler
) -> torch.Tensor:
    """
    • 欠損を ffill/bfill → 0 埋め
    • StandardScaler.transform → float32
    • torch.tensor に変換して返す  (shape: [T, C])
    """
    mat = (
        df_seq[feature_cols]
        .ffill().bfill().fillna(0)
        .values                                # ndarray
    )
    mat = scaler.transform(mat).astype("float32")
    return torch.from_numpy(mat)      

class SequenceDataset(Dataset):
    """
    X:  3-D Tensor / ndarray  (N, T, C)
    y:  2-D Tensor / ndarray  (N, num_classes) ―― one-hot でも float ラベルでも可
    """
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def mixup_collate_fn(alpha: float = 0.2):
    """α>0 なら MixUp，それ以外は普通にまとめる"""
    if alpha <= 0:
        # そのままスタックする PyTorch 標準の collate を返す
        return default_collate

    def _collate(batch):
        X, y = zip(*batch)
        X = torch.stack(X, dim=0)      # (B, T, C)
        y = torch.stack(y, dim=0)      # (B, num_classes)

        lam  = np.random.beta(alpha, alpha)
        perm = torch.randperm(X.size(0))

        X_mix = lam * X + (1 - lam) * X[perm]
        y_mix = lam * y + (1 - lam) * y[perm]
        return X_mix, y_mix

    return _collate

class GaussianNoise(nn.Module):
    def __init__(self, stddev=0.1):
        super().__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training and self.stddev > 0:
            noise = torch.randn_like(x) * self.stddev
            return x + noise
        return x

class TwoBranchModel(nn.Module):
    def __init__(self, imu_dim, tof_dim, n_classes):
        super().__init__()
        self.imu_dim = imu_dim
        self.tof_dim = tof_dim

        # IMU branch
        self.imu_branch = nn.Sequential(
            ResidualSEBlock(imu_dim, 64, 3, drop=0.1),
            ResidualSEBlock(64, 128, 5, drop=0.1)
        )

        # TOF branch
        self.tof_branch = nn.Sequential(
            nn.Conv1d(tof_dim, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )

        # RNN & Dense branches
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, batch_first=True, bidirectional=True)
        self.gru = nn.GRU(input_size=256, hidden_size=128, batch_first=True, bidirectional=True)
        self.noise = GaussianNoise(0.09)
        self.fc_noise = nn.Linear(256, 16)
        self.attn = AttentionLayer(128*2 + 128*2 + 16)

        self.fc = nn.Sequential(
            nn.Linear(128*4 + 16, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        # x: (B, T, D) where D = imu_dim + tof_dim
        imu = x[:, :, :self.imu_dim].permute(0, 2, 1)  # (B, C, T)
        tof = x[:, :, self.imu_dim:].permute(0, 2, 1)

        x1 = self.imu_branch(imu)  # (B, C, T)
        x2 = self.tof_branch(tof)  # (B, C, T)

        merged = torch.cat([x1, x2], dim=1).permute(0, 2, 1)  # → (B, T, D_merged)

        xa, _ = self.lstm(merged)
        xb, _ = self.gru(merged)

        xc = self.noise(merged)
        xc = F.elu(self.fc_noise(xc))

        x = torch.cat([xa, xb, xc], dim=2)
        x = self.attn(x)
        return self.fc(x)

class ModelVariant_GRU(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        num_channels = 11  # Hardcoded for IMU data

        # 1. 
        self.meta_extractor = MetaFeatureExtractor()
        self.meta_dense = nn.Sequential(
            nn.Linear(5 * num_channels, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # 2. 
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    # 输出通道数减少，从16*3=48 -> 12*3=36
                    MultiScaleConv1d(1, 12, kernel_sizes=[3, 5, 7]),
                    # 输出通道数相应调整
                    ResidualSEBlock(36, 48, 3, drop=0.3),
                    ResidualSEBlock(48, 48, 3, drop=0.3),
                )
                for _ in range(num_channels)
            ]
        )

        # 3. 序列核心：使用BiGRU替换BiLSTM，并移除MultiHeadAttention
        self.bigru = nn.GRU(
            input_size=48 * num_channels,
            hidden_size=128,  # 保持hidden_size
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

        # 4. Attention Pooling层保持不变
        self.attention_pooling = AttentionLayer(256)  # 128 * 2 for bidirectional

        # 5. 
        self.head_1 = nn.Sequential(
            nn.Linear(256 + 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        self.head_2 = nn.Sequential(
            nn.Linear(256 + 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),  # For regression task
        )

    def forward(self, x: torch.Tensor):
        # Meta features
        meta = self.meta_extractor(x)
        meta_proj = self.meta_dense(meta)

        # CNN branches
        branch_outputs = []
        for i in range(x.shape[2]):
            channel_input = x[:, :, i].unsqueeze(1)
            processed = self.branches[i](channel_input)
            branch_outputs.append(processed.transpose(1, 2))

        combined = torch.cat(branch_outputs, dim=2)

        # BiGRU processing
        gru_out, _ = self.bigru(combined)  # (B, L/k, 256)

        # Attention pooling
        pooled_output = self.attention_pooling(gru_out)  # (B, 256)

        # Combine with meta features
        fused = torch.cat([pooled_output, meta_proj], dim=1)

        # Final predictions
        z1 = self.head_1(fused)
        z2 = self.head_2(fused)
        return z1, z2