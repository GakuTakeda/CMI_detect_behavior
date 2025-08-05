import os
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
from scipy.spatial.transform import Rotation as R
import copy
import warnings
import random
from scipy.signal import firwin
from typing import Sequence
from torch.nn import init
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from scipy.signal import cont2discrete
from typing import List

def phase_to_int(value):
    if value in ["Relaxes and moves hand to target location", "Moves hand to target location"]:
        return 0
    elif value == "Performs gesture":
        return 1
    else:
        return ["Relaxes and moves hand to target location", "Moves hand to target location", "Performs gesture"]



def remove_outliers(df: pd.DataFrame, threshold: int = 300) -> pd.DataFrame:
    df = df.copy()
    tof_thm_cols = [c for c in df.columns if c.startswith("thm") or c.startswith("tof")]
    null_number = df[tof_thm_cols].eq(-1).sum().sum()/df["sequence_counter"].max()   
    if null_number > threshold:
        return None
    else:
        return df

def remove_gravity_from_acc(acc_data, rot_data):


    if isinstance(acc_data, pd.DataFrame):
        acc_values = acc_data[['acc_x', 'acc_y', 'acc_z']].values
    else:
        acc_values = acc_data

    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

    num_samples = acc_values.shape[0]
    linear_accel = np.zeros_like(acc_values)
    
    gravity_world = np.array([0, 0, 9.81])

    for i in range(num_samples):
        if np.all(np.isnan(quat_values[i])) or np.all(np.isclose(quat_values[i], 0)):
            linear_accel[i, :] = acc_values[i, :] 
            continue

        try:
            rotation = R.from_quat(quat_values[i])
            gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
            linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
        except ValueError:
             linear_accel[i, :] = acc_values[i, :]
             
    return linear_accel

def remove_gravity_from_acc_in_train(df: pd.DataFrame) -> pd.DataFrame:
    acc_cols = [c for c in df.columns if c.startswith('acc_')]
    rot_cols = [c for c in df.columns if c.startswith('rot_')]
    # 出力用の空配列を準備（順序を保つ）
    linear_acc_all = np.zeros((len(df), 3), dtype=np.float32)
    
    for _, seq in df.groupby('sequence_id'):
        idx = seq.index                     # 元の行番号を保存
        linear_acc = remove_gravity_from_acc(seq[acc_cols], seq[rot_cols])
        linear_acc_all[idx, :] = linear_acc  # 元の位置に代入
    
    # 新しい列として DataFrame に追加
    df[['linear_acc_x', 'linear_acc_y', 'linear_acc_z']] = linear_acc_all
    return df

def calculate_angular_velocity_from_quat(rot_data, time_delta=1/200): # Assuming 200Hz sampling rate
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

    num_samples = quat_values.shape[0]
    angular_vel = np.zeros((num_samples, 3))

    for i in range(num_samples - 1):
        q_t = quat_values[i]
        q_t_plus_dt = quat_values[i+1]

        if np.all(np.isnan(q_t)) or np.all(np.isclose(q_t, 0)) or \
           np.all(np.isnan(q_t_plus_dt)) or np.all(np.isclose(q_t_plus_dt, 0)):
            continue

        try:
            rot_t = R.from_quat(q_t)
            rot_t_plus_dt = R.from_quat(q_t_plus_dt)

            # Calculate the relative rotation
            delta_rot = rot_t.inv() * rot_t_plus_dt
            
            # Convert delta rotation to angular velocity vector
            # The rotation vector (Euler axis * angle) scaled by 1/dt
            # is a good approximation for small delta_rot
            angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except ValueError:
            # If quaternion is invalid, angular velocity remains zero
            pass
            
    return angular_vel

def calculate_angular_distance(rot_data):
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

    num_samples = quat_values.shape[0]
    angular_dist = np.zeros(num_samples)

    for i in range(num_samples - 1):
        q1 = quat_values[i]
        q2 = quat_values[i+1]

        if np.all(np.isnan(q1)) or np.all(np.isclose(q1, 0)) or \
           np.all(np.isnan(q2)) or np.all(np.isclose(q2, 0)):
            angular_dist[i] = 0 # Или np.nan, в зависимости от желаемого поведения
            continue
        try:
            # Преобразование кватернионов в объекты Rotation
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)

            # Вычисление углового расстояния: 2 * arccos(|real(p * q*)|)
            # где p* - сопряженный кватернион q
            # В scipy.spatial.transform.Rotation, r1.inv() * r2 дает относительное вращение.
            # Угол этого относительного вращения - это и есть угловое расстояние.
            relative_rotation = r1.inv() * r2
            
            # Угол rotation vector соответствует угловому расстоянию
            # Норма rotation vector - это угол в радианах
            angle = np.linalg.norm(relative_rotation.as_rotvec())
            angular_dist[i] = angle
        except ValueError:
            angular_dist[i] = 0 # В случае недействительных кватернионов
            pass
            
    return angular_dist


def labeling_for_macro(value):

    BFRB = [
        "Above ear - pull hair", "Forehead - pull hairline", "Forehead - scratch",
        "Eyebrow - pull hair", "Eyelash - pull hair", "Neck - pinch skin",
        "Neck - scratch", "Cheek - pinch skin"
    ]
    
    # BFRBリスト内の値であれば、そのインデックスを返す
    if value in BFRB:
        return BFRB.index(value)

    elif value == "classes":
        return [
            "Above ear - pull hair", "Forehead - pull hairline", "Forehead - scratch",
            "Eyebrow - pull hair", "Eyelash - pull hair", "Neck - pinch skin",
            "Neck - scratch", "Cheek - pinch skin"]

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def quaternion_to_euler(qx, qy, qz, qw):
    """
    XYZ (roll, pitch, yaw) への変換
    返り値: (roll[rad], pitch[rad], yaw[rad])
    """
    # roll (x 軸回り)
    sinr = 2.0 * (qw * qx + qy * qz)
    cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr, cosr)

    # pitch (y 軸回り)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = np.where(np.abs(sinp) >= 1,
                     np.sign(sinp) * np.pi/2,           # gimbal lock
                     np.arcsin(sinp))

    # yaw (z 軸回り)
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny, cosy)

    return roll, pitch, yaw


def Macro_eng(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """
    既存 15 ch + 姿勢角 3 ch + 移動平均/分散 (各 15 ch) ＝ 48 ch
    """
    df = df.sort_values(['sequence_id', 'sequence_counter']).copy()

    # --- 既存 15 ch ------------------------------------------------------
    df['acc_mag']        = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    df['rot_angle']      = 2 * np.arccos(df['rot_w'].clip(-1, 1))
    df['acc_mag_jerk']   = df.groupby('sequence_id')['acc_mag'].diff().fillna(0)
    df['rot_angle_vel']  = df.groupby('sequence_id')['rot_angle'].diff().fillna(0)

    rot_cols = ['rot_x', 'rot_y', 'rot_z', 'rot_w']
    rot_data = df[rot_cols].to_numpy()
    ang_vel  = calculate_angular_velocity_from_quat(rot_data)
    df[['angular_vel_x', 'angular_vel_y', 'angular_vel_z']] = ang_vel
    df['angular_dist']   = calculate_angular_distance(rot_data)

    # --- (1) 姿勢系: pitch / roll / yaw ---------------------------------
    roll, pitch, yaw = quaternion_to_euler(df['rot_x'], df['rot_y'], df['rot_z'], df['rot_w'])
    df['roll']  = roll
    df['pitch'] = pitch
    df['yaw']   = yaw

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

class ResidualSECNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size=2, 
                 dropout=0.3, weight_decay=1e-4):
        super().__init__()
        
        # First conv block
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # Second conv block
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # SE block
        self.se = SEBlock(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        self.pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        shortcut = self.shortcut(x)
        
        # First conv
        out = F.relu(self.bn1(self.conv1(x)))
        # Second conv
        out = self.bn2(self.conv2(out))
        
        # SE block
        out = self.se(out)
        
        # Add shortcut
        out += shortcut
        out = F.relu(out)
        
        # Pool and dropout
        out = self.pool(out)
        out = self.dropout(out)
        
        return out


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

class EnhancedResidualSEBlock(nn.Module):
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

        self.se     = EnhancedSEBlock(out_ch)
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
        return out      


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

#データモジュール

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
    def __init__(self, X, y, augment=None):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.augment is not None:
            self.X[idx] = self.augment(self.X[idx])
            
        return self.X[idx], self.y[idx]
    
class SequenceDataset_for_tof(Dataset):
    """
    X:  3-D Tensor / ndarray  (N, T, C)
    y:  2-D Tensor / ndarray  (N, num_classes) ―― one-hot でも float ラベルでも可
    """
    def __init__(self, X, img, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.img = torch.as_tensor(img, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.img[idx], self.y[idx]

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

def pad_truncate(seq: np.ndarray, pad_len: int) -> torch.Tensor:

    # (T, C) → torch.Tensor
    t = torch.as_tensor(seq, dtype=torch.float32)

    T, C = t.shape
    if T >= pad_len:                     # --- truncating = 'post' ---
        t = t[:pad_len]                  # 後ろを切り捨て
    else:                                # --- padding = 'post' ---
        pad_size = (pad_len - T, C)      # 足りない分だけ 0 行を作る
        t = torch.cat([t, t.new_zeros(pad_size)], dim=0)

    return t.unsqueeze(0)  

class Augment:
    """
    時系列 IMU ＋ TOF 入力のデータ拡張 (PyTorch 版)

    Parameters
    ----------
    p_jitter : float
        ジッタ＋スケーリングを適用する確率
    sigma : float
        ジッタの標準偏差
    scale_range : (float, float)
        スケール拡張の下限・上限
    p_dropout : float
        センサ Drop-out を適用する確率
    p_moda : float
        Motion Drift (MODA) を適用する確率
    drift_std : float
        Drift の 1 ステップあたり標準偏差
    drift_max : float
        Drift の上限値 (絶対値でクリップ)
    """

    def __init__(
        self,
        imu_dim: int,
        p_jitter: float = 0.8,
        sigma: float = 0.02,
        scale_range: Sequence[float] = (0.9, 1.1),
        p_dropout: float = 0.3,
        p_moda: float = 0.5,
        drift_std: float = 0.005,
        drift_max: float = 0.25,
    ) -> None:

        self.imu_dim = imu_dim
        self.p_jitter  = p_jitter
        self.sigma     = sigma
        self.scale_min, self.scale_max = scale_range
        self.p_dropout = p_dropout
        self.p_moda    = p_moda
        self.drift_std = drift_std
        self.drift_max = drift_max

    # ---------- Jitter & Scaling ----------
    def jitter_scale(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (T, F)  時系列長 T、特徴次元 F
        """
        noise  = torch.randn_like(x) * self.sigma
        scale  = torch.empty(
            (1, x.shape[1]),
            device=x.device,
            dtype=x.dtype
        ).uniform_(self.scale_min, self.scale_max)
        return (x + noise) * scale

    # ---------- Sensor Drop-out ----------
    def sensor_dropout(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1, device=x.device) < self.p_dropout:
            x = x.clone()              # 勾配履歴を保持
            x[:, self.imu_dim:] = 0.0
        return x

    # ---------- Motion Drift (MODA) ----------
    def motion_drift(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[0]

        drift = torch.randn(
            (T, 1), device=x.device, dtype=x.dtype
        ) * self.drift_std             # 正規乱数
        drift = torch.cumsum(drift, dim=0)
        drift = drift.clamp(-self.drift_max, self.drift_max)

        x = x.clone()
        x[:, :6] += drift              # 3軸加速度＋3軸角速度
        if self.imu_dim > 6:
            x[:, 6:self.imu_dim] += drift   # 磁気・気圧などがある場合
        return x

    # ---------- master call ----------
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            入力系列 (T, F).  勾配不要ならあらかじめ `with torch.no_grad()` 推奨
        imu_dim : int
            x の IMU 部分の次元数 (TOF など他センサとの境界)
        """
        if torch.rand(1, device=x.device) < self.p_jitter:
            x = self.jitter_scale(x)

        if torch.rand(1, device=x.device) < self.p_moda:
            x = self.motion_drift(x)

        x = self.sensor_dropout(x)
        return x

class ConvModule1D(nn.Module):
    def __init__(self, d_model: int, conv_kernel: int = 31, dropout: float = 0.1):
        super().__init__()
        self.ln   = nn.LayerNorm(d_model)
        self.pw1  = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.dw   = nn.Conv1d(d_model, d_model, kernel_size=conv_kernel,
                              padding=conv_kernel // 2, groups=d_model)
        self.bn   = nn.BatchNorm1d(d_model)
        self.act  = nn.SiLU()
        self.pw2  = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                 # x: (B, T, D)
        x = self.ln(x)                    # (B, T, D)
        x = x.transpose(1, 2)             # (B, D, T)
        x = self.pw1(x)                   # (B, 2D, T)
        x = F.glu(x, dim=1)               # (B, D, T)
        x = self.dw(x)                    # (B, D, T)
        x = self.bn(x)
        x = self.act(x)
        x = self.pw2(x)                   # (B, D, T)
        x = self.drop(x)
        return x.transpose(1, 2)          # (B, T, D) に戻す
    

class ConformerBlock(nn.Module):
    """FFN → MHSA → ConvModule → FFN（各残差に 0.5 係数）"""
    def __init__(self, d_model=256, n_heads=4,
                 ffn_expansion=4, conv_kernel=31, dropout=0.1):
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * ffn_expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_expansion, d_model),
            nn.Dropout(dropout),
        )
        self.mhsa     = nn.MultiheadAttention(d_model, n_heads,
                                              dropout=dropout, batch_first=True)
        self.ln_mhsa  = nn.LayerNorm(d_model)
        self.conv_mod = ConvModule1D(d_model, conv_kernel, dropout)
        self.ffn2     = copy.deepcopy(self.ffn1)

    def forward(self, x):                 # (B, T, D)
        x = x + 0.5 * self.ffn1(x)
        x = x + self.ln_mhsa(self.mhsa(x, x, x)[0])
        x = x + self.conv_mod(x)
        x = x + 0.5 * self.ffn2(x)
        return x

class GaussianNoise(nn.Module):
    def __init__(self, stddev=0.1):
        super().__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training and self.stddev > 0:
            noise = torch.randn_like(x) * self.stddev
            return x + noise
        return x
    
def leCunUniform(tensor):
    """ 
        LeCun Uniform Initializer
        References: 
        [1] https://keras.rstudio.com/reference/initializer_lecun_uniform.html
        [2] Source code of _calculate_correct_fan can be found in https://pytorch.org/docs/stable/_modules/torch/nn/init.html
        [3] Yann A LeCun, Léon Bottou, Genevieve B Orr, and Klaus-Robert Müller. Efficient backprop. In Neural networks: Tricks of the trade, pages 9–48. Springer, 2012
    """
    fan_in = init._calculate_correct_fan(tensor, "fan_in")
    limit = np.sqrt(3. / fan_in)
    init.uniform_(tensor, -limit, limit) # fills the tensor with values sampled from U(-limit, limit)

class LMUCell(nn.Module):
    """ 
    LMU Cell

    Parameters:
        input_size (int) : 
            Size of the input vector (x_t)
        hidden_size (int) : 
            Size of the hidden vector (h_t)
        memory_size (int) :
            Size of the memory vector (m_t)
        theta (int) :
            The number of timesteps in the sliding window that is represented using the LTI system
        learn_a (boolean) :
            Whether to learn the matrix A (default = False)
        learn_b (boolean) :
            Whether to learn the matrix B (default = False)
    """

    def __init__(self, input_size, hidden_size, memory_size, theta, learn_a = False, learn_b = False):
        
        super(LMUCell, self).__init__()

        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.f = nn.Tanh()

        A, B = self.stateSpaceMatrices(memory_size, theta)
        A = torch.from_numpy(A).float()
        B = torch.from_numpy(B).float()

        if learn_a:
            self.A = nn.Parameter(A)
        else:
            self.register_buffer("A", A)
    
        if learn_b:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)

        # Declare Model parameters:
        ## Encoding vectors
        self.e_x = nn.Parameter(torch.empty(1, input_size))
        self.e_h = nn.Parameter(torch.empty(1, hidden_size))
        self.e_m = nn.Parameter(torch.empty(1, memory_size))
        ## Kernels
        self.W_x = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_h = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_m = nn.Parameter(torch.empty(hidden_size, memory_size))

        self.initParameters()

    def initParameters(self):
        """ Initialize the cell's parameters """

        # Initialize encoders
        leCunUniform(self.e_x)
        leCunUniform(self.e_h)
        init.constant_(self.e_m, 0)
        # Initialize kernels
        init.xavier_normal_(self.W_x)
        init.xavier_normal_(self.W_h)
        init.xavier_normal_(self.W_m)

    def stateSpaceMatrices(self, memory_size, theta):
        """ Returns the discretized state space matrices A and B """

        Q = np.arange(memory_size, dtype = np.float64).reshape(-1, 1)
        R = (2*Q + 1) / theta
        i, j = np.meshgrid(Q, Q, indexing = "ij")

        # Continuous
        A = R * np.where(i < j, -1, (-1.0)**(i - j + 1))
        B = R * ((-1.0)**Q)
        C = np.ones((1, memory_size))
        D = np.zeros((1,))

        # Convert to discrete
        A, B, C, D, dt = cont2discrete(
            system = (A, B, C, D), 
            dt = 1.0, 
            method = "zoh"
        )
        
        return A, B

    def forward(self, x, state):
        """
        Parameters:
            x (torch.tensor): 
                Input of size [batch_size, input_size]
            state (tuple): 
                h (torch.tensor) : [batch_size, hidden_size]
                m (torch.tensor) : [batch_size, memory_size]
        """

        h, m = state

        # Equation (7) of the paper
        u = F.linear(x, self.e_x) + F.linear(h, self.e_h) + F.linear(m, self.e_m) # [batch_size, 1]

        # Equation (4) of the paper
        m = F.linear(m, self.A) + F.linear(u, self.B) # [batch_size, memory_size]

        # Equation (6) of the paper
        h = self.f(
            F.linear(x, self.W_x) +
            F.linear(h, self.W_h) + 
            F.linear(m, self.W_m)
        ) # [batch_size, hidden_size]

        return h, m
    
class LMU(nn.Module):
    """ 
    LMU layer

    Parameters:
        input_size (int) : 
            Size of the input vector (x_t)
        hidden_size (int) : 
            Size of the hidden vector (h_t)
        memory_size (int) :
            Size of the memory vector (m_t)
        theta (int) :
            The number of timesteps in the sliding window that is represented using the LTI system
        learn_a (boolean) :
            Whether to learn the matrix A (default = False)
        learn_b (boolean) :
            Whether to learn the matrix B (default = False)
    """

    def __init__(self, input_size, hidden_size, memory_size, theta, learn_a = False, learn_b= False):

        super(LMU, self).__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.cell = LMUCell(input_size, hidden_size, memory_size, theta, learn_a, learn_b)


    def forward(self, x, state = None):
        """
        Parameters:
            x (torch.tensor): 
                Input of size [batch_size, seq_len, input_size]
            state (tuple) : (default = None) 
                h (torch.tensor) : [batch_size, hidden_size]
                m (torch.tensor) : [batch_size, memory_size]
        """
        
        # Assuming batch dimension is always first, followed by seq. length as the second dimension
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Initial state (h_0, m_0)
        if state == None:
            h_0 = torch.zeros(batch_size, self.hidden_size)
            m_0 = torch.zeros(batch_size, self.memory_size)
            if x.is_cuda:
                h_0 = h_0.cuda()
                m_0 = m_0.cuda()
            state = (h_0, m_0)

        # Iterate over the timesteps
        output = []
        for t in range(seq_len):
            x_t = x[:, t, :] # [batch_size, input_size]
            h_t, m_t = self.cell(x_t, state)
            state = (h_t, m_t)
            output.append(h_t)

        output = torch.stack(output) # [seq_len, batch_size, hidden_size]
        output = output.permute(1, 0, 2) # [batch_size, seq_len, hidden_size]

        return output, state # state is (h_n, m_n) where n = seq_len
    
class rnn(nn.Module):    #output:ch*48
    def __init__(self, num_channels, attn_ch = 2944):

        super().__init__()
        # 2. Per-channel CNN branches
        self.num_channels = num_channels

        self.branches = nn.ModuleList([
            nn.Sequential(
                MultiScaleConv1d(1, 12, kernel_sizes=[3, 5, 7]),
                EnhancedResidualSEBlock(36, 48, 3, drop=0.3),
                EnhancedResidualSEBlock(48, 48, 3, drop=0.3),
            )
            for _ in range(num_channels)
        ])

        # 3-a. BiGRU
        self.bigru = nn.GRU(
            input_size=48 * num_channels,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

        # 3-b. **BiLSTM を追加**
        self.bilstm = nn.LSTM(
            input_size=48 * num_channels,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.lmu = LMU(
            input_size   = 48 * num_channels,
            hidden_size  = 128,
            memory_size  = 256,   # M の次元
            theta        = 127,   # 記憶のタイムスケール
        )

        self.noise = GaussianNoise(0.09)

        self.attention_pooling = AttentionLayer(attn_ch)

    def forward(self, x):
        branch_outs = []
        for i in range(self.num_channels):                # each channel
            ci = x[:, :, i].unsqueeze(1)           # (B,1,L)
            out = self.branches[i](ci)             # (B,48,L')
            branch_outs.append(out.transpose(1, 2))  # → (B,L',48)

        combined = torch.cat(branch_outs, dim=2)   # (B,L',48*C)

        # ---------- RNN ----------
        gru_out, _  = self.bigru(combined)         # (B,L',256)
        lstm_out, _ = self.bilstm(combined)        # (B,L',256)
        lmu_out, _ = self.lmu(combined)
        noise_out = self.noise(combined)

        rnn_cat = torch.cat([gru_out, lstm_out, lmu_out, noise_out], dim=2)  # (B,L',512)

        # ---------- Attention pooling ----------
        pooled = self.attention_pooling(rnn_cat)   # (B,512)

        return pooled        
    
class move_block(nn.Module):
    def __init__(self, imu_ch: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert imu_ch % n_heads == 0, "d_model は n_heads で割り切れる必要があります"

        self.mhsa = nn.MultiheadAttention(
            embed_dim=imu_ch,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True         # ← [B, L, C] をそのまま渡せる
        )
        self.ln1 = nn.LayerNorm(imu_ch)

        # Position-wise FFN（Transformer エンコーダ層と同じ構成）
        self.ffn = nn.Sequential(
            nn.Linear(imu_ch, 4 * imu_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * imu_ch, imu_ch),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(imu_ch)

    def forward(self, x):           # x : [B, L, C]
        # ① 自己注意（クエリ=キー=バリュー）
        attn_out, _ = self.mhsa(x, x, x)     # [B, L, C]

        # ② 残差 + LayerNorm
        x = self.ln1(x + attn_out)

        # ③ FFN
        ffn_out = self.ffn(x)

        # ④ 残差 + LayerNorm
        out = self.ln2(x + ffn_out)
        return out #imu_ch
    
class MacroImuModel(nn.Module):
    """
    IMU 専用：CNN → BiGRU ＆ BiLSTM → AttentionPooling → 2 ヘッド
    """
    def __init__(self, num_classes: int, len_move: int, len_perform: int):
        super().__init__()
        num_channels = 18  # IMU チャンネル数

        self.len_move = len_move
        self.len_perform = len_perform

        # 1. Meta features
        self.meta_extractor = MetaFeatureExtractor()
        self.meta_dense = nn.Sequential(
            nn.Linear(5 * num_channels, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        attn_ch = 1504
        # 5. Prediction heads
        in_feat = attn_ch + 32*2  + num_channels# pooled + meta
        self.head = nn.Sequential(
            nn.Linear(in_feat, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        self.rnn = rnn(num_channels, attn_ch=attn_ch)

        self.mhsa = move_block(imu_ch=num_channels, n_heads=3)

        self.attn_pooling = AttentionLayer(num_channels)


    def forward(self, x: torch.Tensor):
        """
        x: (B, L, C)  – L=系列長, C=16ch
        """
        # ---------- meta ----------
        x_1 = x[:, :self.len_move, :-1]
        x_2 = x[:, self.len_move:, :-1]
        phase = x[..., -1]

        meta_proj_1 = self.meta_dense(self.meta_extractor(x_1) )          # (B, 32)
        meta_proj_2 = self.meta_dense(self.meta_extractor(x_2) )          # (B, 32)

        pooled = self.rnn(x[..., :-1])          

        attn_1 = self.attn_pooling(self.mhsa(x_1))
        #attn_2 = self.attn_pooling(self.mhsa(x_2))

        # ---------- Fuse & heads ----------
        fused = torch.cat([pooled, meta_proj_1, meta_proj_2, attn_1], dim=1)  # (B,544)
        z_cls = self.head(fused)
        return z_cls


def _scan_lengths(df: pd.DataFrame):
    '''
    Scan all sequences and return (max_move_len, max_perform_len).
    '''
    len_move = 0
    len_perform = 0

    for _, seq in df.groupby('sequence_id'):
        arr = seq['behavior_int'].to_numpy()

        nz0 = np.argmax(arr != 0) if np.any(arr != 0) else len(arr)
        len_move = max(len_move, nz0)

        sub = arr[nz0:]
        nz1 = np.argmax(sub != 1) if np.any(sub != 1) else len(sub)
        len_perform = max(len_perform, nz1)

    return len_move, len_perform


class GestureSequence(Dataset):
    """Lazy Sequence dataset returning (x, y_int)."""

    def __init__(
        self,
        df: pd.DataFrame,
        scaler: StandardScaler,
        feat_cols: List[str],
        imu_idx: List[int],
        len_move: int,
        len_perform: int,
    ) -> None:
        super().__init__()
        self.groups = list(df.groupby("sequence_id", sort=False))
        self.scaler = scaler
        self.feat_cols = feat_cols
        self.imu_idx = imu_idx
        self.len_move = len_move
        self.len_perform = len_perform
        self.F = len(feat_cols)

    # ---- utils ----
    def _pad(self, arr: np.ndarray, max_len: int) -> np.ndarray:
        if arr.size == 0:
            return np.zeros((max_len, self.F), np.float16)
        if len(arr) >= max_len:
            return arr[:max_len]
        pad = np.repeat(arr[-1:], max_len - len(arr), axis=0)
        return np.vstack([arr, pad])

    # ---- dataset ----
    def __len__(self) -> int:  # noqa: D401
        return len(self.groups)

    def __getitem__(self, idx):  # noqa: D401
        _, seq = self.groups[idx]
        y_int = int(seq["gesture_int"].iloc[0])
        feat = seq[self.feat_cols].to_numpy()
        feat = self.scaler.transform(feat).astype(np.float16)
        phase = seq["behavior_int"].to_numpy(dtype=np.int8)
        move = feat[phase == 0]
        perf = feat[phase == 1]
        x = np.concatenate([
            self._pad(move, self.len_move),
            self._pad(perf, self.len_perform),
        ], axis=0)
        phase_pad = np.concatenate([
            np.zeros(self.len_move, np.int8),
            np.ones(self.len_perform, np.int8),
        ])[:, None]
        x = np.concatenate([x, phase_pad], axis=1)  # (L, F+1)
        x = x[:, self.imu_idx]
        return torch.from_numpy(x), y_int

# =====================================================
# collate
# =====================================================

def collate_pad(batch):
    xs, ys = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    xs_pad = []
    for x in xs:
        pad_len = max_len - x.shape[0]
        if pad_len:
            pad = x.new_full((pad_len, x.shape[1]), -1)
            x = torch.cat([x, pad], 0)
        xs_pad.append(x)
    x_stack = torch.stack(xs_pad).float()
    x_stack = torch.nan_to_num(x_stack, nan=0.0, posinf=0.0, neginf=0.0)
    y_tensor = torch.tensor(ys, dtype=torch.long)
    return x_stack, y_tensor
# def make_collate_pad(alpha: float = 0.0):
#     """Return collate that pads & (optionally) MixUp on int targets."""

#     def _collate(batch):
#         xs, ys = zip(*batch)
#         max_len = max(x.shape[0] for x in xs)
#         xs_pad = []
#         for x in xs:
#             pad_len = max_len - x.shape[0]
#             if pad_len:
#                 pad = x.new_full((pad_len, x.shape[1]), -1)
#                 x = torch.cat([x, pad], 0)
#             xs_pad.append(x)
#         x_stack = torch.stack(xs_pad).float()
#         x_stack = torch.nan_to_num(x_stack, nan=0.0, posinf=0.0, neginf=0.0)
#         y_tensor = torch.tensor(ys, dtype=torch.long)

#         if alpha > 0.0:
#             lam = np.random.beta(alpha, alpha)
#             perm = torch.randperm(x_stack.size(0))
#             x_stack = lam * x_stack + (1 - lam) * x_stack[perm]
#             # MixUp for CE: keep two indices + lambda
#             return x_stack, y_tensor, y_tensor[perm], lam
#         return x_stack, y_tensor

#     return _collate


