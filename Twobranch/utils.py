import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from typing import Sequence, Tuple, Union
import pandas as pd
from torch.utils.data.dataloader import default_collate
import lightning as L
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from torchmetrics.classification import MulticlassAccuracy
from sklearn.metrics import f1_score
import warnings
from scipy.spatial.transform import Rotation as R

class calc_f1:
    def __init__(self):
        self.target_gestures = [
            'Above ear - pull hair',
            'Cheek - pinch skin',
            'Eyebrow - pull hair',
            'Eyelash - pull hair',
            'Forehead - pull hairline',
            'Forehead - scratch',
            'Neck - pinch skin',
            'Neck - scratch',
        ]
        self.non_target_gestures = [
            'Write name on leg',
            'Wave hello',
            'Glasses on/off',
            'Text on phone',
            'Write name in air',
            'Feel around in tray and pull out an object',
            'Scratch knee/leg skin',
            'Pull air toward your face',
            'Drink from bottle/cup',
            'Pinch knee/leg skin'
        ]
        self.all_classes = self.target_gestures + self.non_target_gestures

    def binary_score(self, sol, sub):
        y_true_bin = [1 if i in self.target_gestures else 0 for i in sol]
        y_pred_bin = [1 if i in self.target_gestures else 0 for i in sub]
        f1_binary = f1_score(
            y_true_bin,
            y_pred_bin,
            pos_label=True,
            zero_division=0,
            average='binary'
            )
    
        return 0.5 * f1_binary   

    def macro_score(self, sol, sub):
        y_true_mc = [x if x in self.target_gestures else 'non_target' for x in sol]
        y_pred_mc = [x if x in self.target_gestures else 'non_target' for x in sub]

        # Compute macro F1 over all gesture classes
        f1_macro = f1_score(
            y_true_mc,
            y_pred_mc,
            average='macro',
            zero_division=0
        )
    
        return 0.5 * f1_macro

def labeling(value):

    BFRB = [
        "Above ear - pull hair", "Forehead - pull hairline", "Forehead - scratch",
        "Eyebrow - pull hair", "Eyelash - pull hair", "Neck - pinch skin",
        "Neck - scratch", "Cheek - pinch skin"
    ]
    
    non_BFRB = [
        "Drink from bottle/cup", "Glasses on/off", "Pull air toward your face",
        "Pinch knee/leg skin", "Scratch knee/leg skin", "Write name on leg",
        "Text on phone", "Feel around in tray and pull out an object",
        "Write name in air", "Wave hello"
    ]
    
    # BFRBリスト内の値であれば、そのインデックスを返す
    if value in BFRB:
        return BFRB.index(value)
    
    # non_BFRBリスト内の値であれば、BFRBの長さ + そのインデックスを返す
    elif value in non_BFRB:
        return len(BFRB) + non_BFRB.index(value)

    elif value == "classes":
        return [
            "Above ear - pull hair", "Forehead - pull hairline", "Forehead - scratch",
            "Eyebrow - pull hair", "Eyelash - pull hair", "Neck - pinch skin",
            "Neck - scratch", "Cheek - pinch skin", "Drink from bottle/cup", "Glasses on/off", "Pull air toward your face",
            "Pinch knee/leg skin", "Scratch knee/leg skin", "Write name on leg",
            "Text on phone", "Feel around in tray and pull out an object",
            "Write name in air", "Wave hello"
            ]
    
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

def feature_eng(df: pd.DataFrame) -> pd.DataFrame:
    df = remove_gravity_from_acc_in_train(df)
    # df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    # df['rot_angle'] = 2 * np.arccos(df['rot_w'].clip(-1, 1))
    # df['acc_mag_jerk'] = df.groupby('sequence_id')['acc_mag'].diff().fillna(0)
    # df['rot_angle_vel'] = df.groupby('sequence_id')['rot_angle'].diff().fillna(0)
    df['linear_acc_mag'] = np.sqrt(df['linear_acc_x']**2 + df['linear_acc_y']**2 + df['linear_acc_z']**2)
    df['linear_acc_mag_jerk'] = df.groupby('sequence_id')['linear_acc_mag'].diff().fillna(0)
    rot_data = df[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
    angular_vel_group = calculate_angular_velocity_from_quat(rot_data)
    df['angular_vel_x'] = angular_vel_group[:, 0]
    df['angular_vel_y'] = angular_vel_group[:, 1]
    df['angular_vel_z'] = angular_vel_group[:, 2]
    df['angular_distance'] = calculate_angular_distance(rot_data)

    tof_pixel_cols = [f"tof_{i}_v{p}" for i in range(1, 6) for p in range(64)]
    tof_data_np = df[tof_pixel_cols].replace(-1, np.nan).to_numpy()
    reshaped_tof = tof_data_np.reshape(len(df), 5, 64)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'Mean of empty slice'); warnings.filterwarnings('ignore', r'Degrees of freedom <= 0 for slice')
        mean_vals, std_vals = np.nanmean(reshaped_tof, axis=2), np.nanstd(reshaped_tof, axis=2)
        min_vals, max_vals = np.nanmin(reshaped_tof, axis=2), np.nanmax(reshaped_tof, axis=2)
    tof_agg_cols = []
    for i in range(1, 6):
        df[f'tof_{i}_mean'], df[f'tof_{i}_std'] = mean_vals[:, i-1], std_vals[:, i-1]
        df[f'tof_{i}_min'], df[f'tof_{i}_max'] = min_vals[:, i-1], max_vals[:, i-1]
        tof_agg_cols.extend([f'tof_{i}_mean', f'tof_{i}_std', f'tof_{i}_min', f'tof_{i}_max'])
    insert_cols = ['angular_vel_x', 'angular_vel_y', 'angular_vel_z', 'angular_distance']
    cols = list(df.columns)

    for i, col in enumerate(cols):
        if col.startswith('thm_') or col.startswith('tof_'):
            insert_index = i
            break

    cols_wo_insert = [c for c in cols if c not in insert_cols]

    new_order = cols_wo_insert[:insert_index] + insert_cols + cols_wo_insert[insert_index:]
    df = df[new_order]

    return df

def _pad(mat: np.ndarray, pad_len: int) -> np.ndarray:
    F = mat.shape[1]
    out = np.zeros((pad_len, F), dtype='float32')
    seq_len = min(len(mat), pad_len)
    out[:seq_len] = mat[:seq_len]
    return out

def seed_everything(seed: int = 42) -> None:
    """PyTorch で再現性を確保するための共通シード設定.

    Parameters
    ----------
    seed : int, default 42
        乱数シード
    """
    # ── Python & NumPy ──────────────────────────────
    os.environ["PYTHONHASHSEED"] = str(seed)          # hash ランダム化の固定
    random.seed(seed)
    np.random.seed(seed)

    # ── PyTorch ─────────────────────────────────────
    torch.manual_seed(seed)                           # CPU
    torch.cuda.manual_seed(seed)                      # GPU (現在プロセス)
    torch.cuda.manual_seed_all(seed)                  # GPU (全デバイス)

    # アルゴリズムを完全に決定論的に
    torch.use_deterministic_algorithms(True)

    # CuDNN / TF32 設定
    torch.backends.cudnn.deterministic = True         # 毎回同じ結果
    torch.backends.cudnn.benchmark = False            # 入力形状による自動チューニングを無効化
    torch.backends.cuda.matmul.allow_tf32 = False     # TF32 を無効化
    torch.backends.cudnn.allow_tf32 = False

    # CUBLAS のワークスペースを固定（torch インポート前に設定するのが安全）
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"   # または ":4096:8"

time_sum         = lambda x: torch.sum(x, dim=2)      # (B, C, L) → (B, C)
squeeze_last_dim = lambda x: x.squeeze(-1)            # ...,-1 を除去
expand_last_dim  = lambda x: x.unsqueeze(-1)          # ...,+1 を追加

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

# ─────────────────────────────────────────────
# 2. Squeeze‑and‑Excitation Block
# ─────────────────────────────────────────────
class SEBlock(nn.Module):
    """Channel‑wise Squeeze‑and‑Excitation for 1‑D signals.

    Input shape : (B, C, L)  ── channels‑first
    Output shape: (B, C, L)
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)           # GAP → (B, C, 1)
        self.fc1  = nn.Linear(channels, channels // reduction, bias=True)
        self.fc2  = nn.Linear(channels // reduction, channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        se = self.pool(x).view(b, c)                  # (B, C)
        se = F.relu(self.fc1(se), inplace=True)
        se = torch.sigmoid(self.fc2(se)).view(b, c, 1)
        return x * se                                 # broadcast multiply


# ─────────────────────────────────────────────
# 3. Residual CNN Block + SE
# ─────────────────────────────────────────────
class ResidualSEConvBlock(nn.Module):
    """2×Conv1D → SE → residual add → ReLU → MaxPool → Dropout."""
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : int,
        pool_size   : int = 2,
        drop        : float = 0.3,
        wd          : float = 1e-4,
    ):
        super().__init__()
        pad = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_channels)

        self.se    = SEBlock(out_channels)

        # 1×1 conv for dimension match (identity if C 同じ)
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
            ) if in_channels != out_channels else nn.Identity()
        )

        self.relu   = nn.ReLU(inplace=True)
        self.pool   = nn.MaxPool1d(pool_size)
        self.dropout= nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.se(out)

        out = self.relu(out + residual)
        out = self.pool(out)
        out = self.dropout(out)
        return out


# ─────────────────────────────────────────────
# 4. Temporal Attention Layer (additive, softmax along time)
# ─────────────────────────────────────────────
class AttentionPooling(nn.Module):
    """Applies tanh‑dense → softmax → Σ (weighted sum over L)."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.score = nn.Linear(embed_dim, 1)
        self.tanh  = nn.Tanh()

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, L, D)  ※channels‑lastで扱うほうが自然なので転置してください

        Returns
        -------
        context : torch.Tensor
            (B, D)  – 時系列を重み付き平均した文脈ベクトル
        weights : torch.Tensor
            (B, L)  – softmax アテンション重み（解析用に返却）
        """
        scores  = self.tanh(self.score(x)).squeeze(-1)     # (B, L)
        weights = torch.softmax(scores, dim=1)             # (B, L)
        context = torch.bmm(weights.unsqueeze(1), x)       # (B, 1, D)
        return context.squeeze(1), weights                 # (B, D), (B, L)


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
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)


            relative_rotation = r1.inv() * r2
            
            angle = np.linalg.norm(relative_rotation.as_rotvec())
            angular_dist[i] = angle
        except ValueError:
            angular_dist[i] = 0 
            pass
            
    return angular_dist

class GaussianNoise(nn.Module):
    """学習時のみ N(0, σ²) を加える."""
    def __init__(self, sigma: float = 0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training and self.sigma > 0.0:
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        return x


class DenseBN(nn.Module):
    """全結合 → BN → ReLU → Dropout."""
    def __init__(self, in_features, out_features, drop=0.0):
        super().__init__()
        self.fc   = nn.Linear(in_features, out_features, bias=False)
        self.bn   = nn.BatchNorm1d(out_features)
        self.act  = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x
    
class TwoBranchNet(nn.Module):
    """
    * IMU branch  : Residual + SE（深い）
    * TOF branch  : 軽量 Conv
    * マージ後    : Bi‑LSTM, Bi‑GRU, GaussianNoise→Dense
    * Attention   : additive softmax
    * FC head
    """
    def __init__(
        self,
        pad_len: int,
        imu_dim: int,
        tof_dim: int,
        n_classes: int,
        wd: float = 1e-4,
    ):
        super().__init__()
        self.pad_len  = pad_len
        self.imu_dim  = imu_dim
        self.tof_dim  = tof_dim

        # ── IMU branch (B, L, C) → permute → (B, C, L)
        self.imu_block1 = ResidualSEConvBlock(imu_dim,   64, 3, drop=0.1, wd=wd)
        self.imu_block2 = ResidualSEConvBlock(64,       128, 5, drop=0.1, wd=wd)

        # ── TOF branch
        self.tof_conv1 = nn.Sequential(
            nn.Conv1d(tof_dim, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )
        self.tof_conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )

        # ── 時系列マージ後の次元
        merge_channels = 128 + 128      # IMU + TOF
        self.bi_lstm   = nn.LSTM(
            input_size=merge_channels,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.bi_gru    = nn.GRU(
            input_size=merge_channels,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.noise     = GaussianNoise(0.09)
        self.noise_fc  = nn.Linear(merge_channels, 16)

        self.attn      = AttentionPooling(embed_dim=128*2 + 128*2 + 16)  # concat 前の D

        # ── FC head
        self.head = nn.Sequential(
            DenseBN(self.attn.score.in_features, 256, drop=0.5),
            DenseBN(256, 128, drop=0.3),
            nn.Linear(128, n_classes, bias=True),
        )

    # -----------------------------------------
    def forward(self, x):                       # x: (B, L, imu_dim+tof_dim)
        imu, tof = x[..., :self.imu_dim], x[..., self.imu_dim:]

        # channels‑last → channels‑first
        imu = imu.permute(0, 2, 1)              # (B, C, L)
        tof = tof.permute(0, 2, 1)

        # forward branches
        imu = self.imu_block1(imu)
        imu = self.imu_block2(imu)              # (B, 128, L/4)

        tof = self.tof_conv1(tof)
        tof = self.tof_conv2(tof)               # (B, 128, L/4)

        # concat on channel axis → (B, 256, L/4)
        merged = torch.cat([imu, tof], dim=1)

        # RNN expects (B, L, D)
        merged_t = merged.permute(0, 2, 1)

        xa, _ = self.bi_lstm(merged_t)          # (B, L', 256)
        xb, _ = self.bi_gru (merged_t)          # (B, L', 256)

        # noise branch
        xc = self.noise(merged_t)
        xc = F.elu(self.noise_fc(xc))           # (B, L', 16)

        # concat along feature dim
        merged_rnn = torch.cat([xa, xb, xc], dim=-1)  # (B, L', 528)

        # attention pooling → context vector
        ctx, _ = self.attn(merged_rnn)          # (B, 528)

        logits = self.head(ctx)                 # (B, n_classes)
        return logits
    
class ImuNet(nn.Module):
    def __init__(
        self,
        pad_len: int,
        imu_dim: int,
        n_classes: int,
        wd: float = 1e-4,
    ):
        super().__init__()
        self.pad_len  = pad_len
        self.imu_dim  = imu_dim

        # ── IMU branch (B, L, C) → permute → (B, C, L)
        self.imu_block1 = ResidualSEConvBlock(imu_dim,   64, 3, drop=0.1, wd=wd)
        self.imu_block2 = ResidualSEConvBlock(64,       128, 5, drop=0.1, wd=wd)

        # ── 時系列マージ後の次元
        merge_channels = 128     # IMU + TOF
        self.bi_lstm   = nn.LSTM(
            input_size=merge_channels,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.bi_gru    = nn.GRU(
            input_size=merge_channels,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.noise     = GaussianNoise(0.09)
        self.noise_fc  = nn.Linear(merge_channels, 16)

        self.attn      = AttentionPooling(embed_dim=128*2 + 16)  # concat 前の D

        # ── FC head
        self.head = nn.Sequential(
            DenseBN(self.attn.score.in_features, 128, drop=0.5),
            DenseBN(128, 56, drop=0.3),
            nn.Linear(56, n_classes, bias=True),
        )

    # -----------------------------------------
    def forward(self, x):                       # x: (B, L, imu_dim+tof_dim)
        imu, tof = x[..., :self.imu_dim], x[..., self.imu_dim:]

        # channels‑last → channels‑first
        imu = imu.permute(0, 2, 1)              # (B, C, L)
        tof = tof.permute(0, 2, 1)

        # forward branches
        imu = self.imu_block1(imu)
        imu = self.imu_block2(imu)              # (B, 128, L/4)

        # RNN expects (B, L, D)
        merged_t = imu.permute(0, 2, 1)

        xa, _ = self.bi_lstm(merged_t)          # (B, L', 256)
        xb, _ = self.bi_gru (merged_t)          # (B, L', 256)

        # noise branch
        xc = self.noise(merged_t)
        xc = F.elu(self.noise_fc(xc))           # (B, L', 16)

        # concat along feature dim
        merged_rnn = torch.cat([xa, xb, xc], dim=-1)  # (B, L', 528)

        # attention pooling → context vector
        ctx, _ = self.attn(merged_rnn)          # (B, 528)

        logits = self.head(ctx)                 # (B, n_classes)
        return logits    
    
class LitModel(L.LightningModule):
    """
    LightningModule wrapping ModelVariant_GRU
    -----------------------------------------
    * 入力        : x         … (B, L, C)    float32
    * 分類ラベル  : y_cls     … (B,)         int64  or one-hot
    * 回帰ターゲット: y_reg   … (B,)         float32  (オプション)
    """
    def __init__(
        self,
        imu_ch: int,
        tof_ch: int,
        num_classes: int,
        imu_only: bool = False,
        lr_init: float = 5e-4,
        weight_decay: float = 3e-3,
        cls_loss_weight: float = 1.0,
        class_weight: torch.Tensor | None = None,   # CE のクラス重み
    ):
        super().__init__()
        self.save_hyperparameters()

        # --- core model -----------------------------------------------------
        if imu_only:
            self.model = ImuNet(127, imu_ch, num_classes, weight_decay) 
        else:
            self.model = TwoBranchNet(127, imu_ch, tof_ch, num_classes, weight_decay)

        # 損失関数
        self.mse = nn.MSELoss()   # 回帰用

        # logging 用
        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.val_acc   = MulticlassAccuracy(num_classes=num_classes, average="macro")
        
        self.register_buffer(
            "class_weight",
            torch.tensor(class_weight, dtype=torch.float32)
        )

    # --------------------------------------------------------------------- #
    # forward はそのまま
    def forward(self, x):
        return self.model(x)   # -> (logits, regression)

    # --------------------------------------------------------------------- #
    def _shared_step(self, batch, stage: str):

        x, y_cls = batch

        logits= self(x)
        # --- 損失計算 ------------------------------------------------------
        ce = nn.CrossEntropyLoss(weight=self.class_weight)
        loss_cls = ce(logits, y_cls.argmax(dim=1) if y_cls.ndim == 2 else y_cls.to(torch.long))
        loss = self.hparams.cls_loss_weight * loss_cls

        # --- ログ ----------------------------------------------------------
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        if stage == "train":
            self.train_acc(preds, y_cls.argmax(dim=1) if y_cls.ndim == 2 else y_cls)
            self.log_dict(
                {
                    "train/loss": loss,
                    "train/acc": self.train_acc,
                },
                on_step=False, on_epoch=True, prog_bar=True,
            )
        elif stage == "val":
            self.val_acc(preds, y_cls.argmax(dim=1) if y_cls.ndim == 2 else y_cls)
            self.log_dict(
                {
                    "val/loss": loss,
                    "val/acc": self.val_acc,
                },
                on_step=False, on_epoch=True, prog_bar=True,
            )
        else:  # test
            self.log_dict(
                {"test/loss": loss},
                on_step=False, on_epoch=True,
            )
        return loss

    # --------------------------------------------------------------------- #
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    # --------------------------------------------------------------------- #
    def configure_optimizers(self):
        opt = Adam(self.parameters(),
                   lr=self.hparams.lr_init,
                   weight_decay=self.hparams.weight_decay)

        # Plateau で LR 半減
        scheduler = {
            "scheduler": ReduceLROnPlateau(opt,
                                           mode="min",
                                           patience=2,
                                           factor=0.5),
            "monitor": "val/loss",
        }
        return {"optimizer": opt, "lr_scheduler": scheduler}