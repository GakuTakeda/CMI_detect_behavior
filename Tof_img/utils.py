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
import lightning as L
from torch.optim import Adam
from torchmetrics.classification import MulticlassAccuracy

from scipy.signal import cont2discrete
from typing import Any, Optional, List, Tuple, Union
import math
import pathlib
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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

def convert_gesture(value):
    pull = ["Eyebrow - pull hair", "Eyelash - pull hair", "Forehead - pull hairline"]
    if value in pull:
        return "pull"
    else:
        return value

def labeling_for_macro(value):

    BFRB = [
        "Above ear - pull hair", "Forehead - pull hairline", "Forehead - scratch",
        "Eyebrow - pull hair", "Eyelash - pull hair", "Neck - pinch skin",
        "Neck - scratch", "Cheek - pinch skin"
    ]

    pull = ["Eyebrow - pull hair", "Eyelash - pull hair", "Forehead - pull hairline"]
    
    # BFRBリスト内の値であれば、そのインデックスを返す
    if value in BFRB:
        return BFRB.index(value)

    elif value == "classes":
        return [
            "Above ear - pull hair", "Forehead - pull hairline", "Forehead - scratch",
            "Eyebrow - pull hair", "Eyelash - pull hair", "Neck - pinch skin",
            "Neck - scratch", "Cheek - pinch skin"]

def seed_everything(seed: int = 42) -> None:
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
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False            # 入力形状による自動チューニングを無効化
    torch.backends.cuda.matmul.allow_tf32 = False     # TF32 を無効化
    torch.backends.cudnn.allow_tf32 = False

    # CUBLAS のワークスペースを固定（torch インポート前に設定するのが安全）
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"   # または ":4096:8"

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
            angular_dist[i] = 0 
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




def feature_eng(df: pd.DataFrame) -> pd.DataFrame:
    # df = remove_gravity_from_acc_in_train(df)
    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    df['rot_angle'] = 2 * np.arccos(df['rot_w'].clip(-1, 1))
    df['acc_mag_jerk'] = df.groupby('sequence_id')['acc_mag'].diff().fillna(0)
    df['rot_angle_vel'] = df.groupby('sequence_id')['rot_angle'].diff().fillna(0)
    rot_data = df[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
    angular_vel_group = calculate_angular_velocity_from_quat(rot_data)
    df['angular_vel_x'] = angular_vel_group[:, 0]
    df['angular_vel_y'] = angular_vel_group[:, 1]
    df['angular_vel_z'] = angular_vel_group[:, 2]
    df['angular_dist'] = calculate_angular_distance(rot_data)

    insert_cols = ['acc_mag', 'rot_angle', 'acc_mag_jerk', 'rot_angle_vel', 'angular_vel_x', 'angular_vel_y', 'angular_vel_z', 'angular_dist']
    cols = list(df.columns)

    for i, col in enumerate(cols):
        if col.startswith('thm_') or col.startswith('tof_'):
            insert_index = i
            break

    cols_wo_insert = [c for c in cols if c not in insert_cols]

    new_order = cols_wo_insert[:insert_index] + insert_cols + cols_wo_insert[insert_index:]
    df = df[new_order]

    return df


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
    
def down_len(lengths: torch.Tensor, pool_sizes: Sequence[int]) -> torch.Tensor:
    """Conv後に現れる各 MaxPool1d の pool_size を順に適用して長さを更新"""
    l = lengths.clone()
    for p in pool_sizes:
        l = torch.div(l, p, rounding_mode="floor")
    return l.clamp_min(1)

class AttentionLayer(nn.Module):
    """マスク対応のシンプルなアテンションプーリング"""
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.w = nn.Linear(in_dim, hidden, bias=False)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B,T,D], mask: [B,T] (True=有効). Noneなら全有効
        return: [B,D]
        """
        s = self.v(torch.tanh(self.w(x))).squeeze(-1)   # [B,T]
        if mask is not None:
            s = s.masked_fill(~mask, -1e9)
        w = s.softmax(dim=1)                            # [B,T]
        pooled = torch.bmm(w.unsqueeze(1), x).squeeze(1)  # [B,D]
        return pooled


class MultiScaleConv1d(nn.Module):
    """Multi-scale temporal convolution block (長さ保存: stride=1, SAME padding相当)"""
    def __init__(self, in_channels, out_per_kernel, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.convs = nn.ModuleList()
        for ks in kernel_sizes:
            pad = ks // 2  # 奇数カーネル推奨（長さ保存）
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_channels, out_per_kernel, ks, padding=pad, bias=False),
                nn.BatchNorm1d(out_per_kernel),
                nn.ReLU(inplace=True)
            ))

    @property
    def out_channels(self):
        # 合計出力チャネル
        return sum(block[0].out_channels for block in self.convs)

    def forward(self, x):
        outputs = [conv(x) for conv in self.convs]  # list of [N, out_per_kernel, L]
        return torch.cat(outputs, dim=1)            # [N, out_per_kernel*K, L]


class EnhancedSEBlock(nn.Module):
    """Avg/Maxの両方でSE"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction, bias=False),
            nn.SiLU(inplace=True),
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
    (Conv-BN-ReLU)×2 → SE → Add → ReLU → MaxPool → Dropout
    出力長: floor(L_in / pool_size)
    """
    def __init__(self, in_ch, out_ch, k=3, pool_size=2, drop=0.3):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)

        self.shortcut = nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.bn_sc = nn.BatchNorm1d(out_ch) if in_ch != out_ch else nn.Identity()

        self.se     = EnhancedSEBlock(out_ch)
        self.pool   = nn.MaxPool1d(pool_size)  # stride=pool_size
        self.drop   = nn.Dropout(drop)
        self.pool_size = pool_size

    def forward(self, x):                 # x: (N, C_in, L)
        residual = self.bn_sc(self.shortcut(x))
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = F.relu(out + residual)
        out = self.pool(out)
        out = self.drop(out)
        return out                        # (N, C_out, floor(L/p))


class MetaFeatureExtractor(nn.Module):
    """
    時系列マスク付きの簡易メタ特徴 (各Ch 5個: mean, std, min, max, abs-mean)
    入力: x[B,T,C], mask[B,T]
    出力: [B, 5*C]
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        if mask is None:
            valid = torch.ones(B, T, dtype=torch.bool, device=x.device)
        else:
            valid = mask

        # [B,T,1] に拡張
        m = valid.unsqueeze(-1)  # True=有効
        cnt = m.sum(dim=1).clamp_min(1)  # [B,1]

        xm = x * m  # 無効部は0

        mean = xm.sum(dim=1) / cnt
        var = ( (xm - mean.unsqueeze(1)) * m ).pow(2).sum(dim=1) / cnt
        std = torch.sqrt(var + self.eps)

        # min/max は無効部を +inf/-inf にする
        x_min = x.masked_fill(~m, float('inf')).min(dim=1).values
        x_max = x.masked_fill(~m, float('-inf')).max(dim=1).values
        abs_mean = xm.abs().sum(dim=1) / cnt

        feats = torch.cat([mean, std, x_min, x_max, abs_mean], dim=1)  # [B, 5*C]
        return feats

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
        X, img, y = zip(*batch)
        X = torch.stack(X, dim=0)      # (B, T, C)
        img = torch.stack(img, dim=0) 
        y = torch.stack(y, dim=0)      # (B, num_classes)

        lam  = np.random.beta(alpha, alpha)
        perm = torch.randperm(X.size(0))

        X_mix = lam * X + (1 - lam) * X[perm]
        img_mix = lam * img + (1 - lam) * img[perm]
        y_mix = lam * y + (1 - lam) * y[perm]
        return X_mix,img_mix, y_mix

    return _collate

def _pad(mat: np.ndarray, pad_len: int) -> np.ndarray:
    F = mat.shape[1]
    out = np.zeros((pad_len, F), dtype='float32')
    seq_len = min(len(mat), pad_len)
    out[:seq_len] = mat[:seq_len]
    return out


class GaussianNoise(nn.Module):
    def __init__(self, stddev=0.1):
        super().__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training and self.stddev > 0:
            noise = torch.randn_like(x) * self.stddev
            return x + noise
        return x
    
    
class TinyCNN(nn.Module):
    def __init__(self, in_channels=5, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels,16,3,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
    def forward(self,x):
        x = self.net(x)
        return x.view(x.size(0), -1)
    
class ModelVariant_LSTMGRU_TinyCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.imu_dim     = cfg.imu_dim
        self.num_classes = cfg.num_classes

        # ---- ToF branch ----
        self.tof_in_ch   = cfg.tof.in_channels
        self.tof_out_dim = cfg.tof.out_dim
        self.tof_cnn     = TinyCNN(in_channels=self.tof_in_ch, out_dim=self.tof_out_dim)

        # ---- IMU per-channel CNN ----
        ks   = cfg.cnn.multiscale.kernel_sizes
        opk  = cfg.cnn.multiscale.out_per_kernel
        ms_out = opk * len(ks)  # MultiScaleConv1d 出力チャネル
        res_out = cfg.cnn.residual.out_channels
        n_blocks = cfg.cnn.residual.num_blocks
        share_branch = cfg.cnn.share_branch

        def make_branch():
            blocks = [
                MultiScaleConv1d(1, opk, kernel_sizes=ks),     # [B, ms_out, T]
                ResidualSEBlock(ms_out, res_out),
            ]
            for _ in range(n_blocks - 1):
                blocks.append(ResidualSEBlock(res_out, res_out))
            return nn.Sequential(*blocks)

        if share_branch:
            shared = make_branch()
            self.imu_branches = nn.ModuleList([shared for _ in range(self.imu_dim)])
        else:
            self.imu_branches = nn.ModuleList([make_branch() for _ in range(self.imu_dim)])

        # ---- Meta ----
        self.meta = MetaFeatureExtractor()
        self.meta_proj = cfg.meta.proj_dim
        self.meta_dense = nn.Sequential(
            nn.Linear(5 * (self.imu_dim + self.tof_out_dim), self.meta_proj),
            nn.BatchNorm1d(self.meta_proj),
            nn.ReLU(),
            nn.Dropout(cfg.meta.dropout),
        )

        # ---- Sequence encoders (GRU/LSTM) ----
        enc_in = res_out * self.imu_dim + self.tof_out_dim
        self.rnn_hidden = cfg.rnn.hidden_size
        self.rnn_layers = cfg.rnn.num_layers
        self.rnn_bi     = cfg.rnn.bidirectional # bool
        self.rnn_drop   = cfg.rnn.dropout
        bi = 2 if self.rnn_bi else 1

        self.bigru  = nn.GRU(
            enc_in, self.rnn_hidden, batch_first=True,
            bidirectional=self.rnn_bi, num_layers=self.rnn_layers, dropout=self.rnn_drop
        )
        self.bilstm = nn.LSTM(
            enc_in, self.rnn_hidden, batch_first=True,
            bidirectional=self.rnn_bi, num_layers=self.rnn_layers, dropout=self.rnn_drop
        )

        # ---- Noise ----
        self.noise = GaussianNoise(cfg.noise_std)

        # ---- Attention + Head ----
        gru_dim  = self.rnn_hidden * bi
        lstm_dim = self.rnn_hidden * bi
        concat_dim = gru_dim + lstm_dim + enc_in
        self.attn = AttentionLayer(concat_dim)  # mask対応実装を想定（attn(x, mask=...)）

        head_hidden = cfg.head.hidden
        head_drop   = cfg.head.dropout
        head_in     = concat_dim + (self.meta_proj)

        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.BatchNorm1d(head_hidden),
            nn.ReLU(),
            nn.Dropout(head_drop),
            nn.Linear(head_hidden, self.num_classes),
        )

    # ---------- helpers ----------
    @staticmethod
    def _ensure_mask(B: int, T: int, lengths: Optional[torch.Tensor], mask: Optional[torch.Tensor], device) -> torch.Tensor:
        if mask is not None:
            return mask.to(device=device, dtype=torch.bool)
        if lengths is not None:
            ar = torch.arange(T, device=device)[None, :].expand(B, T)
            return (ar < lengths[:, None])
        return torch.ones(B, T, dtype=torch.bool, device=device)

    @staticmethod
    def _resample_mask(mask: torch.Tensor, target_len: int) -> torch.Tensor:
        if mask.size(1) == target_len:
            return mask
        x = mask.float().unsqueeze(1)                           # [B,1,T]
        x = F.adaptive_max_pool1d(x, output_size=target_len)    # [B,1,target_len]
        return (x.squeeze(1) > 0.5)

    def _meta_forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # MetaFeatureExtractor のシグネチャ差異に安全対応
        try:
            return self.meta(x, mask) if mask is not None else self.meta(x)
        except TypeError:
            return self.meta(x)

    # ---------- forward ----------
    def forward(
        self,
        x_imu: torch.Tensor,            # [B,T,C_imu]
        x_tof: torch.Tensor,            # [B,T,C_tof,H,W]
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C_imu = x_imu.shape
        device = x_imu.device

        # mask 準備
        mask_in = self._ensure_mask(B, T, lengths, mask, device)         # [B,T]

        # --- IMU per-channel CNN ---
        imu_feats = []
        for i in range(C_imu):
            xi = x_imu[:, :, i].unsqueeze(1)            # [B,1,T]
            fi = self.imu_branches[i](xi)               # [B,res_out,T]
            imu_feats.append(fi.transpose(1, 2))        # -> [B,T,res_out]
        imu_feat = torch.cat(imu_feats, dim=2)          # [B,T,res_out*C_imu]
        T_imu = imu_feat.size(1)
        mask_imu = self._resample_mask(mask_in, T_imu)  # [B,T_imu]

        # --- ToF CNN ---
        B, T_tof, C, H, W = x_tof.shape
        assert C == self.tof_in_ch, f"ToF channels mismatch: got {C}, expected {self.tof_in_ch}"
        tof_flat = x_tof.view(B*T_tof, C, H, W)         # [B*T,C,H,W]
        tof_vec  = self.tof_cnn(tof_flat)               # [B*T,tof_out_dim]
        tof_feats = tof_vec.view(B, T_tof, -1)          # [B,T_tof,tof_out_dim]
        # 時間長合わせ
        tof_feats = F.adaptive_avg_pool1d(tof_feats.transpose(1, 2), output_size=T_imu).transpose(1, 2)  # [B,T_imu,tof_out_dim]
        mask_tof  = self._resample_mask(mask_in, T_imu)                                                  # [B,T_imu]

        meta_imu = self._meta_forward(x_imu, mask_in)     # [B,5*C_imu]
        meta_tof = self._meta_forward(tof_feats, mask_tof)# [B,5*tof_out_dim]
        meta_vec = torch.cat([meta_imu, meta_tof], dim=1) # [B, 5*(C_imu+tof_out_dim)]
        meta_vec = self.meta_dense(meta_vec)               # [B, meta_proj]

        # --- Sequence fusion & RNN（pack）---
        seq = torch.cat([imu_feat, tof_feats], dim=2)         # [B,T_imu, enc_in]
        enc_in = seq.size(2)

        lengths_enc = mask_imu.logical_or(mask_tof).sum(dim=1).to(torch.long)
        lengths_enc = torch.clamp(lengths_enc, min=1)

        packed = pack_padded_sequence(seq, lengths=lengths_enc.cpu(), batch_first=True, enforce_sorted=False)
        gru_packed, _  = self.bigru(packed)
        lstm_packed, _ = self.bilstm(packed)
        gru_out, _  = pad_packed_sequence(gru_packed,  batch_first=True, total_length=T_imu)  # [B,T_imu,2H]
        lstm_out, _ = pad_packed_sequence(lstm_packed, batch_first=True, total_length=T_imu)  # [B,T_imu,2H]

        noise_out = self.noise(seq)   # [B,T_imu,enc_in]

        x = torch.cat([gru_out, lstm_out, noise_out], dim=2)         # [B,T_imu, 2H+2H+enc_in]
        mask_enc = mask_imu.logical_or(mask_tof)                     # [B,T_imu]

        # --- Attention (mask-aware) + Head ---
        x = self.attn(x, mask=mask_enc)                              # [B, 2H+2H+enc_in]
        x = torch.cat([x, meta_vec], dim=1)                      # [B, ...+meta_proj]
        out = self.head(x)                                           # [B,num_classes]
        return out

class litmodel(L.LightningModule):
    def __init__(
        self,
        cfg,
        lr_init: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: float = 1.0,
        min_lr: float = 1e-6,
        cls_loss_weight: float = 1.0,
        class_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg", "class_weight"])
        self.model = ModelVariant_LSTMGRU_TinyCNN(cfg)

        # metrics
        self.train_acc = MulticlassAccuracy(num_classes=cfg.num_classes, average="macro")
        self.val_acc   = MulticlassAccuracy(num_classes=cfg.num_classes, average="macro")

        # class weight（あるときだけbuffer登録）
        if class_weight is not None:
            cw = torch.as_tensor(class_weight, dtype=torch.float32)
            self.register_buffer("class_weight", cw)
        else:
            self.class_weight = None

        # optional regression（残置）
        self.mse = nn.MSELoss()

    # ------------------------------------------------------------
    def forward(
        self,
        x_imu: torch.Tensor,
        x_tof: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x_imu: [B,T,C_imu], x_tof: [B,T,C_toF,H,W]"""
        return self.model(x_imu, x_tof, lengths, mask)  # -> logits

    # ---- CE（hard/soft両対応） ----
    def _ce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # soft-label
        if target.ndim == 2 and target.dtype != torch.long:
            logp = F.log_softmax(logits, dim=1)
            return -(target * logp).sum(dim=1).mean()
        # hard-label
        hard = target if target.ndim == 1 else target.argmax(dim=1)
        ce = nn.CrossEntropyLoss(weight=self.class_weight)
        return ce(logits, hard)

    # ---- accuracy（形式に自動対応） ----
    def _acc(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        hard = target if target.ndim == 1 else target.argmax(dim=1)
        return self.train_acc(preds, hard) if self.training else self.val_acc(preds, hard)

    # ---------- batch parser ----------
    @staticmethod
    def _parse_batch(batch: Union[Tuple, list]):
        """
        サポートする形式（いずれも lengths, mask は可変長対応用）:
        - (x_imu, x_tof, lengths, mask, y)                      -> 標準
        - (x_imu, x_tof, lengths, mask, y, _, _)                -> 標準（ダミー2要素付き）
        - (x_imu, x_tof, lengths, mask, y_a, y_b, lam)          -> mixup（hard）
        - (x_imu, x_tof, lengths, mask)                         -> 推論等（y なし）
        """
        if not isinstance(batch, (tuple, list)):
            raise RuntimeError(f"Unexpected batch type: {type(batch)}")

        n = len(batch)
        if n == 7:
            x_imu, x_tof, lengths, mask, y_a, y_b, lam = batch
            return dict(x_imu=x_imu, x_tof=x_tof, lengths=lengths, mask=mask,
                        y=y_a, y_b=y_b, lam=lam, is_mixup=True)
        elif n == 6:
            x_imu, x_tof, lengths, mask, y, _ = batch  # 末尾のダミーを無視
            return dict(x_imu=x_imu, x_tof=x_tof, lengths=lengths, mask=mask,
                        y=y, y_b=None, lam=None, is_mixup=False)
        elif n == 5:
            x_imu, x_tof, lengths, mask, y = batch
            return dict(x_imu=x_imu, x_tof=x_tof, lengths=lengths, mask=mask,
                        y=y, y_b=None, lam=None, is_mixup=False)
        elif n == 4:
            x_imu, x_tof, lengths, mask = batch
            return dict(x_imu=x_imu, x_tof=x_tof, lengths=lengths, mask=mask,
                        y=None, y_b=None, lam=None, is_mixup=False)
        else:
            raise RuntimeError(f"Unexpected batch length: {n}")

    # ------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        """
        期待するバッチ：
          - mixup_pad_collate_fn(return_soft=False):
              (x_imu, x_tof, lengths, mask, y_a, y_b, lam)
          - mixup_pad_collate_fn(return_soft=True) などの soft:
              (x_imu, x_tof, lengths, mask, y_soft)
          - 通常:
              (x_imu, x_tof, lengths, mask, y)
        """
        b = self._parse_batch(batch)
        bs = b["x_imu"].size(0)

        if b["is_mixup"]:
            logits = self.forward(b["x_imu"], b["x_tof"], b["lengths"], b["mask"])
            loss = self.hparams.cls_loss_weight * (
                b["lam"] * self._ce(logits, b["y"]) + (1.0 - b["lam"]) * self._ce(logits, b["y_b"])
            )
            preds = logits.argmax(dim=1)
            acc   = b["lam"] * self._acc(preds, b["y"]) + (1.0 - b["lam"]) * self._acc(preds, b["y_b"])
        else:
            if b["y"] is None:
                raise RuntimeError("Training step requires labels in the batch.")
            logits = self.forward(b["x_imu"], b["x_tof"], b["lengths"], b["mask"])
            loss   = self.hparams.cls_loss_weight * self._ce(logits, b["y"])
            preds  = logits.argmax(dim=1)
            acc    = self._acc(preds, b["y"])

        # ログ（学習率も出す）
        log_dict = {"train/loss": loss, "train/acc": acc}
        opt = self.optimizers()
        if opt is not None:
            log_dict["lr"] = opt.param_groups[0]["lr"]
        self.log_dict(log_dict, on_step=True, on_epoch=True, batch_size=bs)
        return loss

    # ------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        b = self._parse_batch(batch)
        bs = b["x_imu"].size(0)
        # 検証では mixup を使わず、y を見る（ダミーがあっても無視）
        if b["y"] is None:
            raise RuntimeError("Validation step requires labels in the batch.")
        logits = self.forward(b["x_imu"], b["x_tof"], b["lengths"], b["mask"])
        loss   = self.hparams.cls_loss_weight * self._ce(logits, b["y"])
        preds  = logits.argmax(dim=1)
        acc    = self._acc(preds, b["y"])
        self.log_dict({"val/loss": loss, "val/acc": acc}, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)

    def test_step(self, batch, batch_idx):
        b = self._parse_batch(batch)
        bs = b["x_imu"].size(0)
        if b["y"] is None:
            raise RuntimeError("Test step requires labels in the batch.")
        logits = self.forward(b["x_imu"], b["x_tof"], b["lengths"], b["mask"])
        loss   = self.hparams.cls_loss_weight * self._ce(logits, b["y"])
        self.log_dict({"test/loss": loss}, on_step=False, on_epoch=True, batch_size=bs)

    # ------------------------------------------------------------
    # Optimizer: AdamW + Warmup(epochs指定) → Cosine（min_lr下限）を step 更新
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr_init,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # 総 step 数（optimizer.step の回数）
        total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        if total_steps is None:
            steps_per_epoch = self.trainer.num_training_batches
            total_steps = steps_per_epoch * self.trainer.max_epochs

        # ★ warmup を「エポック数」から「ステップ数」へ換算
        max_epochs    = max(1, int(self.trainer.max_epochs))
        warmup_epochs = float(self.hparams.warmup_epochs)
        warmup_steps  = max(1, int(total_steps * (warmup_epochs / max_epochs)))
        cosine_steps  = max(1, total_steps - warmup_steps)
        min_lr        = float(self.hparams.min_lr)

        base_lrs = [g["lr"] for g in opt.param_groups]

        def make_lambda(base_lr: float):
            # base_lr に対する下限倍率
            min_factor = 1.0 if min_lr >= base_lr else (min_lr / max(base_lr, 1e-12))

            def lr_lambda(step: int):
                # ① linear warmup（min_factor→1.0）
                if step < warmup_steps:
                    warm = step / float(max(1, warmup_steps))
                    return min_factor + (1.0 - min_factor) * warm
                # ② cosine decay（1.0→min_factor）
                progress = (step - warmup_steps) / float(max(1, cosine_steps))
                cos_term = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_factor + (1.0 - min_factor) * cos_term

            return lr_lambda

        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=[make_lambda(lr) for lr in base_lrs]
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",   # ← ステップ毎に更新
                "frequency": 1,
                "name": "warmup_cosine_minlr",
            },
        }

# ------------------------------
# 可変長: IMU + ToF データセット
# ------------------------------
class SequenceDatasetVarLenToF(Dataset):
    def __init__(self, X_imu_list, X_tof_list, y_array, imu_augmenter=None, tof_augmenter=None):
        """
        X_imu_list: list[Tensor or ndarray], 各要素 [Ti, C_imu]
        X_tof_list: list[Tensor or ndarray], 各要素 [Ti, C_tof(=5), H, W]
        y_array   : np.ndarray[int] or Tensor[int]  （one-hot不要）
        imu_augmenter: callable(x:[Ti,C_imu])->x
        tof_augmenter: callable(x:[Ti,5,H,W])->x
        """
        assert len(X_imu_list) == len(X_tof_list) == len(y_array)
        self.X_imu = [torch.as_tensor(x, dtype=torch.float32) for x in X_imu_list]
        self.X_tof = [torch.as_tensor(x, dtype=torch.float32) for x in X_tof_list]
        self.y     = torch.as_tensor(y_array, dtype=torch.long)
        self.aug_imu = imu_augmenter
        self.aug_tof = tof_augmenter

        # 長さ整合性チェック（必要ならここで truncate/pad を入れてもよい）
        for xi, xt in zip(self.X_imu, self.X_tof):
            if xi.shape[0] != xt.shape[0]:
                raise ValueError(f"IMU と ToF の時間長が一致していません: {xi.shape[0]} vs {xt.shape[0]}")

    def __len__(self):
        return len(self.X_imu)

    def __getitem__(self, i):
        x_imu = self.X_imu[i]
        x_tof = self.X_tof[i]
        if self.aug_imu is not None:
            x_imu = self.aug_imu(x_imu)           # [Ti,C_imu]
        if self.aug_tof is not None:
            x_tof = self.aug_tof(x_tof)           # [Ti,5,H,W]
        return x_imu, x_tof, self.y[i]


# ------------------------------
# 前処理: 1シーケンス → (IMU, ToF) テンソル
# ------------------------------
def preprocess_sequence_tof(
    df_seq: pd.DataFrame,
    feature_cols: list[str],
    imu_cols: list[str],
    tof_cols: list[str],
    scaler: StandardScaler,
    tof_shape: tuple[int,int,int] = (5, 8, 8),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    - 欠損を ffill/bfill → 0
    - StandardScaler.transform
    - IMU と ToF を分離し torch.Tensor へ
    戻り:
      x_imu: [T, C_imu]
      x_tof: [T, 5, H, W]
    """
    mat = (
        df_seq[feature_cols]
        .ffill().bfill().fillna(0)
        .values
    )
    mat = scaler.transform(mat).astype("float32")

    feat_idx = {c: i for i, c in enumerate(feature_cols)}
    imu_idx  = [feat_idx[c] for c in imu_cols]
    tof_idx  = [feat_idx[c] for c in tof_cols]

    x_imu = torch.from_numpy(mat[:, imu_idx])  # [T, C_imu]

    T = mat.shape[0]
    Ct, H, W = tof_shape
    x_tof = torch.from_numpy(mat[:, tof_idx]).view(T, Ct, H, W)  # [T, 5, H, W]
    return x_imu, x_tof


# ------------------------------
# collate: Pad + mask（IMU/ToF 両方）
# ------------------------------
def collate_pad_tof(batch):
    """
    入力（サンプル単位）: (x_imu:[Ti,C], x_tof:[Ti,5,H,W], y)
    出力（バッチ）:
      x_imu_pad:[B,L,C], x_tof_pad:[B,L,5,H,W], lengths:[B], mask:[B,L], y:[B] or [B,C]
    """
    xs_imu, xs_tof, ys = zip(*batch)
    xs_imu = [torch.as_tensor(x, dtype=torch.float32) for x in xs_imu]
    xs_tof = [torch.as_tensor(x, dtype=torch.float32) for x in xs_tof]

    lengths = torch.tensor([x.shape[0] for x in xs_imu], dtype=torch.long)
    B = len(xs_imu)
    L = int(lengths.max().item())

    C_imu = xs_imu[0].shape[1]
    C_tof, H, W = xs_tof[0].shape[1:]

    # IMU pad
    x_imu_pad = torch.zeros((B, L, C_imu), dtype=torch.float32)
    # ToF pad
    x_tof_pad = torch.zeros((B, L, C_tof, H, W), dtype=torch.float32)

    for i, (xi, xt) in enumerate(zip(xs_imu, xs_tof)):
        Ti = xi.shape[0]
        x_imu_pad[i, :Ti] = xi
        x_tof_pad[i, :Ti] = xt

    mask = torch.arange(L)[None, :].repeat(B, 1) < lengths[:, None]   # [B,L], True=有効

    # y は [B] or [B,C] に整形
    ys = torch.stack([torch.as_tensor(y) for y in ys])
    return x_imu_pad, x_tof_pad, lengths, mask, ys


# ------------------------------
# MixUp collate（IMU/ToF 両方ミックス）
# ------------------------------
def mixup_pad_collate_fn_tof(alpha: float = 0.2):
    if alpha <= 0:
        return collate_pad_tof

    def _collate(batch):
        x_imu, x_tof, lengths, mask, y = collate_pad_tof(batch)
        B = x_imu.size(0)
        perm = torch.randperm(B)

        lam = float(np.random.beta(alpha, alpha))
        lam = max(lam, 1.0 - lam)  # 弱混合の抑制

        x_imu_mix = lam * x_imu + (1.0 - lam) * x_imu[perm]
        x_tof_mix = lam * x_tof + (1.0 - lam) * x_tof[perm]

        # マスクは OR、長さは True の数（※長さに依存する後段は mask を使う前提）
        mask_mix    = mask | mask[perm]
        lengths_mix = mask_mix.sum(dim=1)

        y_a, y_b = y, y[perm]
        return x_imu_mix, x_tof_mix, lengths_mix, mask_mix, y_a, y_b, lam

    return _collate 

def _select_cols(all_cols: list[str]) -> tuple[list[str], list[str], list[str]]:
    meta = {'gesture','gesture_int','sequence_type','behavior','orientation',
            'row_id','subject','phase','sequence_id','sequence_counter'}
    feat_cols = [c for c in all_cols if c not in meta]
    imu_cols  = [c for c in feat_cols if not (c.startswith("thm_") or c.startswith("tof_"))]
    tof_cols  = [c for c in feat_cols if c.startswith("tof_")]
    return feat_cols, imu_cols, tof_cols

def _infer_tof_shape(num_tof_cols: int, ct: int = 5, default_hw: Tuple[int,int] = (8,8)) -> Tuple[int,int,int]:
    # num_tof_cols = Ct * H * W を想定
    if num_tof_cols % ct == 0:
        hw = num_tof_cols // ct
        r = int(round(hw ** 0.5))
        if r * r == hw:
            return (ct, r, r)
    # フォールバック
    h, w = default_hw
    assert ct * h * w == num_tof_cols, f"tof_cols({num_tof_cols})が Ct*H*W と一致しません（Ct={ct}, H={h}, W={w}）"
    return (ct, h, w)

class GestureDataModule(L.LightningDataModule):
    """
    可変長 IMU + ToF（5×H×W）DataModule
    - prepare_data: feature_eng → labeling → scaler fit/save → 列保存
    - setup       : scaler適用 → シーケンス毎に IMU/ToF Tensor化（可変長）→ StratifiedGroupKFold → Dataset
    - collate     : 可変長 Pad + mask（MixUp対応）
    """
    def __init__(self, cfg, fold_idx: int):
        super().__init__()
        self.cfg = cfg
        self.fold_idx = fold_idx
        self.n_splits = cfg.data.n_splits

        self.raw_dir = Path(cfg.data.raw_dir)
        self.export_dir = HydraConfig.get().runtime.output_dir / pathlib.Path(cfg.data.export_dir)

        self.batch = cfg.train.batch_size
        self.batch_val = cfg.train.batch_size_val
        self.mixup_a = cfg.train.mixup_alpha

    # --------------------------
    def prepare_data(self):
        df = pd.read_csv(self.raw_dir / "train.csv")
        df = feature_eng(df)
        df["gesture_int"] = df["gesture"].apply(labeling)

        self.export_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.export_dir / "gesture_classes.npy", labeling("classes"))

        feat_cols, imu_cols, tof_cols = _select_cols(df.columns)
        np.save(self.export_dir / "feature_cols.npy", np.array(feat_cols, dtype=object))
        np.save(self.export_dir / "imu_cols.npy",    np.array(imu_cols,   dtype=object))
        np.save(self.export_dir / "tof_cols.npy",    np.array(tof_cols,   dtype=object))

        scaler = StandardScaler().fit(df[feat_cols].ffill().bfill().fillna(0).values)
        joblib.dump(scaler, self.export_dir / "scaler.pkl")

    # --------------------------
    def setup(self, stage: Optional[str] = None):
        df = pd.read_csv(self.raw_dir / "train.csv")
        df = feature_eng(df)
        df["gesture_int"] = df["gesture"].apply(labeling)
        classes = labeling("classes")
        self.num_classes = len(classes)

        # 列とスケーラ
        feat_cols = np.load(self.export_dir / "feature_cols.npy", allow_pickle=True).tolist()
        imu_cols  = np.load(self.export_dir / "imu_cols.npy",    allow_pickle=True).tolist()
        tof_cols  = np.load(self.export_dir / "tof_cols.npy",    allow_pickle=True).tolist()
        scaler: StandardScaler = joblib.load(self.export_dir / "scaler.pkl")

        # ToF 形状
        ct = int(getattr(self.cfg.data, "tof_in_channels", 5))
        if getattr(self.cfg.data, "tof_hw", None) is not None:
            h, w = tuple(self.cfg.data.tof_hw)
            tof_shape = (ct, h, w)
            assert ct * h * w == len(tof_cols), "tof_hw と tof_cols 数が一致しません"
        else:
            tof_shape = _infer_tof_shape(len(tof_cols), ct=ct, default_hw=(8, 8))

        # スケール適用
        df_feat = df[feat_cols].ffill().bfill().fillna(0)
        df[feat_cols] = scaler.transform(df_feat.values)

        # ---- 可変長テンソルを構成 ----
        X_imu_list: List[torch.Tensor] = []
        X_tof_list: List[torch.Tensor] = []
        y_list: List[int] = []
        subjects: List[str] = []

        subj_col = "subject" if "subject" in df.columns else None

        for _, seq in df.groupby("sequence_id"):
            x_imu, x_tof = preprocess_sequence_tof(
                df_seq=seq,
                feature_cols=feat_cols,
                imu_cols=imu_cols,
                tof_cols=tof_cols,
                scaler=scaler,
                tof_shape=tof_shape,
            )
            X_imu_list.append(x_imu)  # [Ti,C_imu]
            X_tof_list.append(x_tof)  # [Ti,Ct,H,W]
            y_list.append(int(seq["gesture_int"].iloc[0]))
            subjects.append(seq[subj_col].iloc[0] if subj_col else int(seq["sequence_id"].iloc[0]))

        y_int = np.array(y_list, dtype=np.int64)

        # ---- StratifiedGroupKFold ----
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.cfg.data.random_seed
        )
        groups = np.array(subjects)
        tr_idx, val_idx = list(skf.split(np.arange(len(X_imu_list)), y_int))[self.fold_idx]

        # ---- Dataset ----
        X_imu_tr = [X_imu_list[i] for i in tr_idx]
        X_tof_tr = [X_tof_list[i] for i in tr_idx]
        y_tr     = y_int[tr_idx]

        X_imu_va = [X_imu_list[i] for i in val_idx]
        X_tof_va = [X_tof_list[i] for i in val_idx]
        y_va     = y_int[val_idx]

        self._train_ds = SequenceDatasetVarLenToF(X_imu_tr, X_tof_tr, y_tr, imu_augmenter=None, tof_augmenter=None)
        self._val_ds   = SequenceDatasetVarLenToF(X_imu_va, X_tof_va, y_va, imu_augmenter=None, tof_augmenter=None)

        # ---- class_weight & steps/epoch ----
        self.class_weight = torch.tensor(
            compute_class_weight(class_weight="balanced", classes=np.arange(self.num_classes), y=y_tr),
            dtype=torch.float32
        )
        self.steps_per_epoch = math.ceil(len(tr_idx) / self.batch)

        # 便利に保存
        np.save(self.export_dir / "tof_shape.npy", np.array(tof_shape))

    # --------------------------
    def train_dataloader(self):
        if self.mixup_a and self.mixup_a > 0:
            collate_fn = mixup_pad_collate_fn_tof(self.mixup_a)
        else:
            collate_fn = collate_pad_tof
        return DataLoader(
            self._train_ds,
            batch_size=self.batch, shuffle=True, drop_last=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_val, shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=collate_pad_tof,
        )