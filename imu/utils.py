from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data.dataloader import default_collate
from scipy.spatial.transform import Rotation as R
import random
import math
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
import joblib
import json
from collections import Counter
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union, Callable
import lightning as L
from torchmetrics.classification import MulticlassAccuracy

# =========================================================
# 可視化ユーティリティ（そのまま使用可）
# =========================================================
def plot_val_gesture_distribution(val_dict, topn=None, save_path=None, title="Validation gesture distribution"):
    gestures = [d["gesture"] for d in val_dict.values()]
    counts = Counter(gestures)
    df = pd.DataFrame(counts.items(), columns=["gesture", "count"]).sort_values("count", ascending=False)
    df["pct"] = df["count"] / df["count"].sum() * 100

    plot_df = df if topn is None else df.head(topn)
    plt.figure(figsize=(8, 0.45 * len(plot_df) + 1))
    plt.barh(plot_df["gesture"], plot_df["count"])
    plt.gca().invert_yaxis()
    for i, (c, p) in enumerate(zip(plot_df["count"], plot_df["pct"])):
        plt.text(c, i, f" {int(c)} ({p:.1f}%)", va="center")
    plt.xlabel("count")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    return df

# =========================================================
# スコア関数（そのまま）
# =========================================================
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
        f1_binary = f1_score(y_true_bin, y_pred_bin, pos_label=True, zero_division=0, average='binary')
        return 0.5 * f1_binary

    def macro_score(self, sol, sub):
        y_true_mc = [x if x in self.target_gestures else 'non_target' for x in sol]
        y_pred_mc = [x if x in self.target_gestures else 'non_target' for x in sub]
        f1_macro = f1_score(y_true_mc, y_pred_mc, average='macro', zero_division=0)
        return 0.5 * f1_macro

# ラベル関連（そのまま）
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

def labeling(value):
    if value == "classes":
        return BFRB + non_BFRB
    if value in BFRB:
        return BFRB.index(value)
    if value in non_BFRB:
        return len(BFRB) + non_BFRB.index(value)
    raise ValueError(f"Unknown label: {value}")

# =========================================================
# 再現性ユーティリティ（そのまま）
# =========================================================
def seed_everything(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

# =========================================================
# IMU向け特徴量（ToF/THM 依存を削除）
# =========================================================

def calculate_angular_velocity_from_quat(rot_data, time_delta=1/200):
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data
    num_samples = quat_values.shape[0]
    angular_vel = np.zeros((num_samples, 3), dtype=np.float32)
    for i in range(num_samples - 1):
        q_t = quat_values[i]; q_t1 = quat_values[i+1]
        if (np.all(np.isnan(q_t)) or np.all(np.isclose(q_t, 0)) or
            np.all(np.isnan(q_t1)) or np.all(np.isclose(q_t1, 0))):
            continue
        try:
            delta_rot = (R.from_quat(q_t).inv() * R.from_quat(q_t1))
            angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except ValueError:
            pass
    return angular_vel


def calculate_angular_distance(rot_data):
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data
    num_samples = quat_values.shape[0]
    angular_dist = np.zeros(num_samples, dtype=np.float32)
    for i in range(num_samples - 1):
        q1 = quat_values[i]; q2 = quat_values[i+1]
        if (np.all(np.isnan(q1)) or np.all(np.isclose(q1, 0)) or
            np.all(np.isnan(q2)) or np.all(np.isclose(q2, 0))):
            angular_dist[i] = 0; continue
        try:
            relative_rotation = (R.from_quat(q1).inv() * R.from_quat(q2))
            angular_dist[i] = np.linalg.norm(relative_rotation.as_rotvec())
        except ValueError:
            angular_dist[i] = 0
    return angular_dist


def feature_eng(df: pd.DataFrame) -> pd.DataFrame:
    """
    IMU のみを対象に、軽い特徴量を追加（列の並び替えはしない）。
    ToF/THM 列が存在しても無視する。
    """
    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    df['rot_angle'] = 2 * np.arccos(df['rot_w'].clip(-1, 1))
    df['acc_mag_jerk'] = df.groupby('sequence_id')['acc_mag'].diff().fillna(0)
    df['rot_angle_vel'] = df.groupby('sequence_id')['rot_angle'].diff().fillna(0)
    rot_data = df[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
    ang_v = calculate_angular_velocity_from_quat(rot_data)
    df['angular_vel_x'] = ang_v[:, 0]
    df['angular_vel_y'] = ang_v[:, 1]
    df['angular_vel_z'] = ang_v[:, 2]
    df['angular_dist'] = calculate_angular_distance(rot_data)
    return df

# =========================================================
# Dataset（IMU のみ）
# =========================================================
class FixedLenIMUDataset(Dataset):
    def __init__(self, X_imu_list: List[np.ndarray], y: np.ndarray):
        self.Xi = X_imu_list
        self.y  = y
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        xi = torch.from_numpy(self.Xi[idx])            # [L, C_imu]
        yy = torch.tensor(self.y[idx], dtype=torch.long)
        return xi, yy


def _randu(a=0.0, b=1.0): return np.random.uniform(a, b)

class AugmentIMUOnly:
    """
    入出力:
      imu: [L, C_imu]  (float32)
    返り値も同形状（maskなし・固定長のまま）
    """
    def __init__(self,
        p_time_shift=0.7, max_shift_ratio=0.1,
        p_time_warp=0.5,  warp_min=0.9, warp_max=1.1,
        p_block_dropout=0.5, n_blocks=(1,3), block_len=(2,6),
        p_imu_jitter=0.9, imu_sigma=0.03,
        p_imu_scale=0.5,  imu_scale_sigma=0.03,
        p_imu_drift=0.5,  drift_std=0.003, drift_clip=0.3,
        p_imu_small_rot=0.0, rot_deg=5.0,
        pad_value=0.0,
    ):
        self.p_time_shift = p_time_shift
        self.max_shift_ratio = max_shift_ratio
        self.p_time_warp = p_time_warp
        self.warp_min, self.warp_max = warp_min, warp_max
        self.p_block_dropout = p_block_dropout
        self.n_blocks, self.block_len = n_blocks, block_len
        self.p_imu_jitter = p_imu_jitter
        self.imu_sigma = imu_sigma
        self.p_imu_scale = p_imu_scale
        self.imu_scale_sigma = imu_scale_sigma
        self.p_imu_drift = p_imu_drift
        self.drift_std = drift_std
        self.drift_clip = drift_clip
        self.p_imu_small_rot = p_imu_small_rot
        self.rot_deg = rot_deg
        self.pad_value = pad_value

    # ---------- time ops ----------
    def _time_shift(self, x, shift):
        L = x.shape[0]
        if shift == 0: return x
        out = np.full_like(x, self.pad_value)
        if shift > 0:
            out[shift:] = x[:L-shift]
        else:
            out[:L+shift] = x[-shift:]
        return out

    def _time_warp(self, x: np.ndarray, scale: float) -> np.ndarray:
        L = x.shape[0]
        Lp = max(2, int(round(L * scale)))
        t = torch.from_numpy(x.astype(np.float32)).transpose(0,1).unsqueeze(0)  # [1,C,L]
        y = F.interpolate(t, size=Lp, mode="linear", align_corners=False)
        y = F.interpolate(y, size=L,  mode="linear", align_corners=False)
        return y.squeeze(0).transpose(0,1).contiguous().numpy()

    def _block_dropout(self, x):
        L = x.shape[0]
        nb = np.random.randint(self.n_blocks[0], self.n_blocks[1]+1)
        for _ in range(nb):
            bl = np.random.randint(self.block_len[0], self.block_len[1]+1)
            s = np.random.randint(0, max(1, L - bl + 1))
            x[s:s+bl] = self.pad_value
        return x

    # ---------- imu ops ----------
    def _imu_jitter(self, imu):        # add noise
        return imu + np.random.randn(*imu.shape).astype(np.float32) * self.imu_sigma

    def _imu_scale(self, imu):         # per-channel scale
        scale = (1.0 + np.random.randn(imu.shape[1]).astype(np.float32) * self.imu_scale_sigma)
        return imu * scale[None, :]

    def _imu_drift(self, imu):
        L, C = imu.shape
        drift = np.cumsum(np.random.randn(L, C).astype(np.float32) * self.drift_std, axis=0)
        np.clip(drift, -self.drift_clip, self.drift_clip, out=drift)
        return imu + drift

    def _imu_small_rot(self, imu):
        th = np.deg2rad(self.rot_deg) * np.random.uniform(-1,1)
        Rz = np.array([[ np.cos(th), -np.sin(th), 0],
                       [ np.sin(th),  np.cos(th), 0],
                       [ 0,           0,          1]], dtype=np.float32)
        def rot3(block):
            return (block @ Rz.T).astype(np.float32)
        imu_out = imu.copy()
        if imu.shape[1] >= 3:
            imu_out[:, :3] = rot3(imu[:, :3])
        if imu.shape[1] >= 6:
            imu_out[:, 3:6] = rot3(imu[:, 3:6])
        return imu_out

    # ---------- main ----------
    def __call__(self, imu: np.ndarray):
        L = imu.shape[0]
        if np.random.rand() < self.p_time_shift:
            shift = int(np.round(_randu(-self.max_shift_ratio, self.max_shift_ratio) * L))
            imu = self._time_shift(imu, shift)
        if np.random.rand() < self.p_time_warp:
            s = _randu(self.warp_min, self.warp_max)
            imu = self._time_warp(imu, s)
        if np.random.rand() < self.p_block_dropout:
            imu = self._block_dropout(imu)
        if np.random.rand() < self.p_imu_jitter: imu = self._imu_jitter(imu)
        if np.random.rand() < self.p_imu_scale:  imu = self._imu_scale(imu)
        if np.random.rand() < self.p_imu_drift:  imu = self._imu_drift(imu)
        if self.p_imu_small_rot > 0 and np.random.rand() < self.p_imu_small_rot:
            imu = self._imu_small_rot(imu)
        return imu.astype(np.float32)


class FixedLenIMUDatasetAug(FixedLenIMUDataset):
    def __init__(self, X_imu_list, y, augmenter: AugmentIMUOnly):
        super().__init__(X_imu_list, y)
        self.aug = augmenter
    def __getitem__(self, idx):
        imu = self.Xi[idx].copy()
        imu = self.aug(imu)
        xi = torch.from_numpy(imu)
        yy = torch.tensor(self.y[idx], dtype=torch.long)
        return xi, yy


# =========================================================
# Collate（IMU のみ）
# =========================================================
def make_collate_pad_imu(return_len_mask: bool = False, pad_value: float = 0.0) -> Callable:
    def _collate(batch):
        xs_imu, ys = zip(*batch)
        xs_imu = [torch.as_tensor(x, dtype=torch.float32) for x in xs_imu]
        lengths = torch.tensor([x.shape[0] for x in xs_imu], dtype=torch.long)
        B = len(xs_imu)
        L = int(lengths.max().item())
        C_imu = xs_imu[0].shape[1]
        x_imu_pad = torch.full((B, L, C_imu), pad_value, dtype=torch.float32)
        for i, xi in enumerate(xs_imu):
            Ti = xi.shape[0]
            x_imu_pad[i, :Ti] = xi
        y = torch.stack([torch.as_tensor(v) for v in ys])
        if return_len_mask:
            mask = (torch.arange(L).unsqueeze(0) < lengths.unsqueeze(1)).to(torch.float32)
            return x_imu_pad, y, lengths, mask
        else:
            return x_imu_pad, y
    return _collate


def mixup_pad_collate_fn_imu(alpha: float = 0.4, p: float = 0.5, pad_value: float = 0.0):
    base = make_collate_pad_imu(return_len_mask=False, pad_value=pad_value)
    if alpha <= 0:
        return base
    def _collate(batch):
        x_imu, y = base(batch)      # x_imu:[B,L,C], y:[B]
        if y.ndim != 1:
            y = y.argmax(dim=1)
        B = x_imu.size(0)
        device = x_imu.device
        perm = torch.randperm(B, device=device)
        apply = (torch.rand(B, device=device) < p)
        lam = torch.ones(B, device=device, dtype=torch.float32)
        lam_vals = torch.distributions.Beta(alpha, alpha).sample((int(apply.sum()),)).to(device)
        lam[apply] = torch.maximum(lam_vals, 1.0 - lam_vals)
        x_imu = lam.view(B,1,1) * x_imu + (1.0 - lam.view(B,1,1)) * x_imu[perm]
        y_a, y_b = y, y[perm]
        return x_imu, y_a, y_b, lam
    return _collate


# =========================================================
# モデル本体（IMU のみ）
# =========================================================
class MultiScaleConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3,5,7]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, k, padding=k//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            ) for k in kernel_sizes
        ])
    def forward(self, x):
        return torch.cat([conv(x) for conv in self.convs], dim=1)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b,c,_ = x.size()
        y = self.pool(x).view(b,c)
        y = self.fc(y).view(b,c,1)
        return x * y

class ResidualSEBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, pool_size=2, drop=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=k//2)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=k//2)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch!=out_ch else nn.Identity()
        self.bn_sc = nn.BatchNorm1d(out_ch) if in_ch!=out_ch else nn.Identity()
        self.se = SEBlock(out_ch)
        self.pool = nn.MaxPool1d(pool_size) if pool_size and pool_size>1 else nn.Identity()
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        res = self.bn_sc(self.shortcut(x))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.se(x)
        x = F.relu(x + res)
        x = self.pool(x)
        return self.drop(x)

class AttentionLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model,1)
    def forward(self, x):
        score = torch.tanh(self.fc(x)).squeeze(-1)
        weights = F.softmax(score, dim=1).unsqueeze(-1)
        return (x*weights).sum(dim=1)

class MetaFeatureExtractor(nn.Module):
    def forward(self,x):
        mean = x.mean(dim=1)
        std = x.std(dim=1)
        maxv,_ = x.max(dim=1)
        minv,_ = x.min(dim=1)
        slope = (x[:,-1,:] - x[:,0,:]) / max(x.size(1)-1,1)
        return torch.cat([mean,std,maxv,minv,slope],dim=1)

class GaussianNoise(nn.Module):
    def __init__(self, stddev=0.09):
        super().__init__()
        self.stddev = stddev
    def forward(self,x):
        if self.training:
            return x + torch.randn_like(x) * self.stddev
        return x

class ModelVariant_LSTMGRU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        assert cfg.imu_dim is not None and cfg.num_classes is not None, "cfg.imu_dim と cfg.num_classes を設定してください。"
        self.imu_dim      = cfg.imu_dim
        self.num_classes  = cfg.num_classes

        ksz               = cfg.cnn.multiscale.kernel_sizes
        out_per_kernel    = cfg.cnn.multiscale.out_per_kernel
        self.ms_in_ch     = 1
        self.ms_total_out = len(ksz) * out_per_kernel
        self.imu_c_mid    = cfg.cnn.residual.out_channels
        self.res_blocks   = cfg.cnn.residual.num_blocks
        self.share_branch = cfg.cnn.share_branch

        self.rnn_hidden   = cfg.rnn.hidden_size
        self.rnn_layers   = cfg.rnn.num_layers
        self.rnn_bidir    = cfg.rnn.bidirectional
        self.rnn_dropout  = cfg.rnn.dropout
        self.noise_std    = cfg.noise_std

        self.meta_stats_per_ch = getattr(cfg.meta, "stats_per_channel", 5)
        self.meta_hidden       = cfg.meta.proj_dim
        self.meta_dropout      = cfg.meta.dropout
        self.head_hidden       = cfg.head.hidden
        self.head_dropout      = cfg.head.dropout

        def make_imu_branch():
            layers = [MultiScaleConv1d(self.ms_in_ch, out_per_kernel)]
            layers.append(ResidualSEBlock(self.ms_total_out, self.imu_c_mid))
            for _ in range(self.res_blocks - 1):
                layers.append(ResidualSEBlock(self.imu_c_mid, self.imu_c_mid))
            return nn.Sequential(*layers)

        if self.share_branch:
            self.imu_branch_shared = make_imu_branch()
            self.imu_branches = None
        else:
            self.imu_branches = nn.ModuleList([make_imu_branch() for _ in range(self.imu_dim)])
            self.imu_branch_shared = None

        # Meta
        self.meta = MetaFeatureExtractor()
        meta_in = self.meta_stats_per_ch * (self.imu_dim)
        self.meta_dense = nn.Sequential(
            nn.Linear(meta_in, self.meta_hidden),
            nn.BatchNorm1d(self.meta_hidden),
            nn.ReLU(),
            nn.Dropout(self.meta_dropout),
        )

        # Sequence encoders（IMU特徴のみ）
        fused_feat_dim = self.imu_c_mid * self.imu_dim
        self.bigru  = nn.GRU(
            fused_feat_dim, self.rnn_hidden,
            batch_first=True, bidirectional=self.rnn_bidir,
            num_layers=self.rnn_layers, dropout=self.rnn_dropout
        )
        self.bilstm = nn.LSTM(
            fused_feat_dim, self.rnn_hidden,
            batch_first=True, bidirectional=self.rnn_bidir,
            num_layers=self.rnn_layers, dropout=self.rnn_dropout
        )
        self.noise  = GaussianNoise(self.noise_std)

        concat_dim = (self.rnn_hidden * (2 if self.rnn_bidir else 1)) \
                   + (self.rnn_hidden * (2 if self.rnn_bidir else 1)) \
                   + fused_feat_dim
        self.attn = AttentionLayer(concat_dim)

        head_in = concat_dim + self.meta_hidden
        self.head = nn.Sequential(
            nn.Linear(head_in, self.head_hidden),
            nn.BatchNorm1d(self.head_hidden),
            nn.ReLU(),
            nn.Dropout(self.head_dropout),
            nn.Linear(self.head_hidden, self.num_classes),
        )

    def forward(self, x_imu):
        """x_imu: (B, T, imu_dim)"""
        B, T, _ = x_imu.shape

        # IMU branch（各軸ごとに共有 or 個別ブランチ）
        imu_feats = []
        for i in range(self.imu_dim):
            xi = x_imu[:, :, i].unsqueeze(1)            # (B,1,T)
            if self.share_branch:
                fi = self.imu_branch_shared(xi)         # (B,C,T)
            else:
                fi = self.imu_branches[i](xi)           # (B,C,T)
            imu_feats.append(fi.transpose(1, 2))         # (B,T,C)
        imu_feat = torch.cat(imu_feats, dim=2)           # (B,T, 48*imu_dim)

        # Meta（時系列から統計量）
        meta_imu = self.meta(x_imu)
        meta = self.meta_dense(meta_imu)

        # Sequence encoders
        seq  = imu_feat                                   # (B,T,fused_feat_dim)
        gru, _  = self.bigru(seq)
        lstm, _ = self.bilstm(seq)
        noise   = self.noise(seq)

        x = torch.cat([gru, lstm, noise], dim=2)         # (B,T,concat_dim)
        x = self.attn(x)                                 # (B,concat_dim)
        x = torch.cat([x, meta], dim=1)                  # (B,concat_dim+meta_hidden)
        return self.head(x)                              # (B,num_classes)


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, final_lr=2e-5, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [self.base_lr * (self.last_epoch + 1) / self.warmup_epochs for _ in self.optimizer.param_groups]
        else:
            decay_epoch = self.last_epoch - self.warmup_epochs
            decay_total = self.total_epochs - self.warmup_epochs
            cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_epoch / decay_total))
            return [self.final_lr + (self.base_lr - self.final_lr) * cosine_decay for _ in self.optimizer.param_groups]


# =========================================================
# LightningModule（IMU のみ）
# =========================================================
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
        self.model = ModelVariant_LSTMGRU(cfg)
        self.train_acc = MulticlassAccuracy(num_classes=cfg.num_classes, average="macro")
        self.val_acc   = MulticlassAccuracy(num_classes=cfg.num_classes, average="macro")
        if class_weight is not None:
            cw = torch.as_tensor(class_weight, dtype=torch.float32)
            self.register_buffer("class_weight", cw)
        else:
            self.class_weight = None
        self.cfg = cfg

    def forward(self, x_imu: torch.Tensor) -> torch.Tensor:
        return self.model(x_imu)

    # CE（hard/soft両対応）
    def _ce(self, logits: torch.Tensor, target: torch.Tensor, use_weight: bool = True) -> torch.Tensor:
        if target.ndim == 2 and target.dtype != torch.long:  # soft-label
            logp = F.log_softmax(logits, dim=1)
            return -(target * logp).sum(dim=1).mean()
        hard = target if target.ndim == 1 else target.argmax(dim=1)
        if use_weight and self.class_weight is not None:
            ce = nn.CrossEntropyLoss(weight=self.class_weight)
        else:
            ce = nn.CrossEntropyLoss()
        return ce(logits, hard)

    def _ce_per_sample(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 2 and target.dtype != torch.long:
            logp = F.log_softmax(logits, dim=1)
            return -(target * logp).sum(dim=1)
        hard = target if target.ndim == 1 else target.argmax(dim=1)
        if self.class_weight is not None:
            ce = nn.CrossEntropyLoss(weight=self.class_weight, reduction='none')
        else:
            ce = nn.CrossEntropyLoss(reduction='none')
        return ce(logits, hard)

    def _acc(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        hard = target if target.ndim == 1 else target.argmax(dim=1)
        return self.train_acc(preds, hard) if self.training else self.val_acc(preds, hard)

    def training_step(self, batch, batch_idx):
        # mixup: (x_imu, y_a, y_b, lam) / 普通: (x_imu, y)
        if len(batch) == 4:
            x_imu, y_a, y_b, lam = batch
            logits = self.forward(x_imu)
            la = self._ce_per_sample(logits, y_a)
            lb = self._ce_per_sample(logits, y_b)
            loss = self.hparams.cls_loss_weight * (lam * la + (1.0 - lam) * lb).mean()
            preds = logits.argmax(dim=1)
            acc   = self._acc(preds, y_a)
        else:
            x_imu, y = batch
            logits = self.forward(x_imu)
            loss   = self.hparams.cls_loss_weight * self._ce(logits, y)
            preds  = logits.argmax(dim=1)
            acc    = self._acc(preds, y)
        # lr ログ（optimizer が未設定のタイミング対策）
        try:
            lr_val = self.optimizers().param_groups[0]["lr"]
        except Exception:
            lr_val = float(self.hparams.lr_init)
        self.log_dict({"train/loss": loss, "train/acc": acc, "lr": lr_val},
                      on_step=True, on_epoch=True, batch_size=x_imu.size(0))
        return loss

    def to_binary(self, y): return [0 if i<9 else 1 for i in y]
    def to_9class(self, y): return [i%9 for i in y]

    def on_validation_epoch_start(self):
        self._val_preds = []
        self._val_trues = []

    def validation_step(self, batch, batch_idx):
        x_imu, y = batch
        if y.ndim == 2: y = y.argmax(dim=1)
        logits = self.forward(x_imu)
        loss = self.hparams.cls_loss_weight * self._ce(logits, y)
        preds = logits.argmax(dim=1)
        self._val_preds.append(preds.cpu())
        self._val_trues.append(y.cpu())
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, batch_size=x_imu.size(0))
        return loss

    def on_validation_epoch_end(self):
        preds = torch.cat(self._val_preds)
        trues = torch.cat(self._val_trues)
        acc18 = accuracy_score(trues, preds)
        f118  = f1_score(trues, preds, average="macro")
        tb_p, tb_t = torch.tensor(self.to_binary(preds)), torch.tensor(self.to_binary(trues))
        acc2 = accuracy_score(tb_t, tb_p); f12 = f1_score(tb_t, tb_p, average="macro")
        t9_p, t9_t = torch.tensor(self.to_9class(preds)), torch.tensor(self.to_9class(trues))
        acc9 = accuracy_score(t9_t, t9_p); f19 = f1_score(t9_t, t9_p, average="macro")
        f1_avg = (f12 + f19) / 2.0
        self.log_dict({
            "val/acc18": acc18, "val/f1_18": f118,
            "val/acc_bin": acc2, "val/f1_bin": f12,
            "val/acc_9": acc9,   "val/f1_9": f19,
            "val/f1_avg": f1_avg,
            "val/neg_f1_18": 1.0 - f118,
        }, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x_imu, y = batch
        logits = self.forward(x_imu)
        loss   = self.hparams.cls_loss_weight * self._ce(logits, y, use_weight=False)
        self.log_dict({"test/loss": loss}, on_step=False, on_epoch=True, batch_size=x_imu.size(0))

    def configure_optimizers(self):
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad: continue
            if p.ndim == 1 or n.endswith(".bias"): no_decay.append(p)
            else: decay.append(p)
        wd = float(self.hparams.weight_decay)
        lr = float(self.hparams.lr_init)
        adamw_params = [{"params": decay, "weight_decay": wd},{"params": no_decay, "weight_decay": 0.0}]
        adam_params  = [{"params": decay, "weight_decay": wd},{"params": no_decay, "weight_decay": 0.0}]

        if self.cfg.scheduler.name in ("warmup_cosine", "cosine", "cos"):
            opt = torch.optim.AdamW(adamw_params, lr=lr, betas=(0.9, 0.999), eps=1e-8)
            sched = WarmupCosineScheduler(
                optimizer=opt,
                warmup_epochs=self.hparams.warmup_epochs,
                total_epochs=self.trainer.max_epochs,
                base_lr=self.hparams.lr_init,
                final_lr=self.hparams.min_lr,
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch", "frequency": 1, "name": "warmup_cosine"}}
        elif self.cfg.scheduler.name in ("plateau", "reduce_on_plateau", "rop", "reduceplateau"):
            opt = torch.optim.Adam(adam_params, lr=lr, betas=(0.9, 0.999), eps=1e-8)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=self.cfg.scheduler.factor, patience=self.cfg.scheduler.patience,
                threshold=self.cfg.scheduler.threshold, threshold_mode=self.cfg.scheduler.threshold_mode,
                cooldown=self.cfg.scheduler.cooldown, min_lr=self.cfg.scheduler.min_lr_sched, eps=1e-8,
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val/neg_f1_18", "interval": "epoch", "frequency": 1, "reduce_on_plateau": True, "name": "plateau"}}
        else:
            raise ValueError(f"Unknown scheduler name: {self.cfg.scheduler.name}")

# =========================================================
# DataModule（IMU のみ）
# =========================================================

def _select_imu_cols(all_cols: list[str]) -> list[str]:
    meta = {'gesture','gesture_int','sequence_type','behavior','orientation',
            'row_id','subject','phase','sequence_id','sequence_counter'}
    imu_cols  = [c for c in all_cols if (c not in meta) and (not c.startswith('tof_')) and (not c.startswith('thm_'))]
    return imu_cols


def crop_or_pad_np(xi: np.ndarray, L: int, mode: str, pad_value: float):
    T = xi.shape[0]
    if T >= L:
        if mode == "random": start = np.random.randint(0, T - L + 1)
        elif mode == "head": start = 0
        elif mode == "tail": start = T - L
        else: start = max((T - L) // 2, 0)
        return xi[start:start+L]
    C = xi.shape[1]
    out_i = np.full((L, C), pad_value, dtype=xi.dtype)
    out_i[:T] = xi
    return out_i


class GestureDataModule(L.LightningDataModule):
    def __init__(self, cfg, fold_idx: int):
        super().__init__()
        self.cfg = cfg
        self.fold_idx = fold_idx
        self.n_splits = cfg.data.n_splits
        self.raw_dir = Path(cfg.data.raw_dir)
        self.export_dir = HydraConfig.get().runtime.output_dir / Path(cfg.data.export_dir)
        self.batch = cfg.train.batch_size
        self.batch_val = cfg.train.batch_size_val
        self.num_workers = cfg.train.num_workers
        self.mixup_a = cfg.train.mixup_alpha
        self.max_seq_len: Optional[int] = self.cfg.data.max_seq_len
        self.pad_percentile: int = self.cfg.data.pad_percentile
        self.pad_value: float = self.cfg.data.pad_value
        self.truncate_mode_train: str = self.cfg.data.truncate_mode_train
        self.truncate_mode_val:   str = self.cfg.data.truncate_mode_val
        self.num_classes: int = 0
        self.imu_cols:  List[str] = []
        self.class_weight: torch.Tensor | None = None
        self.imu_ch: int = 0
        self.dim: int = 0  # 互換用（cfg.imu_dim に設定するため）

    def prepare_data(self):
        df = pd.read_csv(self.raw_dir / "train.csv")
        df = feature_eng(df)
        df["gesture_int"] = df["gesture"].apply(labeling)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.export_dir / "gesture_classes.npy", labeling("classes"))
        imu_cols = _select_imu_cols(df.columns.tolist())
        np.save(self.export_dir / "imu_cols.npy",    np.array(imu_cols,   dtype=object))
        scaler = StandardScaler().fit(df[imu_cols].ffill().bfill().fillna(0).values)
        joblib.dump(scaler, self.export_dir / "scaler.pkl")

    def setup(self, stage: Optional[str] = None):
        df = pd.read_csv(self.raw_dir / "train.csv")
        df = feature_eng(df)
        df["gesture_int"] = df["gesture"].apply(labeling)
        classes = labeling("classes")
        self.num_classes = len(classes)
        self.imu_cols  = np.load(self.export_dir / "imu_cols.npy",    allow_pickle=True).tolist()
        scaler: StandardScaler = joblib.load(self.export_dir / "scaler.pkl")
        self.imu_ch = len(self.imu_cols)
        self.dim = self.imu_ch

        # スケール（IMU のみ）
        df_feat = df[self.imu_cols].copy()
        df_feat = df_feat.replace(-1, 0).fillna(0)
        df_feat = df_feat.mask(df_feat == 0, 1e-3)
        df[self.imu_cols] = scaler.transform(df_feat.values)

        # ---- シーケンス毎に numpy へ ----
        X_imu_list: List[np.ndarray] = []
        y_list: List[int] = []
        subjects: List[str] = []
        seq_ids: List[str] = []
        lengths: List[int] = []

        subj_col = "subject" if "subject" in df.columns else None

        for sid, seq in df.groupby("sequence_id"):
            imu = seq[self.imu_cols].to_numpy(dtype=np.float32, copy=False)  # [T, C_imu]
            T = imu.shape[0]
            lengths.append(T)
            X_imu_list.append(imu)
            y_list.append(int(seq["gesture_int"].iloc[0]))
            subjects.append(seq[subj_col].iloc[0] if subj_col else int(sid))
            seq_ids.append(sid)

        y_int = np.asarray(y_list, dtype=np.int64)

        # ---- 固定長 L ----
        if self.max_seq_len is not None and int(self.max_seq_len) > 0:
            L = int(self.max_seq_len)
        else:
            L = int(np.percentile(lengths, self.pad_percentile))
        self.max_seq_len = L
        np.save(self.export_dir / "sequence_maxlen.npy", L)

        # ---- split ----
        idx_all = np.arange(len(X_imu_list))
        kf = GroupKFold(n_splits=self.n_splits, shuffle=True,
                        random_state=self.cfg.data.random_seed)
        tr_idx, val_idx = list(kf.split(idx_all, y_int, groups=subjects))[self.fold_idx]
        classes_arr = np.array(classes).tolist()

        def pack(indices):
            return {
                str(seq_ids[i]): {
                    "subject": subjects[i],
                    "gesture": classes_arr[int(y_int[i])]
                } for i in indices
            }
        split_map = {"fold": int(self.fold_idx), "train": pack(tr_idx), "val": pack(val_idx)}
        with open(self.export_dir / f"seq_split_fold{self.fold_idx}.json", "w", encoding="utf-8") as f:
            json.dump(split_map, f, ensure_ascii=False, indent=2)
        plot_val_gesture_distribution(split_map["val"], save_path=self.export_dir / f"val_gesture_dist_fold_{self.fold_idx}.png")

        Xtr_imu, ytr = [], []
        for i in tr_idx:
            xi = crop_or_pad_np(X_imu_list[i], L, self.truncate_mode_train, self.pad_value)
            Xtr_imu.append(xi); ytr.append(y_int[i])
        Xva_imu, yva = [], []
        for i in val_idx:
            xi = crop_or_pad_np(X_imu_list[i], L, self.truncate_mode_val, self.pad_value)
            Xva_imu.append(xi); yva.append(y_int[i])

        ytr = np.asarray(ytr, dtype=np.int64)
        yva = np.asarray(yva, dtype=np.int64)

        if self.cfg.aug.no_aug:
            self._train_ds = FixedLenIMUDataset(Xtr_imu, ytr)
        else:
            self._train_ds = FixedLenIMUDatasetAug(
                Xtr_imu, ytr,
                AugmentIMUOnly(
                    p_time_shift=self.cfg.aug.p_time_shift, max_shift_ratio=self.cfg.aug.max_shift_ratio,
                    p_time_warp=self.cfg.aug.p_time_warp, warp_min=self.cfg.aug.warp_min, warp_max=self.cfg.aug.warp_max,
                    p_block_dropout=self.cfg.aug.p_block_dropout, n_blocks=self.cfg.aug.n_blocks, block_len=self.cfg.aug.block_len,
                    p_imu_jitter=self.cfg.aug.p_imu_jitter, imu_sigma=self.cfg.aug.imu_sigma,
                    p_imu_scale=self.cfg.aug.p_imu_scale, imu_scale_sigma=self.cfg.aug.imu_scale_sigma,
                    p_imu_drift=self.cfg.aug.p_imu_drift, drift_std=self.cfg.aug.drift_std, drift_clip=self.cfg.aug.drift_clip,
                    p_imu_small_rot=self.cfg.aug.p_imu_small_rot,
                    pad_value=self.cfg.aug.pad_value,
                )
            )
        self._val_ds   = FixedLenIMUDataset(Xva_imu, yva)

        self.class_weight = torch.tensor(
            compute_class_weight(class_weight="balanced", classes=np.arange(self.num_classes), y=ytr),
            dtype=torch.float32
        )
        self.steps_per_epoch = math.ceil(len(tr_idx) / self.batch)

    def train_dataloader(self):
        return DataLoader(
            self._train_ds,
            batch_size=self.batch,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            collate_fn=mixup_pad_collate_fn_imu(alpha=self.mixup_a, p=self.cfg.train.mixup_prob, pad_value=self.pad_value)
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_val,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            collate_fn=make_collate_pad_imu(return_len_mask=False, pad_value=self.pad_value)
        )
