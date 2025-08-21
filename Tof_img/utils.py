from __future__ import annotations
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
from collections import Counter
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
import json

def plot_val_gesture_distribution(val_dict, topn=None, save_path=None, title="Validation gesture distribution"):
    # 1) 件数カウント
    gestures = [d["gesture"] for d in val_dict.values()]
    counts = Counter(gestures)

    # 2) 件数・割合の表
    df = pd.DataFrame(counts.items(), columns=["gesture", "count"]).sort_values("count", ascending=False)
    df["pct"] = df["count"] / df["count"].sum() * 100

    # 3) 可視化（横棒。ラベルが長くても見やすい）
    plot_df = df if topn is None else df.head(topn)
    plt.figure(figsize=(8, 0.45 * len(plot_df) + 1))
    plt.barh(plot_df["gesture"], plot_df["count"])
    plt.gca().invert_yaxis()  # 上を最大に
    for i, (c, p) in enumerate(zip(plot_df["count"], plot_df["pct"])):
        plt.text(c, i, f" {int(c)} ({p:.1f}%)", va="center")
    plt.xlabel("count")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    return df  # 集計表（全クラス）を返す


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

# ========= その他のサブモジュール（そのまま） =========
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
        x = self.pool(x)  # pool が Identity なら時間長は維持
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


class ModelVariant_LSTMGRU_TinyCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # ===== 必須パラメータ =====
        assert cfg.imu_dim is not None and cfg.num_classes is not None, \
            "cfg.imu_dim と cfg.num_classes を設定してください。"
        self.imu_dim      = cfg.imu_dim
        self.num_classes  = cfg.num_classes

        # ===== TOF =====
        self.tof_in_ch   = cfg.tof.in_channels
        self.tof_out_dim = cfg.tof.out_dim

        # ===== CNN(IMU) =====
        ksz               = cfg.cnn.multiscale.kernel_sizes        # 例 [3,5,7]
        out_per_kernel    = cfg.cnn.multiscale.out_per_kernel      # 例 12
        self.ms_in_ch     = 1                                      # IMUは各軸を1ch入力
        self.ms_total_out = len(ksz) * out_per_kernel              # 例 3*12=36

        self.imu_c_mid    = cfg.cnn.residual.out_channels          # 例 48
        self.res_blocks   = cfg.cnn.residual.num_blocks            # 例 2
        self.share_branch = cfg.cnn.share_branch                   # True で全IMU軸で重み共有

        # ===== RNN / Noise =====
        self.rnn_hidden   = cfg.rnn.hidden_size
        self.rnn_layers   = cfg.rnn.num_layers
        self.rnn_bidir    = cfg.rnn.bidirectional
        self.rnn_dropout  = cfg.rnn.dropout
        self.noise_std    = cfg.noise_std

        # ===== Meta =====
        self.meta_stats_per_ch = getattr(cfg.meta, "stats_per_channel", 5)  # 既定5
        self.meta_hidden       = cfg.meta.proj_dim
        self.meta_dropout      = cfg.meta.dropout

        # ===== Head =====
        self.head_hidden  = cfg.head.hidden
        self.head_dropout = cfg.head.dropout

        # ===== IMU branch (作成関数) =====
        def make_imu_branch():
            layers = [MultiScaleConv1d(self.ms_in_ch, out_per_kernel)]  # 例: 1→(3*12)=36
            # 最初の Residual は 36→48、その後は 48→48 を (num_blocks-1) 回
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

        # ===== TOF branch =====
        self.tof_cnn = TinyCNN(in_channels=self.tof_in_ch, out_dim=self.tof_out_dim)

        # ===== Meta feature =====
        self.meta = MetaFeatureExtractor()
        meta_in = self.meta_stats_per_ch * (self.imu_dim + self.tof_out_dim)
        self.meta_dense = nn.Sequential(
            nn.Linear(meta_in, self.meta_hidden),
            nn.BatchNorm1d(self.meta_hidden),
            nn.ReLU(),
            nn.Dropout(self.meta_dropout),
        )

        # ===== Sequence encoders =====
        fused_feat_dim = self.imu_c_mid * self.imu_dim + self.tof_out_dim
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

        # ===== Attention + Head =====
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

    def forward(self, x_imu, x_tof):
        """
        x_imu: (B, T, imu_dim)
        x_tof: (B, T, C, H, W)
        """
        B, T, _ = x_imu.shape

        # ===== IMU branch =====
        imu_feats = []
        for i in range(self.imu_dim):
            xi = x_imu[:, :, i].unsqueeze(1)            # (B,1,T)
            if self.share_branch:
                fi = self.imu_branch_shared(xi)         # (B,C,T)
            else:
                fi = self.imu_branches[i](xi)           # (B,C,T)
            imu_feats.append(fi.transpose(1, 2))         # (B,T,C)
        imu_feat = torch.cat(imu_feats, dim=2)           # (B,T, 48*imu_dim)

        # ===== TOF branch =====
        B, T, C, H, W = x_tof.shape
        tof_flat  = x_tof.view(B*T, C, H, W)
        tof_feats = self.tof_cnn(tof_flat).view(B, T, -1)  # (B,T,tof_out_dim)

        # 時間長合わせ
        tof_feats = F.adaptive_avg_pool1d(tof_feats.transpose(1, 2), output_size=imu_feat.size(1)).transpose(1, 2)

        # ===== Meta =====
        meta_imu = self.meta(x_imu)       # (B, stats_per_ch * imu_dim)
        meta_tof = self.meta(tof_feats)   # (B, stats_per_ch * tof_out_dim)
        meta = torch.cat([meta_imu, meta_tof], dim=1)
        meta = self.meta_dense(meta)      # (B, meta_hidden)

        # ===== Sequence fusion & encoders =====
        seq  = torch.cat([imu_feat, tof_feats], dim=2)   # (B,T,fused_feat_dim)
        gru, _  = self.bigru(seq)                        # (B,T,hid*(1/2))
        lstm, _ = self.bilstm(seq)                       # (B,T,hid*(1/2))
        noise   = self.noise(seq)                        # (B,T,fused_feat_dim)

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
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [self.base_lr * (self.last_epoch + 1) / self.warmup_epochs for _ in self.optimizer.param_groups]
        else:
            decay_epoch = self.last_epoch - self.warmup_epochs
            decay_total = self.total_epochs - self.warmup_epochs
            cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_epoch / decay_total))
            return [self.final_lr + (self.base_lr - self.final_lr) * cosine_decay for _ in self.optimizer.param_groups]


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

        self.mse = nn.MSELoss()
        self.cfg = cfg

    # ------------------------------------------------------------
    def forward(self, x_imu: torch.Tensor, x_tof: torch.Tensor) -> torch.Tensor:
        """x_imu: [B,T,C_imu], x_tof: [B,T,C_toF,H,W]"""
        return self.model(x_imu, x_tof)  # -> logits

    # ---- CE（hard/soft両対応） ----
    def _ce(self, logits: torch.Tensor, target: torch.Tensor, use_weight: bool = True) -> torch.Tensor:
        # soft-label
        if target.ndim == 2 and target.dtype != torch.long:
            logp = F.log_softmax(logits, dim=1)
            return -(target * logp).sum(dim=1).mean()
        # hard-label
        hard = target if target.ndim == 1 else target.argmax(dim=1)
        if use_weight and self.class_weight is not None:
            ce = nn.CrossEntropyLoss(weight=self.class_weight)
        else:
            ce = nn.CrossEntropyLoss()
        return ce(logits, hard)

    # ---- accuracy ----
    def _acc(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        hard = target if target.ndim == 1 else target.argmax(dim=1)
        return self.train_acc(preds, hard) if self.training else self.val_acc(preds, hard)

    # ---------- batch parser ----------
    @staticmethod
    def _parse_batch(batch: Union[Tuple, list]):
        """
        サポートする形式:
        - (x_imu, x_tof, y)                 -> 通常（collate_pad_tof）
        - (x_imu, x_tof, y_a, y_b, lam)     -> mixup（mixup_pad_collate_fn_tof）
        """
        if not isinstance(batch, (tuple, list)):
            raise RuntimeError(f"Unexpected batch type: {type(batch)}")

        n = len(batch)
        if n == 5:
            x_imu, x_tof, y_a, y_b, lam = batch
            return dict(x_imu=x_imu, x_tof=x_tof, y=y_a, y_b=y_b, lam=lam, is_mixup=True)
        elif n == 3:
            x_imu, x_tof, y = batch
            return dict(x_imu=x_imu, x_tof=x_tof, y=y, y_b=None, lam=None, is_mixup=False)
        else:
            raise RuntimeError(f"Unexpected batch length: {n}")

    # ------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        b = self._parse_batch(batch)
        bs = b["x_imu"].size(0)

        if b["is_mixup"]:
            logits = self.forward(b["x_imu"], b["x_tof"])
            loss = self.hparams.cls_loss_weight * (
                b["lam"] * self._ce(logits, b["y"]) + (1.0 - b["lam"]) * self._ce(logits, b["y_b"])
            )
            preds = logits.argmax(dim=1)
            acc   = b["lam"] * self._acc(preds, b["y"]) + (1.0 - b["lam"]) * self._acc(preds, b["y_b"])
        else:
            logits = self.forward(b["x_imu"], b["x_tof"])
            loss   = self.hparams.cls_loss_weight * self._ce(logits, b["y"])
            preds  = logits.argmax(dim=1)
            acc    = self._acc(preds, b["y"])

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
        logits = self.forward(b["x_imu"], b["x_tof"])
        loss   = self.hparams.cls_loss_weight * self._ce(logits, b["y"])
        preds  = logits.argmax(dim=1)
        acc    = self._acc(preds, b["y"])
        # Plateau を使うなら monitor: "val/loss" を推奨
        self.log_dict({"val/loss": loss, "val/acc": acc}, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)

    def test_step(self, batch, batch_idx):
        b = self._parse_batch(batch)
        bs = b["x_imu"].size(0)
        logits = self.forward(b["x_imu"], b["x_tof"])
        loss   = self.hparams.cls_loss_weight * self._ce(logits, b["y"], use_weight=False)
        self.log_dict({"test/loss": loss}, on_step=False, on_epoch=True, batch_size=bs)

    # === 以下、Cosine=AdamW / Plateau=Adam の切替（そのまま） ===
    def configure_optimizers(self):
        sch_cfg  = getattr(self.hparams, "scheduler", None)

        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or n.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)

        wd = float(self.hparams.weight_decay)
        lr = float(self.hparams.lr_init)

        adamw_params = [
            {"params": decay,    "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        adam_params = [
            {"params": decay,    "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        if self.cfg.scheduler.name in ("warmup_cosine", "cosine", "cos"):
            opt = torch.optim.AdamW(adamw_params, lr=lr, betas=(0.9, 0.999), eps=1e-8)

            sched = WarmupCosineScheduler(
                optimizer=opt,
                warmup_epochs=self.hparams.warmup_epochs,
                total_epochs=self.trainer.max_epochs,
                base_lr=self.hparams.lr_init,
                final_lr=self.hparams.min_lr,
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": sched,
                    "interval": "epoch",
                    "frequency": 1,
                    "name": "warmup_cosine",
                },
            }

        elif self.cfg.scheduler.name in ("plateau", "reduce_on_plateau", "rop", "reduceplateau"):
            opt = torch.optim.Adam(adam_params, lr=lr, betas=(0.9, 0.999), eps=1e-8)

            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode=self.cfg.scheduler.mode, factor=self.cfg.scheduler.factor, patience=self.cfg.scheduler.patience,
                threshold=self.cfg.scheduler.threshold, threshold_mode=self.cfg.scheduler.threshold_mode,
                cooldown=self.cfg.scheduler.cooldown, min_lr=self.cfg.scheduler.min_lr_sched, eps=1e-8,
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": sched,
                    "monitor": self.cfg.scheduler.monitor,
                    "interval": "epoch",
                    "frequency": 1,
                    "reduce_on_plateau": True,
                    "name": "plateau",
                },
            }

        else:
            raise ValueError(f"Unknown scheduler name: {self.cfg.scheduler.name}")



class FixedLenToFDataset(Dataset):
    def __init__(self, X_imu_list: List[np.ndarray], X_tof_list: List[np.ndarray], y: np.ndarray):
        self.Xi = X_imu_list
        self.Xt = X_tof_list
        self.y  = y
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        # テンソル化（float32 / long）
        xi = torch.from_numpy(self.Xi[idx])            # [L, C_imu]
        xt = torch.from_numpy(self.Xt[idx])            # [L, Ct, H, W]
        yy = torch.tensor(self.y[idx], dtype=torch.long)
        return xi, xt, yy
    
def _randu(a=0.0, b=1.0): return np.random.uniform(a, b)
def _randn(s=1.0): return np.random.randn() * s

class AugmentIMUToF:
    """
    入出力:
      imu: [L, C_imu]  (float32)
      tof: [L, Ct, H, W] (float32)
    返り値も同形状（maskなし・固定長のまま）
    """
    def __init__(self,
        p_time_shift=0.7, max_shift_ratio=0.1,
        p_time_warp=0.5,  warp_min=0.9, warp_max=1.1,
        p_block_dropout=0.5, n_blocks=(1,3), block_len=(2,6),

        p_imu_jitter=0.9, imu_sigma=0.03,
        p_imu_scale=0.5,  imu_scale_sigma=0.03,
        p_imu_drift=0.5,  drift_std=0.003, drift_clip=0.3,
        p_imu_small_rot=0.0, rot_deg=5.0,  # 必要なら有効化

        p_tof_ch_drop=0.2,
        p_tof_erasing=0.3, erase_hw=( (2,4), (2,4) ), n_erase=(1,2),
        p_tof_shift=0.3,
        p_tof_gain_noise=0.7, tof_gain=(0.95,1.05), tof_sigma=0.01,
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

        self.p_tof_ch_drop = p_tof_ch_drop
        self.p_tof_erasing = p_tof_erasing
        self.erase_hw = erase_hw
        self.n_erase = n_erase
        self.p_tof_shift = p_tof_shift
        self.p_tof_gain_noise = p_tof_gain_noise
        self.tof_gain = tof_gain
        self.tof_sigma = tof_sigma

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
        """
        x.ndim == 2: [L, C]
        x.ndim == 4: [L, Ct, H, W]
        時間長 L を scale 倍に伸縮→元の L に戻す（値は線形補間）
        """
        L = x.shape[0]
        Lp = max(2, int(round(L * scale)))

        if x.ndim == 2:  # [L, C]
            t = torch.from_numpy(x.astype(np.float32))         # [L, C]
            t = t.transpose(0, 1).unsqueeze(0)                 # [1, C, L]
            y = F.interpolate(t, size=Lp, mode="linear", align_corners=False)
            y = F.interpolate(y, size=L,  mode="linear", align_corners=False)
            y = y.squeeze(0).transpose(0, 1).contiguous().numpy()  # [L, C]
            return y

        if x.ndim == 4:  # [L, Ct, H, W]
            L, Ct, H, W = x.shape
            xf = x.reshape(L, Ct * H * W)                      # [L, Cflat]
            t  = torch.from_numpy(xf.astype(np.float32)).transpose(0, 1).unsqueeze(0)  # [1, Cflat, L]
            y  = F.interpolate(t, size=Lp, mode="linear", align_corners=False)
            y  = F.interpolate(y, size=L,  mode="linear", align_corners=False)
            y  = y.squeeze(0).transpose(0, 1).contiguous().numpy().reshape(L, Ct, H, W)
            return y

        # 想定外の次元はそのまま返す
        return x

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
        # acc_x,y,z と gyro_x,y,z が連続3軸で並んでいる前提なら、各3軸塊に回転を適用
        th = np.deg2rad(self.rot_deg) * np.random.uniform(-1,1)
        # 単純なZ軸近傍回転（必要に応じて拡張）
        Rz = np.array([[ np.cos(th), -np.sin(th), 0],
                       [ np.sin(th),  np.cos(th), 0],
                       [ 0,           0,          1]], dtype=np.float32)
        def rot3(block):
            return (block @ Rz.T).astype(np.float32)
        imu_out = imu.copy()
        # 例: 先頭3列=acc, 次の3列=gyro などに合わせて編集
        if imu.shape[1] >= 3:
            imu_out[:, :3] = rot3(imu[:, :3])
        if imu.shape[1] >= 6:
            imu_out[:, 3:6] = rot3(imu[:, 3:6])
        return imu_out

    # ---------- tof ops ----------
    def _tof_channel_dropout(self, tof):
        Ct = tof.shape[1]
        ch = np.random.randint(0, Ct)
        tof[:, ch] = 0.0
        return tof

    def _tof_erasing(self, tof):
        L, Ct, H, W = tof.shape
        n = np.random.randint(self.n_erase[0], self.n_erase[1]+1)
        for _ in range(n):
            eh = np.random.randint(self.erase_hw[0][0], self.erase_hw[0][1]+1)
            ew = np.random.randint(self.erase_hw[1][0], self.erase_hw[1][1]+1)
            t  = np.random.randint(0, L)
            y0 = np.random.randint(0, max(1, H - eh + 1))
            x0 = np.random.randint(0, max(1, W - ew + 1))
            tof[t, :, y0:y0+eh, x0:x0+ew] = 0.0
        return tof

    def _tof_shift(self, tof):
        dy = np.random.choice([-1, 0, 1])
        dx = np.random.choice([-1, 0, 1])
        if dy == 0 and dx == 0: return tof
        out = np.full_like(tof, self.pad_value)
        if dy >= 0:
            ys = slice(dy, None)
            yd = slice(0, tof.shape[2]-dy)
        else:
            ys = slice(0, dy)
            yd = slice(-dy, None)
        if dx >= 0:
            xs = slice(dx, None)
            xd = slice(0, tof.shape[3]-dx)
        else:
            xs = slice(0, dx)
            xd = slice(-dx, None)
        out[:, :, yd, xd] = tof[:, :, ys, xs]
        return out

    def _tof_gain_noise(self, tof):
        gain = _randu(*self.tof_gain)
        noise = np.random.randn(*tof.shape).astype(np.float32) * self.tof_sigma
        return tof * gain + noise

    # ---------- main ----------
    def __call__(self, imu: np.ndarray, tof: np.ndarray):
        L = imu.shape[0]
        # 時間系
        if np.random.rand() < self.p_time_shift:
            shift = int(np.round(_randu(-self.max_shift_ratio, self.max_shift_ratio) * L))
            imu = self._time_shift(imu, shift)
            tof = self._time_shift(tof, shift)
        if np.random.rand() < self.p_time_warp:
            s = _randu(self.warp_min, self.warp_max)
            imu = self._time_warp(imu, s)
            tof = self._time_warp(tof, s)
        if np.random.rand() < self.p_block_dropout:
            imu = self._block_dropout(imu)
            tof = self._block_dropout(tof)

        # IMU
        if np.random.rand() < self.p_imu_jitter: imu = self._imu_jitter(imu)
        if np.random.rand() < self.p_imu_scale:  imu = self._imu_scale(imu)
        if np.random.rand() < self.p_imu_drift:  imu = self._imu_drift(imu)
        if self.p_imu_small_rot > 0 and np.random.rand() < self.p_imu_small_rot:
            imu = self._imu_small_rot(imu)

        # ToF
        if np.random.rand() < self.p_tof_ch_drop:  tof = self._tof_channel_dropout(tof)
        if np.random.rand() < self.p_tof_erasing:  tof = self._tof_erasing(tof)
        if np.random.rand() < self.p_tof_shift:    tof = self._tof_shift(tof)
        if np.random.rand() < self.p_tof_gain_noise: tof = self._tof_gain_noise(tof)

        return imu.astype(np.float32), tof.astype(np.float32)


# 学習用 Dataset を差し替え
class FixedLenToFDatasetAug(FixedLenToFDataset):
    def __init__(self, X_imu_list, X_tof_list, y, augmenter: AugmentIMUToF):
        super().__init__(X_imu_list, X_tof_list, y)
        self.aug = augmenter
    def __getitem__(self, idx):
        imu = self.Xi[idx].copy()
        tof = self.Xt[idx].copy()
        imu, tof = self.aug(imu, tof)
        xi = torch.from_numpy(imu)              # [L,C]
        xt = torch.from_numpy(tof)              # [L,Ct,H,W]
        yy = torch.tensor(self.y[idx], dtype=torch.long)
        return xi, xt, yy


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

    # y は [B] or [B,C] に整形
    ys = torch.stack([torch.as_tensor(y) for y in ys])
    return x_imu_pad, x_tof_pad, ys


def mixup_pad_collate_fn_tof(alpha: float = 0.2):
    if alpha <= 0:
        return collate_pad_tof

    def _collate(batch):
        x_imu, x_tof, y = collate_pad_tof(batch)
        B = x_imu.size(0)
        perm = torch.randperm(B)

        lam = float(np.random.beta(alpha, alpha))
        lam = max(lam, 1.0 - lam)  # 弱混合の抑制

        x_imu_mix = lam * x_imu + (1.0 - lam) * x_imu[perm]
        x_tof_mix = lam * x_tof + (1.0 - lam) * x_tof[perm]

        y_a, y_b = y, y[perm]
        return x_imu_mix, x_tof_mix, y_a, y_b, lam

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


# ---- pad/truncate（train/val で方針を変える、mask なし）----
def crop_or_pad_pair_np(xi: np.ndarray, xt: np.ndarray, L: int, mode: str, pad_value: float):
    T = xi.shape[0]
    assert xt.shape[0] == T
    if T >= L:
        if mode == "random":
            start = np.random.randint(0, T - L + 1)
        elif mode == "head":
            start = 0
        elif mode == "tail":
            start = T - L
        else:  # center
            start = max((T - L) // 2, 0)
        return xi[start:start+L], xt[start:start+L]
    # pad（先頭詰め）
    C = xi.shape[1]
    Ct, H, W = xt.shape[1], xt.shape[2], xt.shape[3]
    out_i = np.full((L, C), pad_value, dtype=xi.dtype)
    out_t = np.full((L, Ct, H, W), pad_value, dtype=xt.dtype)
    out_i[:T] = xi
    out_t[:T] = xt
    return out_i, out_t

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

        # 固定長 & トリミング方針
        self.max_seq_len: Optional[int] = self.cfg.data.max_seq_len
        self.pad_percentile: int = self.cfg.data.pad_percentile
        self.pad_value: float = self.cfg.data.pad_value
        self.truncate_mode_train: str = self.cfg.data.truncate_mode_train # random/head/tail/center
        self.truncate_mode_val:   str = self.cfg.data.truncate_mode_val

        # 後で埋める
        self.num_classes: int = 0
        self.feat_cols: List[str] = []
        self.imu_cols:  List[str] = []
        self.tof_cols:  List[str] = []
        self.tof_shape: Tuple[int,int,int] = (5, 8, 8)  # (Ct,H,W)
        self.class_weight: torch.Tensor | None = None

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
        self.feat_cols = np.load(self.export_dir / "feature_cols.npy", allow_pickle=True).tolist()
        self.imu_cols  = np.load(self.export_dir / "imu_cols.npy",    allow_pickle=True).tolist()
        self.tof_cols  = np.load(self.export_dir / "tof_cols.npy",    allow_pickle=True).tolist()
        scaler: StandardScaler = joblib.load(self.export_dir / "scaler.pkl")

        self.imu_ch = len(self.imu_cols)

        # ToF 形状
        ct = self.cfg.data.tof_in_channels
        if getattr(self.cfg.data, "tof_hw", None) is not None:
            h, w = tuple(self.cfg.data.tof_hw)
            assert ct * h * w == len(self.tof_cols), "tof_hw と tof_cols 数が一致しません"
            self.tof_shape = (ct, h, w)
        else:
            self.tof_shape = _infer_tof_shape(len(self.tof_cols), ct=ct, default_hw=(8, 8))

        if self.cfg.data.tof_null_thresh is not None:
            drop_info = []
            bad_ids = []
            for sid, seq in df.groupby("sequence_id"):
                tof_np = seq[self.tof_cols].to_numpy(dtype=np.float32, copy=False)
                total = int(tof_np.size)
                nulls = int(np.count_nonzero(tof_np == -1))
                frac = (nulls / total) if total > 0 else 1.0
                if frac >= self.cfg.data.tof_null_thresh:
                    bad_ids.append(sid)
                    drop_info.append((sid, nulls, total, frac))
            if bad_ids:
                # 除外 & ログ保存
                df = df[~df["sequence_id"].isin(bad_ids)].copy()
                pd.DataFrame(drop_info, columns=["sequence_id", "n_null", "n_total", "null_frac"])\
                  .to_csv(self.export_dir / "dropped_sequences_tof_null.csv", index=False)
                print(f"[DataModule] Dropped {len(bad_ids)} sequences by ToF null ratio >= {self.cfg.data.tof_null_thresh}")

        # スケール適用
        df_feat = df[self.feat_cols].copy()
        df_feat = df_feat.replace(-1, 0).fillna(0)
        df_feat = df_feat.mask(df_feat == 0, 1e-3)
        df[self.feat_cols] = df_feat
        df[self.feat_cols] = scaler.transform(df_feat.values)

        # ---- シーケンス毎に numpy へ ----
        X_imu_list: List[np.ndarray] = []
        X_tof_list: List[np.ndarray] = []
        y_list: List[int] = []
        subjects: List[str] = []
        seq_ids: List[str] = []

        subj_col = "subject" if "subject" in df.columns else None
        imu_idx = [self.feat_cols.index(c) for c in self.imu_cols]
        tof_idx = [self.feat_cols.index(c) for c in self.tof_cols]
        Ct, H, W = self.tof_shape

        lengths: List[int] = []
        for sid, seq in df.groupby("sequence_id"):
            arr = seq[self.feat_cols].to_numpy(dtype=np.float32, copy=False)  # [T, F]
            T = arr.shape[0]
            lengths.append(T)

            imu = arr[:, imu_idx]                                 # [T, C_imu]
            tof_flat = arr[:, tof_idx]                             # [T, Ct*H*W]
            tof = tof_flat.reshape(T, Ct, H, W)                    # [T, Ct, H, W]

            X_imu_list.append(imu)
            X_tof_list.append(tof)
            y_list.append(int(seq["gesture_int"].iloc[0]))
            subjects.append(seq[subj_col].iloc[0] if subj_col else int(sid))
            seq_ids.append(sid)

        y_int = np.asarray(y_list, dtype=np.int64)

        # ---- 固定長 L を決める ----
        if self.max_seq_len is not None and int(self.max_seq_len) > 0:
            L = int(self.max_seq_len)
        else:
            L = int(np.percentile(lengths, self.pad_percentile))
        self.max_seq_len = L
        np.save(self.export_dir / "sequence_maxlen.npy", L)
        np.save(self.export_dir / "tof_shape.npy", np.array(self.tof_shape))

        # ---- split（subject があれば StratifiedGroupKFold）----
        idx_all = np.arange(len(X_imu_list))

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                                random_state=self.cfg.data.random_seed)
        tr_idx, val_idx = list(skf.split(idx_all, y_int))[self.fold_idx]
        classes_arr = np.array(classes).tolist()

        def pack(indices):
            return {
                str(seq_ids[i]): {
                    "subject": subjects[i],
                    "gesture": classes_arr[int(y_int[i])]
                } for i in indices
            }

        split_map = {
            "fold": int(self.fold_idx),
            "train": pack(tr_idx),
            "val":   pack(val_idx),
        }

        # JSON
        with open(self.export_dir / f"seq_split_fold{self.fold_idx}.json", "w", encoding="utf-8") as f:
            json.dump(split_map, f, ensure_ascii=False, indent=2)

        plot_val_gesture_distribution(split_map["val"], save_path=self.export_dir / f"val_gesture_dist_fold_{self.fold_idx}.png")

        Xtr_imu, Xtr_tof, ytr = [], [], []
        for i in tr_idx:
            xi, xt = crop_or_pad_pair_np(X_imu_list[i], X_tof_list[i], L, self.truncate_mode_train, self.pad_value)
            Xtr_imu.append(xi)
            Xtr_tof.append(xt)
            ytr.append(y_int[i])

        Xva_imu, Xva_tof, yva = [], [], []
        for i in val_idx:
            xi, xt = crop_or_pad_pair_np(X_imu_list[i], X_tof_list[i], L, self.truncate_mode_val, self.pad_value)
            Xva_imu.append(xi)
            Xva_tof.append(xt)
            yva.append(y_int[i])

        ytr = np.asarray(ytr, dtype=np.int64)
        yva = np.asarray(yva, dtype=np.int64)

        if self.cfg.aug.no_aug:
            # Augmenter を使わない場合は素の Dataset
            self._train_ds = FixedLenToFDataset(Xtr_imu, Xtr_tof, ytr)
        else:
            self._train_ds = FixedLenToFDatasetAug(Xtr_imu, Xtr_tof, ytr, 
                AugmentIMUToF(
                    p_time_shift=self.cfg.aug.p_time_shift, max_shift_ratio=self.cfg.aug.max_shift_ratio,
                    p_time_warp=self.cfg.aug.p_time_warp, warp_min=self.cfg.aug.warp_min, warp_max=self.cfg.aug.warp_max,
                    p_block_dropout=self.cfg.aug.p_block_dropout, n_blocks=self.cfg.aug.n_blocks, block_len=self.cfg.aug.block_len,
                    p_imu_jitter=self.cfg.aug.p_imu_jitter, imu_sigma=self.cfg.aug.imu_sigma,
                    p_imu_scale=self.cfg.aug.p_imu_scale, imu_scale_sigma=self.cfg.aug.imu_scale_sigma,
                    p_imu_drift=self.cfg.aug.p_imu_drift, drift_std=self.cfg.aug.drift_std, drift_clip=self.cfg.aug.drift_clip,
                    p_imu_small_rot=self.cfg.aug.p_imu_small_rot,   # 必要なら0.2程度に
                    p_tof_ch_drop=self.cfg.aug.p_tof_ch_drop,
                    p_tof_erasing=self.cfg.aug.p_tof_erasing, erase_hw=self.cfg.aug.erase_hw, n_erase=self.cfg.aug.n_erase,
                    p_tof_shift=self.cfg.aug.p_tof_shift,
                    p_tof_gain_noise=self.cfg.aug.p_tof_gain_noise, tof_gain=self.cfg.aug.tof_gain, tof_sigma=self.cfg.aug.tof_sigma,
                    pad_value=self.cfg.aug.pad_value,
                )
            )
        self._val_ds   = FixedLenToFDataset(Xva_imu, Xva_tof, yva)  # 検証は素通し

        # ---- class_weight & steps/epoch ----
        self.class_weight = torch.tensor(
            compute_class_weight(class_weight="balanced", classes=np.arange(self.num_classes), y=ytr),
            dtype=torch.float32
        )
        self.steps_per_epoch = math.ceil(len(tr_idx) / self.batch)

    # --------------------------
    def train_dataloader(self):
        return DataLoader(
            self._train_ds,
            batch_size=self.batch,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_val,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )
