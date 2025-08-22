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
import lightning as L
from sklearn.metrics import accuracy_score, f1_score
from torchmetrics.classification import MulticlassAccuracy
from omegaconf import DictConfig
from typing import Optional, Sequence
from scipy.signal import cont2discrete
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
# def remove_outliers(df: pd.DataFrame, threshold: int = 300) -> pd.DataFrame:
#     df = df.copy()
#     df_out = df.copy()
#     tof_thm_cols = [c for c in df.columns if c.startswith("thm") or c.startswith("tof")]
#     by_seq = (
#         df[tof_thm_cols].eq(-1)      # == -1 と同じ
#         .groupby(df['sequence_id'])  # sequence_id ごと
#         .sum()                      # 列ごとの個数
#     )
#     null_id = []
#     for i in range(by_seq.shape[0]):
#         if by_seq.iloc[i].sum()/df.loc[df['sequence_id'] == by_seq.index[i], 'sequence_id'].count() > threshold:
#             null_id.append(by_seq.index[i])
#     return null_id

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
            
    return angular_dist


def feature_eng(df: pd.DataFrame) -> pd.DataFrame:
    # df = remove_gravity_from_acc_in_train(df)
    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    df['rot_angle'] = 2 * np.arccos(df['rot_w'].clip(-1, 1))
    df['acc_mag_jerk'] = df.groupby('sequence_id')['acc_mag'].diff().fillna(0)
    df['rot_angle_vel'] = df.groupby('sequence_id')['rot_angle'].diff().fillna(0)
    # df['linear_acc_mag'] = np.sqrt(df['linear_acc_x']**2 + df['linear_acc_y']**2 + df['linear_acc_z']**2)
    # df['linear_acc_mag_jerk'] = df.groupby('sequence_id')['linear_acc_mag'].diff().fillna(0)
    rot_data = df[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
    angular_vel_group = calculate_angular_velocity_from_quat(rot_data)
    df['angular_vel_x'] = angular_vel_group[:, 0]
    df['angular_vel_y'] = angular_vel_group[:, 1]
    df['angular_vel_z'] = angular_vel_group[:, 2]
    df['angular_dist'] = calculate_angular_distance(rot_data)

    # tof_pixel_cols = [f"tof_{i}_v{p}" for i in range(1, 6) for p in range(64)]
    # tof_data_np = df[tof_pixel_cols].replace(-1, np.nan).to_numpy()
    # reshaped_tof = tof_data_np.reshape(len(df), 5, 64)
    # with warnings.catch_warnings():
    #     warnings.filterwarnings('ignore', r'Mean of empty slice'); warnings.filterwarnings('ignore', r'Degrees of freedom <= 0 for slice')
    #     mean_vals, std_vals = np.nanmean(reshaped_tof, axis=2), np.nanstd(reshaped_tof, axis=2)
    #     min_vals, max_vals = np.nanmin(reshaped_tof, axis=2), np.nanmax(reshaped_tof, axis=2)
    # tof_agg_cols = []
    # for i in range(1, 6):
    #     df[f'tof_{i}_mean'], df[f'tof_{i}_std'] = mean_vals[:, i-1], std_vals[:, i-1]
    #     df[f'tof_{i}_min'], df[f'tof_{i}_max'] = min_vals[:, i-1], max_vals[:, i-1]
    #     tof_agg_cols.extend([f'tof_{i}_mean', f'tof_{i}_std', f'tof_{i}_min', f'tof_{i}_max'])
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


def pad_truncate(x: np.ndarray, pad_len: int) ->np.ndarray:
    # x: [T, C] -> [pad_len, C]
    T, C = x.shape
    if T >= pad_len:
        return x[:pad_len]
    out = x.new_zeros((pad_len, C))
    out[:T] = x
    return out

def preprocess_sequence(
        df_seq: pd.DataFrame,
        feature_cols: list[str],
        scaler: StandardScaler,
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



# --- 追加: 固定長化ユーティリティ ---


class SequenceDatasetFixedLen(Dataset):
    """スクリプト互換：50%でMixUp、one-hot(18)を返す"""
    def __init__(self, X, y_int, num_classes: int, mixup: bool, alpha: float = 0.4, p: float = 0.5):
        self.X = torch.tensor(np.array(X), dtype=torch.float32)   # [N, T, C]
        self.y = torch.tensor(y_int, dtype=torch.long)            # [N]
        self.mixup = mixup
        self.alpha = alpha
        self.p = p
        self.num_classes = num_classes

    def __len__(self): return self.X.shape[0]

    def __getitem__(self, i):
        x, y = self.X[i], self.y[i]
        if self.mixup and np.random.rand() < self.p:
            j = np.random.randint(0, len(self.X))
            lam = np.random.beta(self.alpha, self.alpha)
            x = lam * x + (1 - lam) * self.X[j]
            y_one = F.one_hot(y, num_classes=self.num_classes).float()
            yj_one = F.one_hot(self.y[j], num_classes=self.num_classes).float()
            y_soft = lam * y_one + (1 - lam) * yj_one
            return x, y_soft
        else:
            return x, F.one_hot(y, num_classes=self.num_classes).float()

def collate_fixed(batch):
    xs, ys = zip(*batch)                              # xs: list[[T,C]]
    x = torch.stack(xs, dim=0)                        # -> [B,T,C]
    y = torch.stack([torch.as_tensor(v) for v in ys]) # -> [B]
    return x, y

def mixup_fixed_collate_fn(alpha: float = 0.2, return_soft: bool = False):
    """
    返り値（length/maskなし）:
      - return_soft=False -> (x_mix, y_a, y_b, lam)
      - return_soft=True  -> (x_mix, y_mix)
    """
    if alpha <= 0:
        return collate_fixed

    def _collate(batch):
        x, y = collate_fixed(batch)               # x:[B,T,C], y:[B]
        B = x.size(0)
        perm = torch.randperm(B)
        lam = float(np.random.beta(alpha, alpha))
        lam = max(lam, 1.0 - lam)                 # 推奨: 極端回避
        x_mix = lam * x + (1.0 - lam) * x[perm]

        if return_soft:
            num_classes = int(y.max().item() + 1)
            y_onehot = F.one_hot(y, num_classes=num_classes).to(x.dtype)
            y_mix = lam * y_onehot + (1.0 - lam) * y_onehot[perm]
            return x_mix, y_mix
        else:
            y_a, y_b = y, y[perm]
            return x_mix, y_a, y_b, lam
    return _collate


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

    

class GaussianNoise(nn.Module):
    def __init__(self, stddev=0.1):
        super().__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training and self.stddev > 0:
            noise = torch.randn_like(x) * self.stddev
            return x + noise
        return x
    
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


class ModelVariant_LSTMGRU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        C = cfg.num_channels
        num_classes = cfg.num_classes

        # ----- Meta -----
        self.meta_extractor = MetaFeatureExtractor()
        self.meta_dense = nn.Sequential(
            nn.Linear(5 * C, cfg.model.meta.proj_dim),
            nn.BatchNorm1d(cfg.model.meta.proj_dim),
            nn.ReLU(),
            nn.Dropout(cfg.model.meta.dropout),
        )

        # ----- Per-channel CNN branches -----
        ks = list(cfg.model.cnn.multiscale.kernel_sizes)
        out_per_kernel = int(cfg.model.cnn.multiscale.out_per_kernel)
        ms_out = out_per_kernel * len(ks)
        se_out = int(cfg.model.cnn.se.out_channels)

        def branch():
            return nn.Sequential(
                MultiScaleConv1d(1, out_per_kernel, kernel_sizes=ks),
                EnhancedResidualSEBlock(ms_out, se_out, k=3, pool_size=cfg.model.cnn.pool_sizes[0], drop=cfg.model.cnn.se.drop),
                EnhancedResidualSEBlock(se_out, se_out, k=3, pool_size=cfg.model.cnn.pool_sizes[1], drop=cfg.model.cnn.se.drop),
            )
        self.branches = nn.ModuleList([branch() for _ in range(C)])

        per_step_feat = se_out * C

        # ----- RNNs（pack なしでOK） -----
        bi = 2 if cfg.model.rnn.bidirectional else 1
        self.bigru = nn.GRU(
            input_size=per_step_feat,
            hidden_size=cfg.model.rnn.hidden_size,
            num_layers=cfg.model.rnn.num_layers,
            batch_first=True,
            bidirectional=cfg.model.rnn.bidirectional,
            dropout=cfg.model.rnn.dropout,
        )
        self.bilstm = nn.LSTM(
            input_size=per_step_feat,
            hidden_size=cfg.model.rnn.hidden_size,
            num_layers=cfg.model.rnn.num_layers,
            batch_first=True,
            bidirectional=cfg.model.rnn.bidirectional,
            dropout=cfg.model.rnn.dropout,
        )

        self.noise = GaussianNoise(cfg.model.noise.std)

        gru_dim   = cfg.model.rnn.hidden_size * bi
        lstm_dim  = cfg.model.rnn.hidden_size * bi
        noise_dim = per_step_feat
        attn_in_dim = gru_dim + lstm_dim + noise_dim
        self.attention_pooling = AttentionLayer(attn_in_dim)

        head_hidden = cfg.model.head.hidden
        self.head_1 = nn.Sequential(
            nn.LazyLinear(head_hidden),
            nn.BatchNorm1d(head_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.model.head.dropout),
            nn.Linear(head_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape

        meta = self.meta_extractor(x)      # [B, 5*C]
        meta_proj = self.meta_dense(meta)             # [B, meta_dim]

        # per-channel CNN（Conv1d は [N,C_in,L]）
        outs = []
        for i in range(C):
            ci = x[:, :, i].unsqueeze(1)             # [B,1,T]
            o  = self.branches[i](ci)                # [B, se_out, T']
            outs.append(o.transpose(1, 2))           # -> [B,T',se_out]
        combined = torch.cat(outs, dim=2)            # [B,T',se_out*C]

        # RNN（pack 不要）
        gru_out, _  = self.bigru(combined)           # [B,T',gru_dim]
        lstm_out, _ = self.bilstm(combined)          # [B,T',lstm_dim]
        noise_out   = self.noise(combined)           # [B,T',se_out*C]

        rnn_cat = torch.cat([gru_out, lstm_out, noise_out], dim=2)  # [B,T',attn_in_dim]

        pooled = self.attention_pooling(rnn_cat)  # [B,P]

        # Head
        fused = torch.cat([pooled, meta_proj], dim=1)        # [B,P+meta]
        z_cls = self.head_1(fused)                           # [B,num_classes]
        return z_cls

def _to_onehot(y, num_classes):
    # y が int クラスIDでも one-hot/soft でもOKにする
    if isinstance(y, torch.Tensor):
        if y.ndim == 1 or y.dtype == torch.long:
            return F.one_hot(y, num_classes=num_classes).float()
        return y.float()
    # 万一 Tensor 以外で来た場合のフォールバック
    y = torch.as_tensor(y)
    if y.ndim == 1 or y.dtype == torch.long:
        return F.one_hot(y, num_classes=num_classes).float()
    return y.float()


class litmodel(L.LightningModule):
    def __init__(self, cfg, lr_init: float = 1e-3, weight_decay: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.net = ModelVariant_LSTMGRU(cfg)
        self.cfg = cfg

        # for epoch-end metrics
        self._preds, self._trues = [], []

    def forward(self, x):
        logits = self.net(x)   # スクリプト同様 head_1 を使用
        return logits

    @staticmethod
    def soft_ce(logits, soft_targets):
        logp = F.log_softmax(logits, dim=1)
        return -(soft_targets * logp).sum(dim=1).mean()

    def training_step(self, batch, batch_idx):
        # 2要素 (x, y_soft/onehot) も 4要素 (x, y_a, y_b, lam) も許容
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
            x = x.to(self.device, non_blocking=True)
            y = _to_onehot(y, self.cfg.num_classes).to(self.device, non_blocking=True)
        elif isinstance(batch, (list, tuple)) and len(batch) == 4:
            x, y_a, y_b, lam = batch
            x = x.to(self.device, non_blocking=True)
            y_a = _to_onehot(y_a, self.cfg.num_classes).to(self.device, non_blocking=True)
            y_b = _to_onehot(y_b, self.cfg.num_classes).to(self.device, non_blocking=True)
            # lam は float or Tensor のことが多い。全サンプル同一係数想定。
            if not torch.is_tensor(lam):
                lam = torch.tensor(lam, dtype=x.dtype, device=x.device)
            # ブロードキャスト用に [B,1] へ
            if lam.ndim == 0:
                lam = lam.view(1, 1)
            elif lam.ndim == 1:
                lam = lam.view(-1, 1)
            y = lam * y_a + (1.0 - lam) * y_b
        else:
            raise RuntimeError(f"Unexpected batch format: type={type(batch)}, len={len(batch) if isinstance(batch,(list,tuple)) else 'n/a'}")

        logits = self(x)
        loss = self.soft_ce(logits, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # 検証は通常 (x, y) のはずだが、4要素が来ても y_a を正解とみなして処理
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
        elif isinstance(batch, (list, tuple)) and len(batch) == 4:
            x, y, _, _ = batch   # y_a を採用
        else:
            raise RuntimeError(f"Unexpected val batch format: type={type(batch)}, len={len(batch) if isinstance(batch,(list,tuple)) else 'n/a'}")

        x = x.to(self.device, non_blocking=True)
        y = _to_onehot(y, self.cfg.num_classes).to(self.device, non_blocking=True)

        logits = self(x)
        pred = logits.argmax(dim=1).detach().cpu().numpy()
        true = y.argmax(dim=1).detach().cpu().numpy()

        self._preds.append(pred)
        self._trues.append(true)

    def on_validation_epoch_end(self):
        if len(self._preds) == 0: return
        preds = np.concatenate(self._preds); trues = np.concatenate(self._trues)
        self._preds.clear(); self._trues.clear()

        # 18-class
        acc_18 = accuracy_score(trues, preds)
        f1_18  = f1_score(trues, preds, average="macro")

        # binary/9-class（スクリプト互換：id<9 を 0, それ以外 1 ／ 9クラスは id%9）
        to_bin = lambda a: (a < 9).astype(int)
        to_9   = lambda a: (a % 9).astype(int)

        f1_bin = f1_score(to_bin(trues), to_bin(preds), average="macro")
        f1_9   = f1_score(to_9(trues), to_9(preds), average="macro")
        acc_bin = accuracy_score(to_bin(trues), to_bin(preds))
        acc_9   = accuracy_score(to_9(trues), to_9(preds))
        f1_avg  = (f1_bin + f1_9) / 2.0

        # log
        self.log_dict({
            "val/acc_18": acc_18, "val/f1_18": f1_18,
            "val/acc_bin": acc_bin, "val/f1_bin": f1_bin,
            "val/acc_9": acc_9, "val/f1_9": f1_9,
            "val/f1_avg": f1_avg,
        }, prog_bar=True)

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=self.hparams.lr_init, weight_decay=self.hparams.weight_decay)
        sch = {
            "scheduler": ReduceLROnPlateau(opt, mode="max", patience=3, factor=0.5),
            "monitor": "val/f1_18",   # スクリプトの sched.step(1 - f1_18) と同等に「最大化」で運用
        }
        return {"optimizer": opt, "lr_scheduler": sch}
