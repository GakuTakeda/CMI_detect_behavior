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
from sklearn.metrics import f1_score
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


def extract_seq(train, target):
    train_df = train.copy()
    list_ = []
    for _, df in train_df.groupby("sequence_id"):
        if df["gesture"].iloc[0] == target:
            list_.append(df)
    
    return list_


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

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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


class SequenceDatasetVarLen(Dataset):
    def __init__(self, X_list, y_array, augmenter=None):
        """
        X_list: list[np.ndarray or torch.Tensor]、各要素は [Ti, C]（IMUのみ）
        y_array: np.ndarray[int]  クラスID（one-hot不要）
        augmenter: callable(x)->x  ※時系列オーグメント（任意）
        """
        self.X = [torch.as_tensor(x, dtype=torch.float32) for x in X_list]
        self.y = torch.as_tensor(y_array, dtype=torch.long)
        self.aug = augmenter

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        if self.aug is not None:
            x = self.aug(x)  # ここで [Ti,C] のまま拡張
        return x, self.y[i]

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



def collate_pad(batch):
    xs, ys = zip(*batch)                        # x: [T,C] 可変長, y: int でも one-hot でもOK
    xs = [torch.as_tensor(x, dtype=torch.float32) for x in xs]
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)  # [B,Tmax,C]
    Tmax = padded.size(1)
    mask = (torch.arange(Tmax)[None, :] < lengths[:, None])         # [B,Tmax], True=有効
    ys = torch.stack([torch.as_tensor(y) for y in ys])              # [B] or [B,num_classes]
    return padded, lengths, mask, ys

def mixup_pad_collate_fn(alpha: float = 0.2, return_soft: bool = False):
    """
    alpha<=0 なら通常の collate_pad を返す。
    return_soft=True:
        y が one-hot/確率ラベルなら y_mix を返す。
        y が int クラスIDでも、soft CE 用に自動 one-hot 化（num_classes はバッチ内最大+1で近似）。
    return_soft=False:
        2つのターゲット (y_a, y_b) と係数 lam を返し、loss を
        lam*CE(logits,y_a) + (1-lam)*CE(logits,y_b) で計算する方式。
    """
    if alpha <= 0:
        return collate_pad

    def _collate(batch):
        x, lengths, mask, y = collate_pad(batch)             # x:[B,T,C], lengths:[B], mask:[B,T]
        B = x.size(0)
        perm = torch.randperm(B)
        lam = float(np.random.beta(alpha, alpha))
        lam = max(lam, 1.0 - lam)  # (推奨) 極端な弱混合を防ぐ

        x_mix = lam * x + (1.0 - lam) * x[perm]              # [B,T,C]

        # マスクは OR（どちらか有効なら有効）→ 長さは True の数
        mask_mix = mask | mask[perm]                          # [B,T]
        lengths_mix = mask_mix.sum(dim=1)                     # [B]

        if return_soft:
            if y.dim() == 1:  # int クラスID → とりあえずバッチ内で one-hot 化
                num_classes = int(y.max().item() + 1)
                y = F.one_hot(y, num_classes=num_classes).to(x.dtype)
            y_mix = lam * y + (1.0 - lam) * y[perm]
            return x_mix, lengths_mix, mask_mix, y_mix
        else:
            # ハードラベルCE × 2本の合成
            y_a, y_b = y, y[perm]
            return x_mix, lengths_mix, mask_mix, y_a, y_b, lam

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

    

class GaussianNoise(nn.Module):
    def __init__(self, stddev=0.1):
        super().__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training and self.stddev > 0:
            noise = torch.randn_like(x) * self.stddev
            return x + noise
        return x
    
    
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


# ---------------------------
# Model (完全版)
# ---------------------------
class ModelVariant_LSTMGRU(nn.Module):
    """
    Per-channel CNN → BiGRU & BiLSTM → Noise Skip → AttentionPooling → Head
    可変長対応（pack）＆マスク対応
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        C = cfg.model.model.num_channels
        num_classes = cfg.model.model.num_classes

        # ----- Meta -----
        self.meta_extractor = MetaFeatureExtractor()
        self.meta_dense = nn.Sequential(
            nn.Linear(5 * C, cfg.model.meta.proj_dim),
            nn.BatchNorm1d(cfg.model.meta.proj_dim),
            nn.ReLU(),
            nn.Dropout(cfg.model.meta.dropout),
        )

        # ----- Per-channel CNN branches -----
        ks = list(cfg.model.cnn.multiscale.kernel_sizes)          # e.g. [3,5,7]
        out_per_kernel = int(cfg.model.cnn.multiscale.out_per_kernel)
        ms_out = out_per_kernel * len(ks)                         # ← 実チャネル数
        se_out = int(cfg.model.cnn.se.out_channels)

        # 構成: MSConv (長さ保存) → ResidualSE(pool) → ResidualSE(pool)
        branch = lambda: nn.Sequential(
            MultiScaleConv1d(1, out_per_kernel, kernel_sizes=ks), # [N, ms_out, L]
            EnhancedResidualSEBlock(ms_out, se_out, k=3, pool_size=cfg.model.cnn.pool_sizes[0], drop=cfg.model.cnn.se.drop),
            EnhancedResidualSEBlock(se_out, se_out, k=3, pool_size=cfg.model.cnn.pool_sizes[1], drop=cfg.model.cnn.se.drop),
        )
        self.branches = nn.ModuleList([branch() for _ in range(C)])

        per_step_feat = se_out * C

        # ----- RNNs -----
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

        # ----- Noise skip -----
        self.noise = GaussianNoise(cfg.model.noise.std)

        # ----- Attention Pooling -----
        gru_dim   = cfg.model.rnn.hidden_size * bi
        lstm_dim  = cfg.model.rnn.hidden_size * bi
        noise_dim = per_step_feat
        attn_in_dim = gru_dim + lstm_dim + noise_dim
        self.attention_pooling = AttentionLayer(attn_in_dim)

        # ----- Head -----
        head_hidden = cfg.model.head.hidden
        self.head_1 = nn.Sequential(
            nn.LazyLinear(head_hidden),         # pooled次元が変わっても対応
            nn.BatchNorm1d(head_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.model.head.dropout),
            nn.Linear(head_hidden, num_classes),
        )

        # CNN後の pool_size 群（ブロック2回 → [2,2]）
        self._cnn_pool_sizes = [2, 2]

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        x: [B, T, C]
        lengths: [B]  (真の時系列長)
        mask: [B, T]  (True=有効)  ※無くてもOK。内部で lengths から作れます。
        """
        B, T, C = x.shape

        # ----- meta (マスク付き) -----
        if mask is None:
            mask = (torch.arange(T, device=x.device)[None, :] < lengths[:, None])
        meta = self.meta_extractor(x, mask)           # [B, 5*C]
        meta_proj = self.meta_dense(meta)             # [B, meta_dim]

        # ----- per-channel CNN -----
        outs = []
        # Conv1d は [N,C_in,L] を期待するので (B,1,T) に置き換え
        for i in range(C):
            ci = x[:, :, i].unsqueeze(1)             # [B,1,T]
            o  = self.branches[i](ci)                # [B, se_out, T']
            outs.append(o.transpose(1, 2))           # -> [B,T',se_out]
        combined = torch.cat(outs, dim=2)            # [B,T',se_out*C]
        Tp = combined.size(1)

        # ----- 長さの正確な更新 -----
        lengths_cnn = down_len(lengths, self._cnn_pool_sizes)     # [B]

        # ----- RNN (pack) -----
        packed = pack_padded_sequence(combined, lengths=lengths_cnn.cpu(),
                                      batch_first=True, enforce_sorted=False)
        gru_packed, _  = self.bigru(packed)
        lstm_packed, _ = self.bilstm(packed)

        gru_out, _  = pad_packed_sequence(gru_packed, batch_first=True, total_length=Tp)   # [B,Tp,gru_dim]
        lstm_out, _ = pad_packed_sequence(lstm_packed, batch_first=True, total_length=Tp)  # [B,Tp,lstm_dim]

        noise_out   = self.noise(combined)           # [B,Tp,se_out*C]

        rnn_cat = torch.cat([gru_out, lstm_out, noise_out], dim=2)  # [B,Tp,attn_in_dim]

        # ----- Attention pooling (pad無視) -----
        mask_cnn = (torch.arange(Tp, device=lengths_cnn.device)[None, :] < lengths_cnn[:, None])  # [B,Tp]
        pooled = self.attention_pooling(rnn_cat, mask=mask_cnn)     # [B,P]

        # ----- Head -----
        fused = torch.cat([pooled, meta_proj], dim=1)                # [B, P + meta_dim]
        z_cls = self.head_1(fused)                                   # [B, num_classes]
        return z_cls


class litmodel(L.LightningModule):
    """
    LightningModule wrapping ModelVariant_LSTMGRU (variable-length + mask)
    入力:  x         … [B, T, C]
          lengths   … [B]
          mask      … [B, T] (True=有効)
    ラベル: train(MixUpあり): y_a, y_b, lam  /  val/test: y
    """
    def __init__(
        self,
        cfg,
        num_classes: int,
        lr_init: float = 1e-3,
        weight_decay: float = 1e-4,
        cls_loss_weight: float = 1.0,
        reg_loss_weight: float = 0.1,                 # 使わないなら0でもOK
        class_weight: torch.Tensor | None = None,     # CE のクラス重み (np/torch どちらでも)
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.model = ModelVariant_LSTMGRU(cfg)

        # metrics
        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.val_acc   = MulticlassAccuracy(num_classes=num_classes, average="macro")

        # class weight（あるときだけbuffer登録）
        if class_weight is not None:
            cw = torch.as_tensor(class_weight, dtype=torch.float32)
            self.register_buffer("class_weight", cw)
        else:
            self.class_weight = None

        # optional regression (未使用なら残してOK)
        self.mse = nn.MSELoss()

    # ------------------------------------------------------------
    def forward(self, x, lengths, mask):
        return self.model(x, lengths, mask)  # -> logits

    # ---- 汎用: Cross Entropy（hard/soft両対応） ----
    def _ce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        target: 形状 [B] (int) or [B,C] (one-hot/prob)
        class_weight は hard CE のときのみ適用
        """
        if target.ndim == 2 and target.dtype != torch.long:
            # soft labels（確率 or one-hot float）
            logp = F.log_softmax(logits, dim=1)
            loss = -(target * logp).sum(dim=1).mean()
            return loss
        else:
            hard = target if target.ndim == 1 else target.argmax(dim=1)
            ce = nn.CrossEntropyLoss(weight=self.class_weight)
            return ce(logits, hard)

    # ---- 汎用: accuracy（target の形式に自動対応） ----
    def _acc(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        hard = target if target.ndim == 1 else target.argmax(dim=1)
        return self.train_acc(preds, hard) if self.training else self.val_acc(preds, hard)

    # ------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        """
        mixup_pad_collate_fn(return_soft=False) の想定:
            batch = (x, lengths, mask, y_a, y_b, lam)
        ※ return_soft=True を使うなら batch=(x, lengths, mask, y_mix)
        """
        if len(batch) == 6:
            x, lengths, mask, y_a, y_b, lam = batch
            logits = self.forward(x, lengths, mask)

            loss_a = self._ce(logits, y_a)
            loss_b = self._ce(logits, y_b)
            loss   = self.hparams.cls_loss_weight * (lam * loss_a + (1.0 - lam) * loss_b)

            # 精度は期待値として合成（報告用）
            preds = logits.argmax(dim=1)
            acc_a = self._acc(preds, y_a)
            acc_b = self._acc(preds, y_b)
            acc   = lam * acc_a + (1.0 - lam) * acc_b

            self.log_dict(
                {"train/loss": loss, "train/acc": acc},
                on_step=False, on_epoch=True, prog_bar=True
            )
            return loss

        elif len(batch) == 4:
            # もし return_soft=True を使っている場合はこちら
            x, lengths, mask, y_mix = batch
            logits = self.forward(x, lengths, mask)
            loss   = self.hparams.cls_loss_weight * self._ce(logits, y_mix)
            preds  = logits.argmax(dim=1)
            acc    = self._acc(preds, y_mix)

            self.log_dict(
                {"train/loss": loss, "train/acc": acc},
                on_step=False, on_epoch=True, prog_bar=True
            )
            return loss

        else:
            raise RuntimeError(f"Unexpected train batch format: len={len(batch)}")

    # ------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        # collate_pad: (x, lengths, mask, y)
        if len(batch) == 4:
            x, lengths, mask, y = batch
        elif len(batch) == 6:
            # 万一train用collateが来ても対処
            x, lengths, mask, y, _, _ = batch
        else:
            raise RuntimeError(f"Unexpected val batch format: len={len(batch)}")

        logits = self.forward(x, lengths, mask)
        loss   = self.hparams.cls_loss_weight * self._ce(logits, y)
        preds  = logits.argmax(dim=1)
        acc    = self._acc(preds, y)

        self.log_dict(
            {"val/loss": loss, "val/acc": acc},
            on_step=False, on_epoch=True, prog_bar=True
        )

    def test_step(self, batch, batch_idx):
        # 形式は val と同じでOK
        if len(batch) == 4:
            x, lengths, mask, y = batch
        elif len(batch) == 6:
            x, lengths, mask, y, _, _ = batch
        else:
            raise RuntimeError(f"Unexpected test batch format: len={len(batch)}")

        logits = self.forward(x, lengths, mask)
        loss   = self.hparams.cls_loss_weight * self._ce(logits, y)

        self.log_dict({"test/loss": loss}, on_step=False, on_epoch=True)

    # ------------------------------------------------------------
    def configure_optimizers(self):
        opt = Adam(self.parameters(),
                   lr=self.hparams.lr_init,
                   weight_decay=self.hparams.weight_decay)
        sch = {
            "scheduler": ReduceLROnPlateau(opt, mode="min", patience=2, factor=0.5),
            "monitor": "val/loss",
        }
        return {"optimizer": opt, "lr_scheduler": sch}
    
    
class MacroImuModel(nn.Module):

    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.num_channels = 48
        ch = self.num_channels

        # 1. Meta features --------------------------------------------------
        self.meta_extractor = MetaFeatureExtractor()          # (B, 5*C)
        self.meta_dense = nn.Sequential(
            nn.Linear(5 * ch, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # 2. Per-channel CNN ----------------------------------------------
        self.branches = nn.ModuleList([
            nn.Sequential(
                MultiScaleConv1d(1, 12, kernel_sizes=[3, 5, 7]),      # 12 × 3 = 36
                EnhancedResidualSEBlock(36, 48, 3, drop=0.3),
                EnhancedResidualSEBlock(48, 48, 3, drop=0.3),
            )
            for _ in range(ch)
        ])                         # 出力: (B, L', 48) × ch → 結合後 (B, L', 48*ch)

        # 3-a. BiGRU / 3-b. BiLSTM / 3-c. LMU & Noise ----------------------
        rnn_in = 48 * ch              # 720
        self.bigru  = nn.GRU(rnn_in, 128, num_layers=2,
                             batch_first=True, bidirectional=True, dropout=0.2)
        self.bilstm = nn.LSTM(rnn_in, 128, num_layers=2,
                              batch_first=True, bidirectional=True, dropout=0.2)
        self.lmu    = LMU(rnn_in, hidden_size=128, memory_size=256, theta=127)
        self.noise  = GaussianNoise(0.09)          # そのまま加算-正規化用

        # rnn_cat のチャンネル数を計算
        self.rnn_feat_dim = (128*2) + (128*2) + 128 + rnn_in   # 256+256+128+720 = 1360

        # 4. Attention Pooling --------------------------------------------
        self.attention_pool = AttentionLayer(self.rnn_feat_dim)  # → (B, 1360)

        # 5. 分類ヘッド ----------------------------------------------------
        in_feat = self.rnn_feat_dim + 32        # pooled + meta
        self.classifier = nn.Sequential(
            nn.Linear(in_feat, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)         # 8-class logits
        )

    # ---------------- forward ----------------
    def forward(self, x: torch.Tensor):
        """
        x: (B, L, C=15)
        """
        # meta
        meta        = self.meta_extractor(x)            # (B, 5*C)
        meta_proj   = self.meta_dense(meta)             # (B, 32)

        # CNN per-channel
        branch_outs = []
        for i in range(x.size(2)):
            ci   = x[:, :, i].unsqueeze(1)              # (B,1,L)
            out  = self.branches[i](ci)                 # (B, 48, L')
            branch_outs.append(out.transpose(1, 2))     # (B, L', 48)
        combined = torch.cat(branch_outs, dim=2)        # (B, L', 48*C)

        # RNN + LMU
        gru_out, _  = self.bigru(combined)              # (B, L', 256)
        lstm_out, _ = self.bilstm(combined)             # (B, L', 256)
        lmu_out, _  = self.lmu(combined)                # (B, L', 128)
        noise_out   = self.noise(combined)              # (B, L', 720)

        rnn_cat = torch.cat([gru_out, lstm_out, lmu_out, noise_out], dim=2)  # (B, L', 1360)

        # Attention pooling
        pooled = self.attention_pool(rnn_cat)           # (B, 1360)

        # Fuse & classify
        fused  = torch.cat([pooled, meta_proj], dim=1)  # (B, 1392)
        logits = self.classifier(fused)                 # (B, 8)

        return logits          # ← CrossEntropyLoss にそのまま投入

