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

class SequenceDatasetVarLen(Dataset):
    def __init__(self, X_list, y_array, augmenter=None):
        self.use_aug = callable(augmenter)
        self.aug = augmenter

        self.X = []
        for x in X_list:
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            x = np.asarray(x, dtype=np.float32)
            self.X.append(x)

        self.y = torch.as_tensor(y_array, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        if self.use_aug:
            # numpyでaugmentしてからtorch化
            x = self.aug(x.copy()).astype(np.float32)
        x = torch.from_numpy(x)  # [T,C] torch.Tensor に戻す
        return x, self.y[i]


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

class GaussianNoise_mask(nn.Module):
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

class AttentionLayer_mask(nn.Module):
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


class MultiScaleConv1d_mask(nn.Module):
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


class EnhancedSEBlock_mask(nn.Module):
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


class EnhancedResidualSEBlock_mask(nn.Module):
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

        self.se     = EnhancedSEBlock_mask(out_ch)
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


class MetaFeatureExtractor_mask(nn.Module):
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


# ===== TCN blocks =====
class TCNResidualBlock_mask(nn.Module):
    """
    Depthwise(膨張)Conv1d → PointwiseConv1d を2回 + 残差 + SE + GELU
    入出力長は保持（SAME相当）。
    入出力形状: [B,T,Cin] -> [B,T,Cout]
    """
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, dilation: int = 1, drop: float = 0.2, use_se: bool = True):
        super().__init__()
        pad = dilation * (k - 1) // 2

        # Block A
        self.dw1 = nn.Conv1d(in_ch, in_ch, kernel_size=k, padding=pad, dilation=dilation,
                             groups=in_ch, bias=False)
        self.pw1 = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)

        # Block B
        self.dw2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, padding=pad, dilation=dilation,
                             groups=out_ch, bias=False)
        self.pw2 = nn.Conv1d(out_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)

        # residual
        self.short = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn_sc  = nn.Identity() if in_ch == out_ch else nn.BatchNorm1d(out_ch)

        self.se   = EnhancedSEBlock_mask(out_ch) if use_se else nn.Identity()
        self.drop = nn.Dropout(drop)

    def forward(self, x_btC: torch.Tensor) -> torch.Tensor:        # x: [B,T,C]
        x = x_btC.transpose(1, 2)                                   # -> [B,C,T]
        sc = self.bn_sc(self.short(x))

        y = self.dw1(x); y = self.pw1(y); y = self.bn1(y); y = F.gelu(y)
        y = self.dw2(y); y = self.pw2(y); y = self.bn2(y)
        y = self.se(y)
        y = F.gelu(y + sc)
        y = self.drop(y)
        return y.transpose(1, 2)                                    # -> [B,T,Cout]


class TCNStack_mask(nn.Module):
    """
    dilation を [base, base*2, base*4, ...] と増やすスタック。
    すべて長さ保存。入出力は [B,T,C]。
    """
    def __init__(self, ch: int, k: int = 3, n_layers: int = 5, base_dilation: int = 1,
                 drop: float = 0.2, use_se: bool = True, out_ch: Optional[int] = None):
        super().__init__()
        layers = []
        in_ch = ch
        for i in range(n_layers):
            d = base_dilation * (2 ** i)
            layers.append(TCNResidualBlock_mask(in_ch, ch, k=k, dilation=d, drop=drop, use_se=use_se))
            in_ch = ch
        self.blocks = nn.Sequential(*layers)

        # 出力次元の調整（必要なときだけ）
        self.to_out = nn.Identity() if (out_ch is None or out_ch == ch) else nn.Sequential(
            nn.Linear(ch, out_ch, bias=False),
            nn.LayerNorm(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:             # [B,T,C]
        y = self.blocks(x)
        return y if isinstance(self.to_out, nn.Identity) else self.to_out(y)


# ===== Model: per-channel CNN → TCN → NoiseSkip → AttnPool → Head =====
class ModelVariant_TCN_mask(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        C = cfg.model.model.num_channels
        num_classes = cfg.model.model.num_classes

        # ----- Meta -----
        self.meta_extractor = MetaFeatureExtractor_mask()
        self.meta_dense = nn.Sequential(
            nn.Linear(5 * C, cfg.model.meta.proj_dim),
            nn.BatchNorm1d(cfg.model.meta.proj_dim),
            nn.ReLU(),
            nn.Dropout(cfg.model.meta.dropout),
        )

        # ----- Per-channel CNN branches -----
        ks = list(cfg.model.cnn.multiscale.kernel_sizes)                 # e.g. [3,5,7]
        out_per_kernel = int(cfg.model.cnn.multiscale.out_per_kernel)
        ms_out = out_per_kernel * len(ks)
        se_out = int(cfg.model.cnn.se.out_channels)

        branch = lambda: nn.Sequential(
            MultiScaleConv1d_mask(1, out_per_kernel, kernel_sizes=ks),   # [N, ms_out, L]
            EnhancedResidualSEBlock_mask(ms_out, se_out, k=3, pool_size=cfg.model.cnn.pool_sizes[0], drop=cfg.model.cnn.se.drop),
            EnhancedResidualSEBlock_mask(se_out, se_out, k=3, pool_size=cfg.model.cnn.pool_sizes[1], drop=cfg.model.cnn.se.drop),
        )
        self.branches = nn.ModuleList([branch() for _ in range(C)])
        per_step_feat = se_out * C                                       # CNN後の時刻あたり特徴次元

        # ----- TCN -----
        tcn_cfg = getattr(cfg.model, "tcn", None)
        if tcn_cfg is None:
            # デフォルト（cfgが未定義でも動くように）
            tcn_k = 3; tcn_layers = 5; tcn_base = 1; tcn_drop = 0.2; tcn_out = per_step_feat
        else:
            tcn_k     = int(getattr(tcn_cfg, "kernel_size", 3))
            tcn_layers= int(getattr(tcn_cfg, "n_layers", 5))
            tcn_base  = int(getattr(tcn_cfg, "base_dilation", 1))
            tcn_drop  = float(getattr(tcn_cfg, "dropout", 0.2))
            tcn_out   = int(getattr(tcn_cfg, "out_channels", per_step_feat))

        self.tcn = TCNStack_mask(
            ch=per_step_feat, k=tcn_k, n_layers=tcn_layers,
            base_dilation=tcn_base, drop=tcn_drop, use_se=True, out_ch=tcn_out
        )

        # ----- Noise skip -----
        self.noise = GaussianNoise_mask(cfg.model.noise.std)

        # ----- Attention Pooling -----
        attn_in_dim = tcn_out + per_step_feat
        self.attention_pooling = AttentionLayer_mask(attn_in_dim)

        # ----- Head -----
        head_hidden = cfg.model.head.hidden
        self.head_1 = nn.Sequential(
            nn.LazyLinear(head_hidden),
            nn.BatchNorm1d(head_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.model.head.dropout),
            nn.Linear(head_hidden, num_classes),
        )

        # CNN後の pool_size 群（ブロック2回 → [2,2]）
        self._cnn_pool_sizes = list(cfg.model.cnn.pool_sizes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        x: [B, T, C]
        lengths: [B]
        mask: [B, T] (True=有効)  ※無ければ lengths から作成
        """
        B, T, C = x.shape
        if mask is None:
            mask = (torch.arange(T, device=x.device)[None, :] < lengths[:, None])

        # ----- Meta -----
        meta = self.meta_extractor(x, mask)                  # [B, 5*C]
        meta_proj = self.meta_dense(meta)                    # [B, meta_dim]

        # ----- Per-channel CNN -----
        outs = []
        for i in range(C):
            ci = x[:, :, i].unsqueeze(1)                     # [B,1,T]
            o  = self.branches[i](ci)                        # [B, se_out, Tp]
            outs.append(o.transpose(1, 2))                   # -> [B,Tp,se_out]
        combined = torch.cat(outs, dim=2)                    # [B,Tp, per_step_feat]
        Tp = combined.size(1)

        # ----- 長さ/マスク更新（MaxPool の分だけ短縮） -----
        lengths_cnn = down_len(lengths, self._cnn_pool_sizes)              # [B]
        mask_cnn = (torch.arange(Tp, device=lengths_cnn.device)[None, :] < lengths_cnn[:, None])  # [B,Tp]

        # ----- TCN + Noise skip -----
        tcn_out   = self.tcn(combined)                        # [B,Tp, tcn_out]
        noise_out = self.noise(combined)                      # [B,Tp, per_step_feat]
        seq_feat  = torch.cat([tcn_out, noise_out], dim=2)    # [B,Tp, tcn_out + per_step_feat]

        # ----- AttnPool（pad無視） → Head -----
        pooled = self.attention_pooling(seq_feat, mask=mask_cnn)           # [B, P]
        fused  = torch.cat([pooled, meta_proj], dim=1)                      # [B, P + meta_dim]
        z_cls  = self.head_1(fused)                                        # [B, num_classes]
        return z_cls



class litmodel_mask(L.LightningModule):
    """
    LightningModule wrapping ModelVariant_LSTMGRU_mask (variable-length + mask)
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
        self.model = ModelVariant_TCN_mask(cfg) 

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
