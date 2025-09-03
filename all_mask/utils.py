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


def feature_eng(
    df: pd.DataFrame,
    *,
    rot_fillna: bool = False,
    add_linear_acc: bool = True,
    add_energy_feats: bool = True,
    tof_mode: int = 1,
    tof_region_stats: tuple[str, ...] = ("mean", "std", "min", "max"),
    tof_sensor_ids: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> pd.DataFrame:
    """
    CMIFeDataset の命名/仕様に揃えた特徴量生成。
    - IMU派生: acc_mag/rot_angle/jerk系, 角速度, 角距離(=angular_distance)
    - 直線加速度: remove_gravity_from_acc による linear_acc_x/y/z (+ magnitude/jerk)
    - ToF: sensorごとの mean/std/min/max と、必要なら 1D分割リージョン統計
           (tof_mode>1 → 'tof{mode}_{i}_region_{r}_{stat}', tof_mode==-1 → {2,4,8,16,32})
    - 列順: 最初の thm_/tof_ カラムの直前に IMU派生を挿入（既存 DataModule と互換）
    """
    # --- IMU 基本 ---
    if rot_fillna:
        df['rot_w'] = df['rot_w'].fillna(1)
        df[['rot_x', 'rot_y', 'rot_z']] = df[['rot_x', 'rot_y', 'rot_z']].fillna(0)

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

class AugmentMultiModal:
    """
    x: np.ndarray [L, C_all] を受け取り、同形状を返す
    時間系の変換は x 全体に一括 → その後に各モダリティ別ノイズ
    """
    def __init__(
        self,
        idx_imu, idx_thm, idx_tof,
        # 時間系（全体）
        p_time_shift=0.7, max_shift_ratio=0.1,
        p_time_warp=0.5,  warp_min=0.9, warp_max=1.1,
        p_block_dropout=0.5, n_blocks=(1,3), block_len=(2,6),
        pad_value=0.0,
        # IMUノイズ
        p_imu_jitter=0.9, imu_sigma=0.03,
        p_imu_scale=0.5,  imu_scale_sigma=0.03,
        p_imu_drift=0.5,  drift_std=0.003, drift_clip=0.3,
        p_imu_small_rot=0.0, rot_deg=5.0,
        # THM/ToF ノイズ（列ベース）
        p_thm_ch_drop=0.2, thm_drop_frac=0.05, thm_sigma=0.01,
        p_tof_ch_drop=0.2, tof_drop_frac=0.05, tof_sigma=0.01,
    ):
        self.idx_imu = np.asarray(idx_imu)
        self.idx_thm = np.asarray(idx_thm)
        self.idx_tof = np.asarray(idx_tof)

        # 時間系
        self.p_time_shift=p_time_shift; self.max_shift_ratio=max_shift_ratio
        self.p_time_warp=p_time_warp;   self.warp_min=warp_min; self.warp_max=warp_max
        self.p_block_dropout=p_block_dropout; self.n_blocks=n_blocks; self.block_len=block_len
        self.pad_value=pad_value

        # IMU
        self.p_imu_jitter=p_imu_jitter; self.imu_sigma=imu_sigma
        self.p_imu_scale=p_imu_scale;   self.imu_scale_sigma=imu_scale_sigma
        self.p_imu_drift=p_imu_drift;   self.drift_std=drift_std; self.drift_clip=drift_clip
        self.p_imu_small_rot=p_imu_small_rot; self.rot_deg=rot_deg

        # THM/ToF
        self.p_thm_ch_drop=p_thm_ch_drop; self.thm_drop_frac=thm_drop_frac; self.thm_sigma=thm_sigma
        self.p_tof_ch_drop=p_tof_ch_drop; self.tof_drop_frac=tof_drop_frac; self.tof_sigma=tof_sigma

        self._np = np; self._torch = torch; self._F = F

    # ---- 時間系（全体）----
    def _time_shift(self, x, shift):
        L = x.shape[0]
        if shift == 0: return x
        out = self._np.full_like(x, self.pad_value)
        if shift > 0:
            out[shift:] = x[:L-shift]
        else:
            out[:L+shift] = x[-shift:]
        return out

    def _time_warp(self, x, scale):
        # 線形補間で L→Lp→L
        t = self._torch.from_numpy(x.astype(self._np.float32)).transpose(0,1).unsqueeze(0)  # [1,C,L]
        L = t.size(-1)
        Lp = max(2, int(round(L * scale)))
        y  = self._F.interpolate(t, size=Lp, mode="linear", align_corners=False)
        y  = self._F.interpolate(y, size=L,  mode="linear", align_corners=False)
        return y.squeeze(0).transpose(0,1).contiguous().numpy()

    def _block_dropout(self, x):
        L = x.shape[0]
        nb = self._np.random.randint(self.n_blocks[0], self.n_blocks[1]+1)
        for _ in range(nb):
            bl = self._np.random.randint(self.block_len[0], self.block_len[1]+1)
            s  = self._np.random.randint(0, max(1, L - bl + 1))
            x[s:s+bl] = self.pad_value
        return x

    # ---- IMU ノイズ ----
    def _imu_jitter(self, m): return m + self._np.random.randn(*m.shape).astype(self._np.float32) * self.imu_sigma
    def _imu_scale(self, m):
        scale = (1.0 + self._np.random.randn(m.shape[1]).astype(self._np.float32) * self.imu_scale_sigma)
        return m * scale[None, :]
    def _imu_drift(self, m):
        L,C = m.shape
        drift = self._np.cumsum(self._np.random.randn(L,C).astype(self._np.float32) * self.drift_std, axis=0)
        self._np.clip(drift, -self.drift_clip, self.drift_clip, out=drift)
        return m + drift
    def _imu_small_rot(self, m):
        # x/y/z の3軸が連続している前提なら軽い回転（チャネルが >=6 なら加速度&ジャイロに同回転）
        th = self._np.deg2rad(self.rot_deg) * self._np.random.uniform(-1,1)
        Rz = self._np.array([[ self._np.cos(th), -self._np.sin(th), 0],
                             [ self._np.sin(th),  self._np.cos(th), 0],
                             [ 0, 0, 1]], dtype=self._np.float32)
        out = m.copy()
        if m.shape[1] >= 3: out[:, :3] = (m[:, :3] @ Rz.T).astype(self._np.float32)
        if m.shape[1] >= 6: out[:, 3:6] = (m[:, 3:6] @ Rz.T).astype(self._np.float32)
        return out

    # ---- 汎用：列ドロップ＋ノイズ ----
    def _ch_drop_and_noise(self, m, p_drop, drop_frac, sigma):
        if m.shape[1] == 0: return m
        if self._np.random.rand() < p_drop and m.shape[1] > 0:
            k = max(1, int(round(m.shape[1] * drop_frac)))
            cols = self._np.random.choice(m.shape[1], size=k, replace=False)
            m[:, cols] = 0.0
        if sigma > 0:
            m = m + self._np.random.randn(*m.shape).astype(self._np.float32) * sigma
        return m

    # ---- main ----
    def __call__(self, x):
        x = x.astype(self._np.float32, copy=True)

        # --- 全体の時間操作 ---
        L = x.shape[0]
        if self._np.random.rand() < self.p_time_shift:
            shift = int(round(self._np.random.uniform(-self.max_shift_ratio, self.max_shift_ratio) * L))
            x = self._time_shift(x, shift)
        if self._np.random.rand() < self.p_time_warp:
            s = float(self._np.random.uniform(self.warp_min, self.warp_max))
            x = self._time_warp(x, s)
        if self._np.random.rand() < self.p_block_dropout:
            x = self._block_dropout(x)

        # --- モダリティ別ノイズ ---
        def view(idx): return x[:, idx] if idx.size > 0 else x[:, 0:0]

        # IMU
        if self.idx_imu.size > 0:
            imu = view(self.idx_imu)
            if self._np.random.rand() < 1.0:                  # フラグで順序調整OK
                if self._np.random.rand() < self.p_imu_jitter: imu = self._imu_jitter(imu)
                if self._np.random.rand() < self.p_imu_scale:  imu = self._imu_scale(imu)
                if self._np.random.rand() < self.p_imu_drift:  imu = self._imu_drift(imu)
                if self.p_imu_small_rot > 0 and self._np.random.rand() < self.p_imu_small_rot:
                    imu = self._imu_small_rot(imu)
            x[:, self.idx_imu] = imu

        # THM
        if self.idx_thm.size > 0:
            thm = view(self.idx_thm)
            thm = self._ch_drop_and_noise(thm, self.p_thm_ch_drop, self.thm_drop_frac, self.thm_sigma)
            x[:, self.idx_thm] = thm

        # ToF
        if self.idx_tof.size > 0:
            tof = view(self.idx_tof)
            tof = self._ch_drop_and_noise(tof, self.p_tof_ch_drop, self.tof_drop_frac, self.tof_sigma)
            x[:, self.idx_tof] = tof

        return x.astype(self._np.float32)


def _ensure_list(x, n=None):
    if isinstance(x, (list, tuple)):
        return list(x) if (n is None or len(x) == n) else (list(x) + [x[-1]]*(n-len(x)))
    return [x] if n is None else [x]*n

def _down_len(lengths: torch.Tensor, pool_sizes: Sequence[int]) -> torch.Tensor:
    l = lengths.clone()
    for p in pool_sizes:
        l = torch.div(l, int(p), rounding_mode="floor")
    return l.clamp_min(1)

class GaussianNoise(nn.Module):
    def __init__(self, std: float = 0.0):
        super().__init__()
        self.std = float(std)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.std > 0:
            return x + torch.randn_like(x) * self.std
        return x

class SE1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x.mean(dim=-1)                 # [B,C]
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))     # [B,C]
        return x * w.unsqueeze(-1)

class ResidualSEConv1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, pool: int = 2, drop: float = 0.3):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.se    = SE1D(out_ch, reduction=8)
        self.short = (nn.Identity() if in_ch == out_ch
                      else nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, bias=False),
                                         nn.BatchNorm1d(out_ch)))
        self.pool  = nn.MaxPool1d(pool)
        self.drop  = nn.Dropout(drop)
        self.pool_size = pool
    def forward(self, x):
        res = self.short(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.se(x)
        x = F.relu(x + res, inplace=True)
        x = self.pool(x)
        x = self.drop(x)
        return x

class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, pool: int = 2, drop: float = 0.2):
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv1d(in_ch, out_ch, k, padding=pad, bias=False)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(pool)
        self.drop = nn.Dropout(drop)
        self.pool_size = pool
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = self.pool(x)
        x = self.drop(x)
        return x

class TimeAttention(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.score = nn.Linear(in_dim, 1)
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # x: [B,T,D], mask: [B,T] (True=valid)
        s = torch.tanh(self.score(x)).squeeze(-1)     # [B,T]
        if mask is not None:
            s = s.masked_fill(~mask, float("-inf"))
        w = F.softmax(s, dim=1)                       # [B,T]
        ctx = torch.bmm(w.unsqueeze(1), x).squeeze(1) # [B,D]
        return ctx
    
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

    def _post_pool_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        ps_imu = self._imu_pool_sizes
        ps_tof = self._tof_pool_sizes
        pool_seq = ps_imu if len(ps_imu) >= len(ps_tof) else ps_tof
        return _down_len(lengths, pool_seq)

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

        # ---- RNN（pack/pad）----
        packed = nn.utils.rnn.pack_padded_sequence(merged, lengths_cnn.cpu(),
                                                   batch_first=True, enforce_sorted=False)
        rnn_p, _  = self.rnn(packed)
        rnn_out, _  = nn.utils.rnn.pad_packed_sequence(rnn_p,  batch_first=True, total_length=Tp)  # [B,Tp,rnn_out_dim]

        # ---- TCN（pack不要）----
        tcn_out = self.tcn(merged)                 # [B,Tp,tcn_out]

        # ---- Noise + TD（最後にconcatする相方）----
        noisy   = self.noise(merged)               # [B,Tp,F]
        td_feat = self.timed_act(self.timed_fc(noisy))  # [B,Tp,16]

        # ---- 最後に concat（RNN/TCN/TD）----
        seq_cat = torch.cat([rnn_out, tcn_out, td_feat], dim=2)  # [B,Tp, rnn_out_dim + tcn_out + 16]
        pooled  = self.attn(seq_cat, mask=mask_cnn)              # [B,D_att]

        # ---- meta（全特徴: mean/std/min/max/abs-mean）----
        with torch.no_grad():
            valid = mask if mask is not None else torch.ones(B, T, dtype=torch.bool, device=device)
        m   = valid.unsqueeze(-1)
        cnt = m.sum(dim=1).clamp_min(1)
        xm  = x * m
        mean = xm.sum(dim=1) / cnt
        var = ( (xm - mean.unsqueeze(1)) * m ).pow(2).sum(dim=1) / cnt
        std = torch.sqrt(var + self.eps)

        # min/max は無効部を +inf/-inf にする
        x_min = x.masked_fill(~m, float('inf')).min(dim=1).values
        x_max = x.masked_fill(~m, float('-inf')).max(dim=1).values
        abs_mean = xm.abs().sum(dim=1) / cnt

        feats = torch.cat([mean, std, x_min, x_max, abs_mean], dim=1)  # [B, 5*C]
        return feats


# ===== Model: per-channel CNN → (TimeMixer) → NoiseSkip → AttnPool → Head =====
class ModelVariant_WAVE_mask(nn.Module):  # ← クラス名は据え置き（中身はWaveNet無し）
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        C_total = int(cfg.model.model.num_channels)
        num_classes = int(cfg.model.model.num_classes)

        # ---- modalities ----
        imu_dim = int(getattr(cfg.model.modalities, "imu_dim", C_total))
        tof_dim = int(getattr(cfg.model.modalities, "tof_dim", 0))
        assert imu_dim + tof_dim == C_total, "num_channels は imu_dim + tof_dim と一致させて下さい。"
        self.imu_dim, self.tof_dim = imu_dim, tof_dim
        self.use_coattn = (imu_dim > 0 and tof_dim > 0)

        # ----- Meta -----
        self.meta_extractor = MetaFeatureExtractor_mask()
        self.meta_dense = nn.Sequential(
            nn.Linear(5 * C_total, int(cfg.model.meta.proj_dim)),
            nn.BatchNorm1d(int(cfg.model.meta.proj_dim)),
            nn.ReLU(),
            nn.Dropout(float(cfg.model.meta.dropout)),
        )

        # ----- Per-channel CNN branches -----
        ks = list(cfg.model.cnn.multiscale.kernel_sizes)     # e.g. [3,5,7]
        out_per_kernel = int(cfg.model.cnn.multiscale.out_per_kernel)
        ms_out = out_per_kernel * len(ks)
        se_out = int(cfg.model.cnn.se.out_channels)

        branch = lambda: nn.Sequential(
            MultiScaleConv1d_mask(1, out_per_kernel, kernel_sizes=ks),   # [N, ms_out, L]
            EnhancedResidualSEBlock_mask(ms_out, se_out, k=3,
                                         pool_size=int(cfg.model.cnn.pool_sizes[0]),
                                         drop=float(cfg.model.cnn.se.drop)),
            EnhancedResidualSEBlock_mask(se_out, se_out, k=3,
                                         pool_size=int(cfg.model.cnn.pool_sizes[1]),
                                         drop=float(cfg.model.cnn.se.drop)),
        )
        self.branches = nn.ModuleList([branch() for _ in range(C_total)])

        # 各モダリティの時刻特徴次元
        per_step_imu = se_out * imu_dim
        per_step_tof = se_out * tof_dim
        per_step_all = se_out * C_total

        # ----- Co-Attention (IMU↔ToF) -----
        if self.use_coattn:
            dim = int(getattr(cfg.model.coattn, "dim", 256))
            heads = int(getattr(cfg.model.coattn, "num_heads", 4))
            drop  = float(getattr(cfg.model.coattn, "dropout", 0.1))

            self.proj_q_imu  = nn.Linear(per_step_imu, dim, bias=False)
            self.proj_kv_tof = nn.Linear(per_step_tof, dim, bias=False)
            self.proj_q_tof  = nn.Linear(per_step_tof, dim, bias=False)
            self.proj_kv_imu = nn.Linear(per_step_imu, dim, bias=False)

            self.attn_imu = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
            self.attn_tof = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
            self.ln_imu = nn.LayerNorm(dim)
            self.ln_tof = nn.LayerNorm(dim)

            fused_in = 2 * dim
        else:
            fused_in = per_step_all  # 片側のみの時は全結合特徴を使う

        # ----- TimeMixer（WaveNetの完全撤去・代替の1×1線形）-----
        # cfg.model.timemix を読み取り（無ければ既定）
        def _get(cfgobj, key, default):
            if cfgobj is None:
                return default
            val = getattr(cfgobj, key, default)
            if val is None or (isinstance(val, str) and val.strip().lower() in ("null", "none", "")):
                return default
            return val

        tm_cfg   = getattr(cfg.model, "timemix", None)
        tm_out   = int(_get(tm_cfg, "out_channels", fused_in))
        tm_drop  = float(_get(tm_cfg, "dropout", 0.0))
        tm_act   = str(_get(tm_cfg, "activation", "elu")).lower()

        act_layer = nn.ELU if tm_act == "elu" else (nn.GELU if tm_act == "gelu" else nn.ReLU)
        if tm_out == fused_in and tm_drop == 0.0 and tm_act == "elu":
            # まったく同次元＆ドロップ無し＆標準actならIdentityでもOK
            self.time_mixer = nn.Sequential(
                nn.Linear(fused_in, tm_out, bias=False),
                nn.ELU(inplace=True),
            )
        else:
            self.time_mixer = nn.Sequential(
                nn.Linear(fused_in, tm_out, bias=False),
                act_layer(inplace=True) if tm_act != "gelu" else act_layer(),  # GELUはinplace不可
                nn.Dropout(tm_drop),
            )
        self.tm_out = tm_out

        # ----- Noise skip -----
        self.noise = GaussianNoise(float(cfg.model.noise.std))

        # ----- Attention Pooling & Head -----
        self.attention_pooling = TimeAttention(self.tm_out + fused_in)
        head_hidden = int(cfg.model.head.hidden)
        self.head_1 = nn.Sequential(
            nn.LazyLinear(head_hidden),
            nn.BatchNorm1d(head_hidden),
            nn.ReLU(),
            nn.Dropout(float(cfg.model.head.dropout)),
            nn.Linear(head_hidden, num_classes),
        )

        # CNN後の pool_size 群（ブロック2回 → [2,2]）
        self._cnn_pool_sizes = list(cfg.model.cnn.pool_sizes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        x: [B, T, C_total], lengths: [B], mask: [B, T] (True=有効)
        """
        B, T, C_total = x.shape
        device = x.device
        if mask is None:
            mask = (torch.arange(T, device=device)[None, :] < lengths[:, None])

        # ----- Meta -----
        meta      = self.meta_extractor(x, mask)      # [B, 5*C]
        meta_proj = self.meta_dense(meta)             # [B, meta_dim]

        # ----- Per-channel CNN（IMUとToFを分けて処理）-----
        outs_imu, outs_tof = [], []
        for i in range(C_total):
            ci = x[:, :, i].unsqueeze(1)             # [B,1,T]
            o  = self.branches[i](ci)                # [B, se_out, Tp]
            o  = o.transpose(1, 2)                   # [B,Tp,se_out]
            if i < self.imu_dim:
                outs_imu.append(o)
            else:
                outs_tof.append(o)

        Tp = outs_imu[0].size(1) if len(outs_imu) > 0 else outs_tof[0].size(1)
        lengths_cnn = _down_len(lengths, self._cnn_pool_sizes)
        mask_cnn = (torch.arange(Tp, device=device)[None, :] < lengths_cnn[:, None])  # True=有効
        key_padding = ~mask_cnn  # MHA用（Trueが無効）

        # 連結
        combined_imu = torch.cat(outs_imu, dim=2) if len(outs_imu) > 0 else None       # [B,Tp,se_out*imu_dim]
        combined_tof = torch.cat(outs_tof, dim=2) if len(outs_tof) > 0 else None       # [B,Tp,se_out*tof_dim]
        combined_all = torch.cat([t for t in [combined_imu, combined_tof] if t is not None], dim=2)

        # ----- Co-Attention（必要なときだけ）-----
        if self.use_coattn:
            q_imu  = self.proj_q_imu(combined_imu)             # [B,Tp,D]
            kv_tof = self.proj_kv_tof(combined_tof)            # [B,Tp,D]
            ctx_imu, _ = self.attn_imu(q_imu, kv_tof, kv_tof, key_padding_mask=key_padding, need_weights=False)
            ctx_imu = self.ln_imu(q_imu + ctx_imu)

            q_tof  = self.proj_q_tof(combined_tof)
            kv_imu = self.proj_kv_imu(combined_imu)
            ctx_tof, _ = self.attn_tof(q_tof, kv_imu, kv_imu, key_padding_mask=key_padding, need_weights=False)
            ctx_tof = self.ln_tof(q_tof + ctx_tof)

            fused_seq = torch.cat([ctx_imu, ctx_tof], dim=2)   # [B,Tp, 2*D]
        else:
            fused_seq = combined_all                            # [B,Tp, per_step_all]

        # ----- TimeMixer + Noise skip（WaveNet撤去）-----
        mix_out  = self.time_mixer(fused_seq)                   # [B,Tp, tm_out]
        noise_out= self.noise(fused_seq)                        # [B,Tp, fused_in]
        seq_feat = torch.cat([mix_out, noise_out], dim=2)       # [B,Tp, tm_out + fused_in]

        # ----- AttnPool（pad無視） → Head -----
        pooled = self.attention_pooling(seq_feat, mask=mask_cnn)  # [B, P]
        fused  = torch.cat([pooled, meta_proj], dim=1)            # [B, P + meta_dim]
        z_cls  = self.head_1(fused)                               # [B, num_classes]
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
        self.model = ModelVariant_WAVE_mask(cfg)

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
    
   