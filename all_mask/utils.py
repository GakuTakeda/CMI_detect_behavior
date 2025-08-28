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
    add_linear_acc: bool = False,
    add_energy_feats: bool = False,
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

    # --- 角速度 & 角距離（クォータニオン系列に対して）---
    rot_data = df[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
    ang_vel = calculate_angular_velocity_from_quat(rot_data)         # ndarray [T,3]
    df['angular_vel_x'] = ang_vel[:, 0]
    df['angular_vel_y'] = ang_vel[:, 1]
    df['angular_vel_z'] = ang_vel[:, 2]
    # CMIFeDataset と揃える（angular_distance）
    df['angular_distance'] = calculate_angular_distance(rot_data)

    # --- 直線加速度（重力除去）---
    if add_linear_acc:
        lin_list = []
        for _, g in df.groupby('sequence_id', sort=False):
            acc_g = g[['acc_x', 'acc_y', 'acc_z']]
            rot_g = g[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
            lin = remove_gravity_from_acc(acc_g, rot_g)  # [len(g), 3]
            lin_list.append(pd.DataFrame(
                lin, columns=['linear_acc_x', 'linear_acc_y', 'linear_acc_z'], index=g.index
            ))
        lin_df = pd.concat(lin_list).sort_index()
        df = pd.concat([df, lin_df], axis=1)
        df['linear_acc_mag'] = np.sqrt(
            df['linear_acc_x']**2 + df['linear_acc_y']**2 + df['linear_acc_z']**2
        )
        df['linear_acc_mag_jerk'] = df.groupby('sequence_id')['linear_acc_mag'].diff().fillna(0)

    # --- エネルギー系（任意、クラスの compute_cross_axis_energy に合わせた命名）---
    if add_energy_feats:
        def _cross_energy(group: pd.DataFrame) -> pd.DataFrame:
            axes = ['x', 'y', 'z']
            feats = {}
            # パワー和（|FFT|^2 の総和）
            for ax in axes:
                fft_val = np.fft.fft(group[f'acc_{ax}'].values)
                feats[f'er_{ax}'] = float(np.sum(np.abs(fft_val)**2))
            # 比率
            for i, a1 in enumerate(axes):
                for a2 in axes[i+1:]:
                    feats[f'er_r_{a1}{a2}'] = feats[f'er_{a1}'] / (feats[f'er_{a2}'] + 1e-6)
            # 相関（振幅スペクトル間）
            for i, a1 in enumerate(axes):
                for a2 in axes[i+1:]:
                    c = np.corrcoef(np.abs(np.fft.fft(group[f'acc_{a1}'].values)),
                                    np.abs(np.fft.fft(group[f'acc_{a2}'].values)))[0, 1]
                    feats[f'er_c_{a1}{a2}'] = float(c)
            return pd.DataFrame(feats, index=[group.index[0]])

        er_df = df.groupby('sequence_id', group_keys=False).apply(_cross_energy)
        df = df.join(er_df, how='left')
        df[er_df.columns] = df.groupby('sequence_id')[er_df.columns].ffill()

    # --- ToF 集約（生ピクセルは 'tof_{i}_v{p}' 前提；値 -1 は欠損扱い）---
    if tof_mode != 0:
        new_cols = {}
        # 単一モード or 複数モード（-1 → 2/4/8/16/32）
        modes = [tof_mode] if tof_mode != -1 else [2, 4, 8, 16, 32]

        for i in tof_sensor_ids:
            pix_cols = [c for c in df.columns if c.startswith(f'tof_{i}_v')]
            if len(pix_cols) == 0:
                continue  # このセンサが無ければスキップ

            tof_pix = df[pix_cols].replace(-1, np.nan)

            # 全体統計
            for stat in tof_region_stats:
                if stat == 'mean':
                    new_cols[f'tof_{i}_mean'] = tof_pix.mean(axis=1)
                elif stat == 'std':
                    new_cols[f'tof_{i}_std']  = tof_pix.std(axis=1)
                elif stat == 'min':
                    new_cols[f'tof_{i}_min']  = tof_pix.min(axis=1)
                elif stat == 'max':
                    new_cols[f'tof_{i}_max']  = tof_pix.max(axis=1)

            # リージョン統計
            for m in modes:
                if m <= 1:
                    continue
                assert tof_pix.shape[1] % m == 0, f"ToF pixels(={tof_pix.shape[1]}) not divisible by mode={m}"
                region_size = tof_pix.shape[1] // m
                for r in range(m):
                    reg = tof_pix.iloc[:, r*region_size:(r+1)*region_size]
                    prefix = f'tof{m}_{i}_region_{r}_'
                    for stat in tof_region_stats:
                        if stat == 'mean':
                            new_cols[prefix+'mean'] = reg.mean(axis=1)
                        elif stat == 'std':
                            new_cols[prefix+'std']  = reg.std(axis=1)
                        elif stat == 'min':
                            new_cols[prefix+'min']  = reg.min(axis=1)
                        elif stat == 'max':
                            new_cols[prefix+'max']  = reg.max(axis=1)

        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # --- 列の並びを調整：最初の thm_/tof_ の直前に IMU派生を挿入 ---
    #   CMIFeDataset の命名に合わせて一覧を用意
    insert_cols = [
        'acc_mag', 'rot_angle', 'acc_mag_jerk', 'rot_angle_vel',
        'angular_vel_x', 'angular_vel_y', 'angular_vel_z',
        'angular_distance'
    ]
    if add_linear_acc:
        insert_cols += ['linear_acc_x', 'linear_acc_y', 'linear_acc_z', 'linear_acc_mag', 'linear_acc_mag_jerk']
    if add_energy_feats:
        insert_cols += [
            'er_x', 'er_y', 'er_z',
            'er_r_xy', 'er_r_xz', 'er_r_yz',
            'er_c_xy', 'er_c_xz', 'er_c_yz'
        ]

    # 実在する列だけに絞る
    insert_cols = [c for c in insert_cols if c in df.columns]

    cols = list(df.columns)
    insert_index = len(cols)
    for i, c in enumerate(cols):
        if c.startswith('thm_') or c.startswith('tof_'):
            insert_index = i
            break

    # まだ入っていないなら挿入
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

# ---------- main model (cfg-driven) ----------
class TwoBranch_mask(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        mcfg = cfg.model

        # ---- modalities ----
        imu_dim = int(mcfg.modalities.imu_dim)
        tof_dim = int(mcfg.modalities.tof_dim)
        assert imu_dim + tof_dim == int(mcfg.model.num_channels), \
            "num_channels は imu_dim + tof_dim と一致させてください。"

        self.imu_dim, self.tof_dim = imu_dim, tof_dim
        self.num_classes = int(mcfg.model.num_classes)

        # ---- branches config ----
        ps = mcfg.cnn.pool_sizes             # 例: [2,2]
        imu_filters  = mcfg.branches.imu.filters
        imu_kernels  = mcfg.branches.imu.kernels
        imu_drops    = mcfg.branches.imu.dropouts
        tof_filters  = mcfg.branches.tof.filters
        tof_kernels  = mcfg.branches.tof.kernels
        tof_drops    = mcfg.branches.tof.dropouts
        # パディング→長さ更新で使う pool_sizes を各ブロック回数に合わせて展開
        self._imu_pool_sizes = ps[:len(imu_filters)] + [ps[-1]] * max(0, len(imu_filters)-len(ps))
        self._tof_pool_sizes = ps[:len(tof_filters)] + [ps[-1]] * max(0, len(tof_filters)-len(ps))

        # ---- IMU residual+SE blocks ----
        self.imu_blocks = None
        if imu_dim > 0:
            blocks = []
            in_ch = imu_dim
            for f, k, d, p in zip(imu_filters, imu_kernels, imu_drops, self._imu_pool_sizes):
                print(imu_kernels)
                print(k)
                blocks.append(ResidualSEConv1D(in_ch, f, k, pool=p, drop=d))
                in_ch = f
            self.imu_blocks = nn.Sequential(*blocks)
            imu_out_ch = imu_filters[-1]
        else:
            imu_out_ch = 0

        # ---- ToF light conv blocks ----
        self.tof_blocks = None
        if tof_dim > 0:
            blocks = []
            in_ch = tof_dim
            for f, k, d, p in zip(tof_filters, tof_kernels, tof_drops, self._tof_pool_sizes):
                blocks.append(ConvBlock1D(in_ch, f, k=k, pool=p, drop=d))
                in_ch = f
            self.tof_blocks = nn.Sequential(*blocks)
            tof_out_ch = tof_filters[-1]
        else:
            tof_out_ch = 0

        merged_feat = int(imu_out_ch + tof_out_ch)

        # ---- RNNs ----
        rcfg = mcfg.rnn
        self.bigru = nn.GRU(
            input_size=merged_feat,
            hidden_size=int(rcfg.hidden_size),
            num_layers=int(rcfg.num_layers),
            dropout=float(rcfg.dropout) if int(rcfg.num_layers) > 1 else 0.0,
            bidirectional=bool(rcfg.bidirectional),
            batch_first=True,
        )
        self.bilstm = nn.LSTM(
            input_size=merged_feat,
            hidden_size=int(rcfg.hidden_size),
            num_layers=int(rcfg.num_layers),
            dropout=float(rcfg.dropout) if int(rcfg.num_layers) > 1 else 0.0,
            bidirectional=bool(rcfg.bidirectional),
            batch_first=True,
        )
        bi = 2 if bool(rcfg.bidirectional) else 1

        # ---- time-distributed dense + noise ----
        self.noise = GaussianNoise(float(mcfg.noise.std))
        self.timed_fc  = nn.Linear(merged_feat, 16)
        self.timed_act = nn.ELU(inplace=True)

        att_in = int(rcfg.hidden_size)*bi + int(rcfg.hidden_size)*bi + 16
        self.attn = TimeAttention(att_in)

        # ---- meta features（全特徴）----
        self.use_meta = True
        meta_dim = int(mcfg.meta.proj_dim)
        self.meta_dropout = float(mcfg.meta.dropout)
        self.meta_proj = nn.Sequential(
            nn.Linear(5 * (imu_dim + tof_dim), meta_dim),
            nn.BatchNorm1d(meta_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.meta_dropout)
        )

        # ---- Head ----
        hcfg = mcfg.head
        h_hidden = int(hcfg.hidden)
        h_drop   = float(hcfg.dropout)
        self.head = nn.Sequential(
            nn.Linear(att_in + meta_dim, h_hidden, bias=False),
            nn.BatchNorm1d(h_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(h_drop),
            nn.Linear(h_hidden, self.num_classes)
        )

    @staticmethod
    def _to_ncl(x):  # [B,T,C] -> [B,C,T]
        return x.transpose(1, 2).contiguous()
    @staticmethod
    def _to_ntc(x):  # [B,C,T] -> [B,T,C]
        return x.transpose(1, 2).contiguous()

    def _post_pool_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        # 各ブランチで pool してから結合するので、長さは同じ回数だけ縮む前提（構成を合わせている）
        # IMU/ToF でブロック数が違う場合は、より短い方に合わせるのが安全 → ここでは「両者のpool適用回数の最大」を採用
        ps_imu = self._imu_pool_sizes
        ps_tof = self._tof_pool_sizes
        # 長さ更新は「両方のpool列のうち長い方」に合わせる（=多く縮む想定）
        pool_seq = ps_imu if len(ps_imu) >= len(ps_tof) else ps_tof
        return _down_len(lengths, pool_seq)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B,T,C], lengths:[B], mask:[B,T](True=有効) or None
        """
        B, T, C = x.shape
        device = x.device

        # split
        x_imu = x[:, :, :self.imu_dim] if self.imu_dim > 0 else x[:, :, 0:0]
        x_tof = x[:, :, self.imu_dim:self.imu_dim + self.tof_dim] if self.tof_dim > 0 else x[:, :, 0:0]

        feats = []
        if self.imu_blocks is not None:
            z = self._to_ncl(x_imu)     # [B,C,T] -> blocks -> [B,F,L']
            z = self.imu_blocks(z)
            z = self._to_ntc(z)         # [B,L',F]
            feats.append(z)
        if self.tof_blocks is not None:
            y = self._to_ncl(x_tof)
            y = self.tof_blocks(y)
            y = self._to_ntc(y)
            feats.append(y)

        merged = feats[0] if len(feats) == 1 else torch.cat(feats, dim=2)  # [B,T',F]
        Tp = merged.size(1)

        # lengths/mask after pooling
        lengths_cnn = self._post_pool_lengths(lengths)
        if mask is None:
            mask = (torch.arange(T, device=device)[None, :] < lengths[:, None])
        mask_cnn = (torch.arange(Tp, device=device)[None, :] < lengths_cnn[:, None])

        # RNNs (pack/pad)
        packed = nn.utils.rnn.pack_padded_sequence(merged, lengths_cnn.cpu(),
                                                   batch_first=True, enforce_sorted=False)
        gru_p, _  = self.bigru(packed)
        lstm_p,_  = self.bilstm(packed)
        gru_out, _  = nn.utils.rnn.pad_packed_sequence(gru_p,  batch_first=True, total_length=Tp)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_p, batch_first=True, total_length=Tp)

        noisy   = self.noise(merged)
        td_feat = self.timed_act(self.timed_fc(noisy))

        rnn_cat = torch.cat([gru_out, lstm_out, td_feat], dim=2)  # [B,T',D]
        pooled  = self.attn(rnn_cat, mask=mask_cnn)               # [B,D_att]

        # meta（全特徴で mean/std/min/max/abs-mean の5*C）
        with torch.no_grad():
            if mask is None:
                valid = torch.ones(B, T, dtype=torch.bool, device=device)
            else:
                valid = mask
        m = valid.unsqueeze(-1)
        cnt = m.sum(dim=1).clamp_min(1)
        xm = x * m
        mean = xm.sum(dim=1) / cnt
        var  = ((xm - mean.unsqueeze(1)) * m).pow(2).sum(dim=1) / cnt
        std  = torch.sqrt(var + 1e-6)
        x_min = x.masked_fill(~m, float('inf')).min(dim=1).values
        x_max = x.masked_fill(~m, float('-inf')).max(dim=1).values
        abs_mean = xm.abs().sum(dim=1) / cnt
        meta = torch.cat([mean, std, x_min, x_max, abs_mean], dim=1)

        meta_proj = self.meta_proj(meta)
        fused = torch.cat([pooled, meta_proj], dim=1)

        logits = self.head(fused)
        return logits


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
        self.model = TwoBranch_mask(cfg)

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
    
   