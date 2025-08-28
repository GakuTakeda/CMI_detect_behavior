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
    
    return list_,

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_behavior(seq, target: str):
    seq = feature_eng(seq)

    # ---------------- 行動→色 ----------------
    behav_colors = {
        "Relaxes and moves hand to target location": "#c0dffd",
        "Moves hand to target location"            : "#c0dffd",
        "Hand at target location"                  : "#c4f2c4",
        "Performs gesture"                         : "#ffc8c8",
    }

    # ---------------- 変化点で区間抽出 ----------------
    seg_id   = (seq["behavior"] != seq["behavior"].shift()).cumsum()
    grouped  = seq.groupby(seg_id, sort=False)

    # ---------------- プロット ----------------
    # ←★ ここに列を追加するだけ★→
    signals = [
        ("acc_x",   "acceleration_X"),
        ("acc_y",   "acceleration_Y"),
        ("acc_z",   "acceleration_Z"),
        ("acc_mag", "acceleration_magnitude"),
        ("rot_x",   "rotation_X"),
        ("rot_y",   "rotation_Y"),
        ("rot_z",   "rotation_Z"),
        ("rot_w",   "rotation_W"),
        ("rot_angle", "rotation_angle"),
    ]

    # 軸を「signals の要素数」だけ作成
    fig, axes = plt.subplots(len(signals), 1, figsize=(12, 2.2*len(signals)), sharex=True)

    # ▲― signals と axes の数が一致するよう保証される
    for ax, (col, label) in zip(axes, signals):
        if col not in seq.columns:
            ax.text(0.5, 0.5, f"'{col}' not found", ha="center", va="center", transform=ax.transAxes)
            ax.set_ylabel(label)
            continue

        ax.plot(seq["sequence_counter"], seq[col], label=label)
        ax.set_ylabel(label)

    # ----------- 背景塗り -----------
    legend_patches = []
    seen_behaviors = set()

    for _, segment in grouped:
        behav = segment["behavior"].iloc[0]
        if behav not in behav_colors:
            continue
        x0 = segment["sequence_counter"].iloc[0]
        x1 = segment["sequence_counter"].iloc[-1]

        for ax in axes:
            ax.axvspan(x0, x1, color=behav_colors[behav], alpha=0.25, linewidth=0)

        if behav not in seen_behaviors:
            legend_patches.append(mpatches.Patch(color=behav_colors[behav],
                                                 alpha=0.25, label=behav))
            seen_behaviors.add(behav)

    axes[0].legend(handles=legend_patches, loc="upper right", fontsize=8)
    axes[-1].set_xlabel("sequence_counter")

    plt.title(target)
    plt.tight_layout()
    plt.show()


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

# def calc_z_coodinate(df):
#     df = df.copy()
#     cols = ["x", "y", "z"]
#     df[cols] = 0
#     for _, seq df.groupby("sequence_id"):
#         for 



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

class TwoBranchModel(nn.Module):
    def __init__(self, imu_ch, tof_ch, n_classes, dropouts=[0.3, 0.3, 0.3, 0.3, 0.4, 0.5, 0.3]):
        super().__init__()

        self.imu_dim = imu_ch
        self.tof_dim = tof_ch

        self.fir_nchan = imu_ch

        weight_decay = 3e-3

        numtaps = 33  
        fir_coef = firwin(numtaps, cutoff=1.0, fs=10.0, pass_zero=False)
        fir_kernel = torch.tensor(fir_coef, dtype=torch.float32).view(1, 1, -1)
        fir_kernel = fir_kernel.repeat(imu_ch, 1, 1)  # (imu_dim, 1, numtaps)
        self.register_buffer("fir_kernel", fir_kernel)
        
        # IMU deep branch
        self.imu_block1 = ResidualSECNNBlock(imu_ch, 64, 3, dropout=dropouts[0], weight_decay=weight_decay)
        self.imu_block2 = ResidualSECNNBlock(64, 128, 5, dropout=dropouts[1], weight_decay=weight_decay)
        
        # TOF/Thermal lighter branch
        self.tof_conv1 = nn.Conv1d(tof_ch, 64, 3, padding=1, bias=False)
        self.tof_bn1 = nn.BatchNorm1d(64)
        self.tof_pool1 = nn.MaxPool1d(2)
        self.tof_drop1 = nn.Dropout(dropouts[2])
        
        self.tof_conv2 = nn.Conv1d(64, 128, 3, padding=1, bias=False)
        self.tof_bn2 = nn.BatchNorm1d(128)
        self.tof_pool2 = nn.MaxPool1d(2)
        self.tof_drop2 = nn.Dropout(dropouts[3])
        
        # BiLSTM
        self.bilstm = nn.LSTM(256, 128, bidirectional=True, batch_first=True)
        self.lstm_dropout = nn.Dropout(dropouts[4])
        
        # Attention
        self.attention = AttentionLayer(256)  # 128*2 for bidirectional
        
        # Dense layers
        self.dense1 = nn.Linear(256, 256, bias=False)
        self.bn_dense1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(dropouts[5])
        
        self.dense2 = nn.Linear(256, 128, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(dropouts[6])
        
        self.classifier = nn.Linear(128, n_classes)
        
    def forward(self, x):
        # Split input
        
        imu = x[:, :, :self.fir_nchan].transpose(1, 2)  # (batch, imu_dim, seq_len)
        tof = x[:, :, self.fir_nchan:].transpose(1, 2)  # (batch, tof_dim, seq_len)

        filtered = F.conv1d(
            imu[:, :self.fir_nchan, :],        # (B,7,T)
            self.fir_kernel,
            padding=self.fir_kernel.shape[-1] // 2,
            groups=self.fir_nchan,
        )
        
        imu = torch.cat([filtered, imu[:, self.fir_nchan:, :]], dim=1)  
        # IMU branch
        x1 = self.imu_block1(imu)
        x1 = self.imu_block2(x1)
        
        # TOF branch
        x2 = F.relu(self.tof_bn1(self.tof_conv1(tof)))
        x2 = self.tof_drop1(self.tof_pool1(x2))
        x2 = F.relu(self.tof_bn2(self.tof_conv2(x2)))
        x2 = self.tof_drop2(self.tof_pool2(x2))
        
        # Concatenate branches
        merged = torch.cat([x1, x2], dim=1).transpose(1, 2)  # (batch, seq_len, 256)
        
        # BiLSTM
        lstm_out, _ = self.bilstm(merged)
        lstm_out = self.lstm_dropout(lstm_out)
        
        # Attention
        attended = self.attention(lstm_out)
        
        # Dense layers
        x = F.relu(self.bn_dense1(self.dense1(attended)))
        x = self.drop1(x)
        x = F.relu(self.bn_dense2(self.dense2(x)))
        x = self.drop2(x)
        
        # Classification
        logits = (self.classifier(x))
        return logits
    
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
    def __init__(self, imu_dim, num_classes):
        super().__init__()
        self.imu_dim = imu_dim
        self.tof_out_dim = 32

        # IMU branches
        self.imu_branches = nn.ModuleList([
            nn.Sequential(
                MultiScaleConv1d(1,12),
                ResidualSEBlock(36,48),
                ResidualSEBlock(48,48),
            ) for _ in range(imu_dim)
        ])

        # TOF branch
        self.tof_cnn = TinyCNN(in_channels=5, out_dim=32)

        # Meta feature
        self.meta = MetaFeatureExtractor()
        self.meta_dense = nn.Sequential(
            nn.Linear(5*(imu_dim + self.tof_out_dim), 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Sequence encoders
        self.bigru = nn.GRU(48*imu_dim + self.tof_out_dim, 128, batch_first=True, bidirectional=True, num_layers=2, dropout=0.2)
        self.bilstm = nn.LSTM(48*imu_dim + self.tof_out_dim, 128, batch_first=True, bidirectional=True, num_layers=2, dropout=0.2)
        self.noise = GaussianNoise(0.09)

        # Attention + Head
        concat_dim = 256 + 256 + (48*imu_dim + self.tof_out_dim)
        self.attn = AttentionLayer(concat_dim)
        self.head = nn.Sequential(
            nn.Linear(concat_dim + 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_imu, x_tof):


        # ===== IMU branch =====
        imu_feats = []
        for i in range(x_imu.shape[2]):
            xi = x_imu[:,:,i].unsqueeze(1)  # (B,T,1)
            fi = self.imu_branches[i](xi).transpose(1,2)  # (B,T,F)
            imu_feats.append(fi)
        imu_feat = torch.cat(imu_feats, dim=2)  # (B,T,48*imu_dim)

        # ===== TOF branch =====
        B,T,C,H,W = x_tof.shape
        tof_flat = x_tof.view(B*T, C, H, W)  # (B*T,5,8,8)

        tof_feats = self.tof_cnn(tof_flat).view(B, T, -1)  # (B,T,32)


        # ===== align time dimension =====
        tof_feats = F.adaptive_avg_pool1d(tof_feats.transpose(1,2), output_size=imu_feat.size(1)).transpose(1,2)


        # ===== Meta features =====
        meta_imu = self.meta(x_imu)       # (B,5*imu_dim)
        meta_tof = self.meta(tof_feats)    # (B,5*32)
        meta = torch.cat([meta_imu, meta_tof], dim=1)
        meta = self.meta_dense(meta)       # (B,64)


        # ===== Sequence fusion =====
        seq = torch.cat([imu_feat, tof_feats], dim=2)  # (B,T,48*imu_dim+32)

        gru,_ = self.bigru(seq)   # (B,T,256)
        lstm,_ = self.bilstm(seq) # (B,T,256)
        noise = self.noise(seq)   # (B,T,48*imu_dim+32)
        x = torch.cat([gru, lstm, noise], dim=2)  # (B,T,256+256+...)


        # ===== Attention & Head =====
        x = self.attn(x)  # (B,256+256+...)


        x = torch.cat([x, meta], dim=1)  # (B, ...)
        out = self.head(x)                # (B,num_classes)

        return out

class ModelVariant_LSTMGRU(nn.Module):
    """
    IMU 専用：CNN → BiGRU ＆ BiLSTM → AttentionPooling → 2 ヘッド
    """
    def __init__(self, num_classes: int):
        super().__init__()
        num_channels = 48  # IMU チャンネル数

        # 1. Meta features
        self.meta_extractor = MetaFeatureExtractor()
        self.meta_dense = nn.Sequential(
            nn.Linear(5 * num_channels, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # 2. Per-channel CNN branches
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

        # 4. Attention Pooling (GRU 256 + LSTM 256 = 512)
        self.attention_pooling = AttentionLayer(2944)

        # 5. Prediction heads
        in_feat = 2944 + 32  # pooled + meta
        self.head_1 = nn.Sequential(
            nn.Linear(in_feat, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        self.head_2 = nn.Sequential(
            nn.Linear(in_feat, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),   # regression
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B, L, C)  – L=系列長, C=16ch
        """
        # ---------- meta ----------
        meta = self.meta_extractor(x)              # (B, 5*C)
        meta_proj = self.meta_dense(meta)          # (B, 32)

        # ---------- CNN branches ----------
        branch_outs = []
        for i in range(x.shape[2]):                # each channel
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

        # ---------- Fuse & heads ----------
        fused = torch.cat([pooled, meta_proj], dim=1)  # (B,544)
        z_cls = self.head_1(fused)
        z_reg = self.head_2(fused)
        return z_cls, z_reg

# ---------- Main Model ----------
class ModelVariant_Conf(nn.Module):
    """
    IMU 専用:
      ▸ Per-channel CNN
      ▸ BiGRU ＆ BiLSTM ＆ Conformer (並列)
      ▸ AttentionPooling
      ▸ 2-Head (分類 + 回帰)
    """
    def __init__(self,
                 num_classes: int,
                 num_channels: int = 11):
        super().__init__()

        # 1. Meta features ----------------------------------------------------
        self.meta_extractor = MetaFeatureExtractor()          # (B, 5*C)
        self.meta_dense = nn.Sequential(
            nn.Linear(5 * num_channels, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # 2. Per-channel CNN --------------------------------------------------
        self.branches = nn.ModuleList([
            nn.Sequential(
                MultiScaleConv1d(1, 12, kernel_sizes=[3, 5, 7]),
                ResidualSEBlock(36, 48, 3, drop=0.2),
                ResidualSEBlock(48, 48, 3, drop=0.2),
            )
            for _ in range(num_channels)
        ])

        proj_dim = 48 * num_channels      # 48 per-channel × #channels                         # → (B, L', 256)

        # 3-c. Conformer ------------------------------------------------------
        d_model = 256
        self.in_proj = nn.Linear(proj_dim, d_model)
        self.conformer = nn.Sequential(
            ConformerBlock(d_model=d_model, n_heads=4),
            ConformerBlock(d_model=d_model, n_heads=4),
        )                                  # → (B, L', 256)

        # 4. Attention pooling -----------------------------------------------
        rnn_feat_dim = 256 + 256 + 256     # GRU + LSTM + Conformer
        self.attention_pooling = AttentionLayer(rnn_feat_dim)  # (B, 768)

        # 5. Prediction heads -------------------------------------------------
        in_feat = rnn_feat_dim + 32
        self.head = nn.Sequential(
            nn.Linear(in_feat, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
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

    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, L, C)   - L: 時系列長, C: IMU チャンネル数
        Returns:
            z_cls: (B, num_classes)
            z_reg: (B, 1)
        """
        B, L, C = x.shape

        # ===== Meta features =====
        meta = self.meta_extractor(x)          # (B, 5*C)
        meta_proj = self.meta_dense(meta)      # (B, 32)

        # ===== CNN branches per channel =====
        branch_outs = []
        for i in range(C):
            ci = x[:, :, i].unsqueeze(1)       # (B, 1, L)
            out_i = self.branches[i](ci)       # (B, 48, L')
            branch_outs.append(out_i.transpose(1, 2))  # → (B, L', 48)

        combined = torch.cat(branch_outs, dim=2)       # (B, L', 48*C)

        # ===== BiGRU & BiLSTM =====
        gru_out, _  = self.bigru(combined)     # (B, L', 256)
        lstm_out, _ = self.bilstm(combined)    # (B, L', 256)

        # ===== Conformer =====
        conf_in = self.in_proj(combined)       # (B, L', 256)
        conf_out = self.conformer(conf_in)     # (B, L', 256)

        # ===== Concat & Attention pooling =====
        feat_cat = torch.cat([gru_out, lstm_out, conf_out], dim=2)  # (B, L', 768)
        pooled = self.attention_pooling(feat_cat)                   # (B, 768)

        # ===== Heads =====
        fused = torch.cat([pooled, meta_proj], dim=1)   # (B, 800)
        z_cls = self.head(fused)                 # (B, 1)
        return z_cls
    
class ModelVariant_Conf(nn.Module):
    """
    IMU 専用:
      ▸ Per-channel CNN
      ▸ BiGRU ＆ BiLSTM ＆ Conformer (並列)
      ▸ AttentionPooling
      ▸ 2-Head (分類 + 回帰)
    """
    def __init__(self,
                 num_classes: int,
                 num_channels: int = 15,
                 dense_drop: float = 0.2,
                 conv_drop: float = 0.3,
                 noise_std: float = 0.09,
                 ):
        super().__init__()

        # 1. Meta features ----------------------------------------------------
        self.meta_extractor = MetaFeatureExtractor()          # (B, 5*C)
        self.meta_dense = nn.Sequential(
            nn.Linear(5 * num_channels, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dense_drop),
        )

        # 2. Per-channel CNN --------------------------------------------------
        self.branches = nn.ModuleList([
            nn.Sequential(
                MultiScaleConv1d(1, 12, kernel_sizes=[3, 5, 7]),
                ResidualSEBlock(36, 48, 3, drop=conv_drop),
                ResidualSEBlock(48, 48, 3, drop=conv_drop),
            )
            for _ in range(num_channels)
        ])

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

        proj_dim = 48 * num_channels      # 48 per-channel × #channels                         # → (B, L', 256)

        # 3-c. Conformer ------------------------------------------------------
        d_model = 256
        self.in_proj = nn.Linear(proj_dim, d_model)
        self.conformer = nn.Sequential(
            ConformerBlock(d_model=d_model, n_heads=4),
            ConformerBlock(d_model=d_model, n_heads=4),
        )                                  # → (B, L', 256)

        self.noise = GaussianNoise(noise_std)

        # 4. Attention pooling -----------------------------------------------
        rnn_feat_dim = 1488       # Conformer
        self.attention_pooling = AttentionLayer(rnn_feat_dim)  # (B, 768)

        # 5. Prediction heads -------------------------------------------------
        in_feat = rnn_feat_dim + 32
        self.head = nn.Sequential(
            nn.Linear(in_feat, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )


    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, L, C)   - L: 時系列長, C: IMU チャンネル数
        Returns:
            z_cls: (B, num_classes)
            z_reg: (B, 1)
        """
        B, L, C = x.shape

        # ===== Meta features =====
        meta = self.meta_extractor(x)          # (B, 5*C)
        meta_proj = self.meta_dense(meta)      # (B, 32)

        # ===== CNN branches per channel =====
        branch_outs = []
        for i in range(C):
            ci = x[:, :, i].unsqueeze(1)       # (B, 1, L)
            out_i = self.branches[i](ci)       # (B, 48, L')
            branch_outs.append(out_i.transpose(1, 2))  # → (B, L', 48)

        combined = torch.cat(branch_outs, dim=2)       # (B, L', 48*C)

        # ===== Conformer =====
        conf_in = self.in_proj(combined)       # (B, L', 256)
        conf_out = self.conformer(conf_in)     # (B, L', 256)
        noise_out = self.noise(combined)
        gru_out, _  = self.bigru(combined)     # (B, L', 256)
        lstm_out, _ = self.bilstm(combined)    # (B, L', 256)

        out = torch.cat([gru_out, lstm_out, conf_out, noise_out], dim=2)

        pooled = self.attention_pooling(out)                   # (B, 768)

        # ===== Heads =====
        fused = torch.cat([pooled, meta_proj], dim=1)   # (B, 800)
        z_cls = self.head(fused)                # (B, 1)
        return z_cls
    
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

    # --- (2) 時間的統計: 移動平均 & 分散 -------------------------------
    base_cols = [
        'acc_x','acc_y','acc_z','rot_x','rot_y','rot_z','rot_w',
        'acc_mag','rot_angle','acc_mag_jerk','rot_angle_vel',
        'angular_vel_x','angular_vel_y','angular_vel_z','angular_dist'
    ]  # ←15本

    grouped = df.groupby('sequence_id')

    # rolling 計算 (min_periods=1 で序盤も値が入る)
    for col in base_cols:
        df[f'{col}_ma{k}']  = grouped[col].transform(lambda x: x.rolling(k, min_periods=1).mean())
        df[f'{col}_var{k}'] = grouped[col].transform(lambda x: x.rolling(k, min_periods=1).var().fillna(0))

    return df
    
def hand_anno(value):
    if value ==  'Above ear - pull hair':
        return "ear"
    elif value == 'Cheek - pinch skin':
        return "cheek"
    elif value == 'Eyebrow - pull hair':
        return "eyebrow"
    elif value == 'Eyelash - pull hair':
        return 'eyelash'
    elif value in ['Forehead - pull hairline', 'Forehead - scratch']:
        return 'forehead'
    elif value in ["Neck - pinch skin","Neck - scratch"]:
        return "neck"
    else:
        return ["ear", "cheek", "eyebrow", "eyelash", "forehead", "neck"]

    
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

