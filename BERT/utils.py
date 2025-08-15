import os
import torch
from pathlib import Path
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm.notebook import tqdm
from torch.amp import autocast
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, LabelEncoder
from transformers import BertConfig, BertModel
import random
from torch.utils.data import Dataset, DataLoader
import math
import pandas as pd
from torch.utils.data.dataloader import default_collate
import lightning as L
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from torchmetrics.classification import MulticlassAccuracy
from sklearn.metrics import f1_score
from typing import Optional, Sequence, Dict, Any, Union, Tuple
import numpy as np, joblib, pathlib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from hydra.core.hydra_config import HydraConfig
import math

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
    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    df['rot_angle'] = 2 * np.arccos(df['rot_w'].clip(-1, 1))
    df['acc_mag_jerk'] = df.groupby('sequence_id')['acc_mag'].diff().fillna(0)
    df['rot_angle_vel'] = df.groupby('sequence_id')['rot_angle'].diff().fillna(0)
    df['linear_acc_mag'] = np.sqrt(df['linear_acc_x']**2 + df['linear_acc_y']**2 + df['linear_acc_z']**2)
    df['linear_acc_mag_jerk'] = df.groupby('sequence_id')['linear_acc_mag'].diff().fillna(0)
    rot_data = df[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
    angular_vel_group = calculate_angular_velocity_from_quat(rot_data)
    df['angular_vel_x'] = angular_vel_group[:, 0]
    df['angular_vel_y'] = angular_vel_group[:, 1]
    df['angular_vel_z'] = angular_vel_group[:, 2]
    df['angular_distance'] = calculate_angular_distance(rot_data)


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


class SEBlock(nn.Module):
    def __init__(self, channels, reduction = 8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, L)
        se = F.adaptive_avg_pool1d(x, 1).squeeze(-1)      # -> (B, C)
        se = F.relu(self.fc1(se), inplace=True)          # -> (B, C//r)
        se = self.sigmoid(self.fc2(se)).unsqueeze(-1)    # -> (B, C, 1)
        return x * se                

class ResNetSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, wd = 1e-4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        # SE
        self.se = SEBlock(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          padding=0, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x) :
        identity = self.shortcut(x)              # (B, out, L)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)                       # (B, out, L)
        out = out + identity
        return self.relu(out)

class Bert_model(nn.Module):
    """
    IMU: 1D-ResNetSE ×2(各ブロック末尾でMaxPool1d) → [T'×feat_dim]
    BERT(embeddingはinputs_embeds)＋CLS → Head
    可変長(lengths)・mask対応（BERT attention_mask まで伝播）
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # --- CNN branch ---
        self.imu_branch = nn.Sequential(
            self.residual_se_cnn_block(cfg.dim,      cfg.channels,  cfg.se1.num_layers, drop=cfg.se1.dropout),
            self.residual_se_cnn_block(cfg.channels, cfg.feat_dim,  cfg.se2.num_layers, drop=cfg.se2.dropout),
        )
        # CNN側のpoolサイズ列（上の2ブロックで pool=2 を想定）
        self._cnn_pool_sizes: Sequence[int] = (2, 2)

        # --- BERT encoder (embedding無しで使う) ---
        bert_cfg = BertConfig(
            hidden_size=cfg.feat_dim,
            num_hidden_layers=cfg.bert_layers,
            num_attention_heads=cfg.bert_heads,
            intermediate_size=cfg.feat_dim * 4,
            max_position_embeddings=getattr(cfg, "max_position_embeddings", 1024),  # 必要なら増やす
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.feat_dim))
        self.bert = BertModel(bert_cfg)

        # --- classifier head ---
        self.classifier = nn.Sequential(
            nn.Linear(cfg.feat_dim, cfg.cls1.channels, bias=False),
            nn.LayerNorm(cfg.cls1.channels),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.cls1.dropout),
            nn.Linear(cfg.cls1.channels, cfg.cls2.channels, bias=False),
            nn.LayerNorm(cfg.cls2.channels),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.cls2.dropout),
            nn.Linear(cfg.cls2.channels, cfg.num_classes),
        )


    def residual_se_cnn_block(self, in_channels, out_channels, num_layers, pool_size=2, drop=0.3, wd=1e-4):
        # ※ ResNetSEBlock は既存実装を利用
        return nn.Sequential(
            *[ResNetSEBlock(in_channels=in_channels, out_channels=in_channels) for _ in range(num_layers)],
            ResNetSEBlock(in_channels, out_channels, wd=wd),
            nn.MaxPool1d(pool_size, ceil_mode=True),   # 時間長を/2
            nn.Dropout(drop),
        )

    @staticmethod
    def _down_len(lengths: torch.Tensor, pool_sizes: Sequence[int], ceil: bool = False) -> torch.Tensor:
        out = lengths.clone()
        if ceil:
            for p in pool_sizes:
                out = torch.div(out + (p - 1), p, rounding_mode="floor")  # ← ceil相当
        else:
            for p in pool_sizes:
                out = torch.div(out, p, rounding_mode="floor")
        return out.clamp_min(1)

    @staticmethod
    def _downsample_mask(mask: torch.Tensor, pool_sizes: Sequence[int], ceil: bool = False) -> torch.Tensor:
        x = mask.float().unsqueeze(1)  # [B,1,T]
        for p in pool_sizes:
            x = F.max_pool1d(x, kernel_size=p, stride=p, ceil_mode=ceil)
        return x.squeeze(1).to(torch.bool)  # [B,T′]

    def forward(
        self,
        x: torch.Tensor,                       # [B,T,C]
        lengths: Optional[torch.Tensor] = None,# [B] 実長
        mask: Optional[torch.Tensor] = None,   # [B,T] True=有効
    ):
        B, T, C = x.shape

        # --- 入力maskを用意 ---
        if mask is None:
            if lengths is not None:
                ar = torch.arange(T, device=x.device)[None, :].expand(B, T)
                mask = ar < lengths[:, None]         # [B,T] True=有効
            else:
                mask = torch.ones(B, T, dtype=torch.bool, device=x.device)

        # --- CNN（Conv1dは [N,C_in,L] 前提） ---
        feat = self.imu_branch(x.permute(0, 2, 1))     # [B, feat_dim, T']
        feat = feat.permute(0, 2, 1).contiguous()      # [B, T', feat_dim]
        Tp   = feat.size(1)

        # --- マスク／長さもCNNに合わせて縮める ---
        mask_cnn = self._downsample_mask(mask, self._cnn_pool_sizes, ceil=True)    # [B,T']
        if lengths is not None:
            lengths_cnn = self._down_len(lengths, self._cnn_pool_sizes) # [B]
        else:
            lengths_cnn = mask_cnn.sum(dim=1)

        # --- BERT入力を組み立て（CLS付与 & attention_mask） ---
        cls = self.cls_token.expand(B, 1, feat.size(-1))                 # [B,1,H]
        bert_in = torch.cat([cls, feat], dim=1)                          # [B, T'+1, H]

        # attention_mask: 1=有効, 0=Pad
        attn_mask = torch.zeros(B, Tp + 1, dtype=torch.long, device=x.device)
        attn_mask[:, 0] = 1                                              # CLSは常に有効
        attn_mask[:, 1:] = mask_cnn.long()

        # --- BERT（inputs_embedsで通す）---
        outputs = self.bert(inputs_embeds=bert_in, attention_mask=attn_mask)
        cls_vec = outputs.last_hidden_state[:, 0, :]                     # [B,H]

        # （元の実装に合わせてLayerNormを軽く）
        cls_vec = F.layer_norm(cls_vec, cls_vec.shape[1:])
        logits = self.classifier(cls_vec)                                 # [B,num_classes]
        return logits

class litmodel(L.LightningModule):
    def __init__(
        self,
        cfg,
        lr_init: float = 5e-4,
        min_lr: float = 2e-5,
        weight_decay: float = 3e-3,
        cls_loss_weight: float = 1.0,
        class_weight: Optional[torch.Tensor] = None,  # CE のクラス重み
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg", "class_weight"])
        self.cfg = cfg

        # --- core model -----------------------------------------------------
        # あなたの実装名に合わせて変更してください（Bert_model ならそちらに）
        self.model = Bert_model(cfg)

        # --- metrics --------------------------------------------------------
        self.train_acc = MulticlassAccuracy(num_classes=cfg.num_classes, average="macro")
        self.val_acc   = MulticlassAccuracy(num_classes=cfg.num_classes, average="macro")

        # --- class weight (CE) ---------------------------------------------
        if class_weight is not None:
            self.register_buffer("class_weight", torch.as_tensor(class_weight, dtype=torch.float32))
        else:
            self.class_weight = None  # buffer未登録

    # --------------------------------------------------------------------- #
    # モデルforward（可変長＆mask対応）
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor]=None, mask: Optional[torch.Tensor]=None):
        return self.model(x, lengths=lengths, mask=mask)

    def _ce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        target: [B] (int) or [B,C] (one-hot/prob)
        class_weight は hard CE のときのみ適用
        """
        if target.ndim == 2 and target.dtype != torch.long:
            logp = F.log_softmax(logits, dim=1)
            return -(target * logp).sum(dim=1).mean()
        hard = target if target.ndim == 1 else target.argmax(dim=1)
        ce = nn.CrossEntropyLoss(weight=self.class_weight)
        return ce(logits, hard)

    # ---- 汎用: accuracy（target の形式に自動対応） -------------------------
    def _acc(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        hard = target if target.ndim == 1 else target.argmax(dim=1)
        return self.train_acc(preds, hard) if self.training else self.val_acc(preds, hard)

    # ------------------------------------------------------------
    def training_step(self, batch: Tuple, batch_idx: int):
        """
        mixup_pad_collate_fn(return_soft=False): batch=(x, lengths, mask, y_a, y_b, lam)
        mixup_pad_collate_fn(return_soft=True):  batch=(x, lengths, mask, y_mix)
        collate_pad:                              batch=(x, lengths, mask, y)
        """
        if len(batch) == 6:
            x, lengths, mask, y_a, y_b, lam = batch
            logits = self.forward(x, lengths, mask)
            # lam は float 想定（バッチ全体同値）
            loss = self.hparams.cls_loss_weight * (
                lam * self._ce(logits, y_a) + (1.0 - lam) * self._ce(logits, y_b)
            )
            preds = logits.argmax(dim=1)
            # 精度は期待値で合成（報告用）
            acc = lam * self._acc(preds, y_a) + (1.0 - lam) * self._acc(preds, y_b)

        elif len(batch) == 4:
            x, lengths, mask, y = batch
            logits = self.forward(x, lengths, mask)
            loss  = self.hparams.cls_loss_weight * self._ce(logits, y)
            preds = logits.argmax(dim=1)
            acc   = self._acc(preds, y)

        else:
            raise RuntimeError(f"Unexpected train batch format: len={len(batch)}")

        # ログ（学習率も出す）
        log_dict = {"train/loss": loss, "train/acc": acc}
        opt = self.optimizers()
        if opt is not None:
            log_dict["lr"] = opt.param_groups[0]["lr"]
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # ------------------------------------------------------------
    def validation_step(self, batch: Tuple, batch_idx: int):
        if len(batch) == 6:
            x, lengths, mask, y, _, _ = batch
        elif len(batch) == 4:
            x, lengths, mask, y = batch
        else:
            raise RuntimeError(f"Unexpected val batch format: len={len(batch)}")

        logits = self.forward(x, lengths, mask)
        loss   = self.hparams.cls_loss_weight * self._ce(logits, y)
        preds  = logits.argmax(dim=1)
        acc    = self._acc(preds, y)
        self.log_dict({"val/loss": loss, "val/acc": acc}, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple, batch_idx: int):
        if len(batch) == 6:
            x, lengths, mask, y, _, _ = batch
        elif len(batch) == 4:
            x, lengths, mask, y = batch
        else:
            raise RuntimeError(f"Unexpected test batch format: len={len(batch)}")
        logits = self.forward(x, lengths, mask)
        loss   = self.hparams.cls_loss_weight * self._ce(logits, y)
        self.log_dict({"test/loss": loss}, on_step=False, on_epoch=True)

    # --------------------------------------------------------------------- #
    # Optimizer & Cosine Scheduler (with Warmup)
    def configure_optimizers(self):
        # 1) WDの適用先を分ける（Norm/BN/biasはWD=0を推奨）
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or n.endswith(".bias") or "norm" in n.lower() or "bn" in n.lower():
                no_decay.append(p)
            else:
                decay.append(p)

        opt = torch.optim.AdamW(
            [
                {"params": decay,    "weight_decay": self.hparams.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.hparams.lr_init, betas=(0.9, 0.999), eps=1e-8
        )

        # 2) 総ステップとウォームアップ「ステップ」を厳密化
        total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        if total_steps is None:
            steps_per_epoch = self.trainer.num_training_batches
            total_steps = steps_per_epoch * self.trainer.max_epochs

        steps_per_epoch = self.trainer.num_training_batches
        warmup_steps = int(getattr(self.cfg.train, "warmup_epochs", 0) * max(1, steps_per_epoch))  # epoch→step
        warmup_steps = min(warmup_steps, max(0, total_steps - 1))
        cosine_steps = max(1, total_steps - warmup_steps)

        # 3) Linear warmup → Cosine へ
        start_factor = max(self.hparams.min_lr / max(self.hparams.lr_init, 1e-12), 0.0)
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=start_factor, end_factor=1.0, total_iters=warmup_steps
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cosine_steps, eta_min=self.hparams.min_lr
        )
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step", 
                            "frequency": 1, "name": "linear+cosine"},
        }

    
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

class GestureDataModule(L.LightningDataModule):
    """
    BERT用IMUデータモジュール（可変長 & attention_mask 前提）
    - scaler: IMU列のみでfit→ export_dir/scaler_imu.pkl
    - IMU列一覧: export_dir/imu_feature_cols.npy
    - gesture classes: export_dir/gesture_classes.npy
    """
    def __init__(self, cfg, fold_idx: int):
        super().__init__()
        self.cfg       = cfg
        self.fold_idx  = fold_idx
        self.n_splits  = cfg.data.n_splits

        self.raw_dir    = Path(cfg.data.raw_dir)
        self.export_dir = (HydraConfig.get().runtime.output_dir / pathlib.Path(cfg.data.export_dir))
        self.batch      = cfg.train.batch_size
        self.batch_val  = cfg.train.batch_size_val
        self.mixup_a    = cfg.train.mixup_alpha

        self.num_classes = None
        self.imu_ch      = None
        self.class_weight = None
        self.steps_per_epoch = None

    # ---------------------------
    # 前処理資材の作成（IMUのみ）
    # ---------------------------
    def prepare_data(self):
        df = pd.read_csv(self.raw_dir / "train.csv")
        df = feature_eng(df)  # ここでIMUエンジニアド特徴も作られている想定
        df["gesture_int"] = df["gesture"].apply(labeling)
        self.export_dir.mkdir(parents=True, exist_ok=True)

        # ラベルクラス保存
        np.save(self.export_dir/"gesture_classes.npy", labeling("classes"))

        # 特徴列定義
        meta = {'gesture','gesture_int','sequence_type','behavior','orientation',
                'row_id','subject','phase','sequence_id','sequence_counter'}
        feat_cols = [c for c in df.columns if c not in meta]
        # IMUのみ（thm_/tof_ を除外）
        imu_cols = [c for c in feat_cols if not (c.startswith("thm_") or c.startswith("tof_"))]
        

        # 保存：IMU列
        np.save(self.export_dir/"feature_cols.npy", np.array(imu_cols, dtype=object))

        # StandardScaler（IMU列のみでfit）
        scaler_imu = StandardScaler().fit(
            df[imu_cols].ffill().bfill().fillna(0).values
        )
        joblib.dump(scaler_imu, self.export_dir/"scaler.pkl")

    # ---------------------------
    # 分割・Dataset作成
    # ---------------------------
    def setup(self, stage="fit"):
        df = pd.read_csv(self.raw_dir / "train.csv")
        df = feature_eng(df)
        df["gesture_int"] = df["gesture"].apply(labeling)
        self.num_classes  = len(labeling("classes"))

        # 列
        saved_imu = self.export_dir / "feature_cols.npy"
        imu_cols = np.load(saved_imu, allow_pickle=True).tolist()
        self.dim = len(imu_cols)

        # スケーラ（IMUのみ）
        scaler = joblib.load(self.export_dir/"scaler.pkl")

        # ---- 全シーケンスを「可変長のまま」作る（IMUのみ） ----
        X_list, y_int, subjects = [], [], []
        for _, seq in df.groupby("sequence_id"):
            x_imu = (
                seq[imu_cols]
                .ffill().bfill().fillna(0)
                .to_numpy(dtype=np.float32)
            )  # [T, C_imu]
            x_imu = scaler.transform(x_imu)           # スケーリング
            X_list.append(torch.from_numpy(x_imu))    # Tensor [T, C_imu]

            y_int.append(int(seq["gesture_int"].iloc[0]))
            subjects.append(seq["subject"].iloc[0])
        y_int = np.array(y_int)

        # ---- StratifiedGroupKFold（subject でグループ）----
        sgkf = StratifiedGroupKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.cfg.seed
        )
        # 必要に応じた層化変換（例: 一部クラスをまとめる）
        y_processed = [(y if y < 8 else -1) for y in y_int] #ここがよくなかったりする？

        tr_idx, val_idx = list(
            sgkf.split(np.arange(len(X_list)), y_processed, np.array(subjects))
        )[self.fold_idx]

        # ---- Dataset（可変長）----
        X_tr  = [X_list[i] for i in tr_idx]
        X_val = [X_list[i] for i in val_idx]
        y_tr  = y_int[tr_idx]
        y_val = y_int[val_idx]

        # Augmentは必要に応じて（BERTでも使うなら後段で）
        self.ds_tr_imu  = SequenceDatasetVarLen(X_tr, y_tr, augmenter=None)
        self.ds_val_imu = SequenceDatasetVarLen(X_val, y_val, augmenter=None)

        # ---- class_weight（fold内）----
        self.class_weight = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(self.num_classes),
            y=y_tr
        )
        self.steps_per_epoch = math.ceil(len(tr_idx) / self.batch)

    def train_dataloader_imu(self):
        """
        返り値（例：utils側の実装想定）
          x:[B,L,C], lengths:[B], mask:[B,L] (True=有効), y あるいは (y_a,y_b,lam)
        """
        return DataLoader(
            self.ds_tr_imu,
            batch_size=self.batch, shuffle=True,
            collate_fn=mixup_pad_collate_fn(self.mixup_a, return_soft=False),
            drop_last=True, num_workers=4, pin_memory=True, persistent_workers=True
        )

    def val_dataloader_imu(self):
        return DataLoader(
            self.ds_val_imu,
            batch_size=self.batch_val, shuffle=False,
            collate_fn=collate_pad,
            num_workers=4, pin_memory=True, persistent_workers=True
        )

    # （必要なら別名でBERT用を生やしてもOK）
    def train_dataloader_bert(self):
        return self.train_dataloader_imu()

    def val_dataloader_bert(self):
        return self.val_dataloader_imu()
