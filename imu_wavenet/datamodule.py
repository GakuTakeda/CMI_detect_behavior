# datamodule_cv.py という新ファイルにすると分かりやすい
import numpy as np, pandas as pd, torch, joblib, pathlib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import Dataset, DataLoader
import lightning as L
from utils import (preprocess_sequence, SequenceDatasetVarLen, 
                   mixup_pad_collate_fn, collate_pad, feature_eng, AugmentIMUOnly, labeling, labeling_for_macro)
from sklearn.utils.class_weight import compute_class_weight
from hydra.core.hydra_config import HydraConfig
import math



class GestureDataModule(L.LightningDataModule):
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

    def prepare_data(self):
        df = pd.read_csv(self.raw_dir / "train.csv")
        if self.cfg.train.mode == "8_class":
            df = df[df["gesture"].isin(labeling_for_macro("classes"))].reset_index(drop=True)
            df["gesture_int"] = df["gesture"].apply(labeling_for_macro)
        else:
            df["gesture_int"] = df["gesture"].apply(labeling)
        df = feature_eng(df)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.export_dir/"gesture_classes.npy", labeling("classes"))

        meta = {'gesture','gesture_int','sequence_type','behavior','orientation',
                'row_id','subject','phase','sequence_id','sequence_counter'}
        feat_cols = [c for c in df.columns if c not in meta]
        np.save(self.export_dir/"feature_cols.npy", feat_cols)

        scaler = StandardScaler().fit(
            df[feat_cols].ffill().bfill().fillna(0).values
        )
        joblib.dump(scaler, self.export_dir/"scaler.pkl")

    def setup(self, stage="fit"):
        df = pd.read_csv(self.raw_dir / "train.csv")
        df = feature_eng(df)

        meta = {'gesture','gesture_int','sequence_type','behavior','orientation',
                'row_id','subject','phase','sequence_id','sequence_counter'}
        if self.cfg.train.mode == "8_class":
            df = df[df["gesture"].isin(labeling_for_macro("classes"))].reset_index(drop=True)
            df["gesture_int"] = df["gesture"].apply(labeling_for_macro)
            self.num_classes  = len(labeling_for_macro("classes"))
        else:
            df["gesture_int"] = df["gesture"].apply(labeling)
            self.num_classes  = len(labeling("classes"))

        feat_cols = [c for c in df.columns if c not in meta]
        self.feat_cols = feat_cols

        # IMUカラムのみ選択
        imu_cols = [c for c in feat_cols if not (c.startswith("thm_") or c.startswith("tof_"))]
        self.imu_ch = len(imu_cols)
        imu_idx = [feat_cols.index(c) for c in imu_cols]

        scaler = joblib.load(self.export_dir/"scaler.pkl")

        # ---- 全シーケンスを「可変長のまま」作る ----
        X_list, y_int = [], []  # ← subjects を追加
        for _, seq in df.groupby("sequence_id"):
            x_all = preprocess_sequence(seq, feat_cols, scaler)  # [T, F]
            x_imu = x_all[:, imu_idx].to(torch.float32)         # [T, C_imu]
            X_list.append(x_imu)
            y_int.append(int(seq["gesture_int"].iloc[0]))

        y_int = np.array(y_int)

        # ---- StratifiedGroupKFold（subject でグループ、y_processed で層化）----
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,                
            random_state=self.cfg.data.random_seed
        )

        # fold_idx 番目を取得
        tr_idx, val_idx = list(
            skf.split(X=np.arange(len(X_list)), y=y_int)
        )[self.fold_idx]


        # ---- Augmenter（任意：IMUのみ）----
        self.augmenter = AugmentIMUOnly(
                    p_time_shift=self.cfg.aug.p_time_shift, max_shift_ratio=self.cfg.aug.max_shift_ratio,
                    p_time_warp=self.cfg.aug.p_time_warp, warp_min=self.cfg.aug.warp_min, warp_max=self.cfg.aug.warp_max,
                    p_block_dropout=self.cfg.aug.p_block_dropout, n_blocks=self.cfg.aug.n_blocks, block_len=self.cfg.aug.block_len,
                    p_imu_jitter=self.cfg.aug.p_imu_jitter, imu_sigma=self.cfg.aug.imu_sigma,
                    p_imu_scale=self.cfg.aug.p_imu_scale, imu_scale_sigma=self.cfg.aug.imu_scale_sigma,
                    p_imu_drift=self.cfg.aug.p_imu_drift, drift_std=self.cfg.aug.drift_std, drift_clip=self.cfg.aug.drift_clip,
                    p_imu_small_rot=self.cfg.aug.p_imu_small_rot,
                    pad_value=self.cfg.aug.pad_value,
                )

        # ---- Dataset（可変長）----
        X_tr = [X_list[i] for i in tr_idx]
        X_val = [X_list[i] for i in val_idx]
        y_tr = y_int[tr_idx]
        y_val = y_int[val_idx]

        if self.cfg.aug.no_aug:
            self.augmenter = None
        self.ds_tr_imu  = SequenceDatasetVarLen(X_tr, y_tr, augmenter=self.augmenter)
        self.ds_val_imu = SequenceDatasetVarLen(X_val, y_val, augmenter=None)

        # ---- class_weight（fold内）----
        self.class_weight = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(self.num_classes),
            y=y_int[tr_idx]
        )
        self.steps_per_epoch = math.ceil(len(tr_idx) / self.batch)

    # ---------- DataLoaders ----------
    def train_dataloader_imu(self):
        # ハードラベルMixUp（推奨）：y_a, y_b, lam を返す
        return DataLoader(
            self.ds_tr_imu,
            batch_size=self.batch, shuffle=True,
            collate_fn=mixup_pad_collate_fn(self.mixup_a, return_soft=False),
            drop_last=True, num_workers=4, pin_memory=True, persistent_workers=True
        )

    def val_dataloader_imu(self):
        # 検証はMixUpなし、Pad+Maskのみ
        return DataLoader(
            self.ds_val_imu,
            batch_size=self.batch_val,
            collate_fn=collate_pad,
            num_workers=4, pin_memory=True, persistent_workers=True
        )
