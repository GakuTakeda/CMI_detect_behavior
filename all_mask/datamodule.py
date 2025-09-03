# datamodule_cv.py という新ファイルにすると分かりやすい
import numpy as np, pandas as pd, torch, joblib, pathlib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, GroupKFold
from torch.utils.data import Dataset, DataLoader
import lightning as L
from utils import (preprocess_sequence, SequenceDatasetVarLen, 
                   mixup_pad_collate_fn, collate_pad, feature_eng, AugmentMultiModal, labeling, labeling_for_macro)
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

        # 後で参照する属性
        self.idx_imu = None
        self.idx_thm = None
        self.idx_tof = None
        self.imu_ch  = None
        self.tof_ch  = None  # (= THM+ToF の合計)

    # ------- helper: モダリティの列インデックスを作る -------
    @staticmethod
    def _build_modality_indices(feature_cols: list[str]):
        idx_imu, idx_thm, idx_tof = [], [], []
        for i, c in enumerate(feature_cols):
            if c.startswith("thm_"):
                idx_thm.append(i)
            elif c.startswith("tof_"):
                idx_tof.append(i)
            else:
                idx_imu.append(i)
        return (
            np.asarray(idx_imu, dtype=np.int32),
            np.asarray(idx_thm, dtype=np.int32),
            np.asarray(idx_tof, dtype=np.int32),
        )

    def prepare_data(self):
        df = pd.read_csv(self.raw_dir / "train.csv")
        if self.cfg.train.mode == "8_class":
            df = df[df["gesture"].isin(labeling_for_macro("classes"))].reset_index(drop=True)
            df["gesture_int"] = df["gesture"].apply(labeling_for_macro)
            classes = labeling_for_macro("classes")
        else:
            df["gesture_int"] = df["gesture"].apply(labeling)
            classes = labeling("classes")

        # IMU派生含む追加特徴
        df = feature_eng(df)

        # 特徴列（メタ除外）
        meta = {'gesture','gesture_int','sequence_type','behavior','orientation',
                'row_id','subject','phase','sequence_id','sequence_counter'}
        feat_cols = [c for c in df.columns if c not in meta]

        # 保存
        self.export_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.export_dir/"gesture_classes.npy", classes)
        np.save(self.export_dir/"feature_cols.npy", feat_cols)

        # スケーラは全特徴でfit（IMUだけでなくTHM/ToFも含める）
        scaler = StandardScaler().fit(
            df[feat_cols].ffill().bfill().fillna(0).values
        )
        joblib.dump(scaler, self.export_dir/"scaler.pkl")

        # モダリティの列インデックスも保存（再現性確保）
        idx_imu, idx_thm, idx_tof = self._build_modality_indices(feat_cols)
        np.savez(self.export_dir/"modal_indices.npz",
                 idx_imu=idx_imu, idx_thm=idx_thm, idx_tof=idx_tof)

    def setup(self, stage="fit"):
        df = pd.read_csv(self.raw_dir / "train.csv")
        df = feature_eng(df)

        if self.cfg.train.mode == "8_class":
            df = df[df["gesture"].isin(labeling_for_macro("classes"))].reset_index(drop=True)
            df["gesture_int"] = df["gesture"].apply(labeling_for_macro)
            self.num_classes  = len(labeling_for_macro("classes"))
        else:
            df["gesture_int"] = df["gesture"].apply(labeling)
            self.num_classes  = len(labeling("classes"))

        # 特徴列は prepare_data と同一手順で（順序ずれ防止のため、保存済みがあればそれを優先）
        meta = {'gesture','gesture_int','sequence_type','behavior','orientation',
                'row_id','subject','phase','sequence_id','sequence_counter'}
        feat_cols_current = [c for c in df.columns if c not in meta]
        saved_feat_cols_path = self.export_dir/"feature_cols.npy"
        if saved_feat_cols_path.exists():
            feat_cols = list(np.load(saved_feat_cols_path, allow_pickle=True))
        else:
            feat_cols = feat_cols_current
        self.feat_cols = feat_cols

        # モダリティの列インデックス（保存済み優先）
        modal_npz = self.export_dir/"modal_indices.npz"
        if modal_npz.exists():
            z = np.load(modal_npz)
            idx_imu = z["idx_imu"]; idx_thm = z["idx_thm"]; idx_tof = z["idx_tof"]
        else:
            idx_imu, idx_thm, idx_tof = self._build_modality_indices(feat_cols)
        self.idx_imu = idx_imu
        self.idx_thm = idx_thm
        self.idx_tof = idx_tof
        self.imu_ch  = int(len(idx_imu))
        self.tof_ch  = int(len(idx_thm) + len(idx_tof))

        scaler = joblib.load(self.export_dir/"scaler.pkl")

        # ---- 全シーケンス（可変長）を作る：★全特徴で作成 ★
        X_list, y_int, lengths, subjects = [], [], [], []
        for _, seq in df.groupby("sequence_id"):
            x_all = preprocess_sequence(seq, self.feat_cols, scaler)  # [T, F_all]（IMU+THM+ToF）
            X_list.append(x_all)                                      # ← もうIMUだけに切らない
            y_int.append(int(seq["gesture_int"].iloc[0]))
            lengths.append(x_all.shape[0])
            subjects.append(seq["subject"].iloc[0])
        y_int = np.array(y_int)

        # ---- StratifiedKFold（元実装に合わせる）----
        skf = StratifiedGroupKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=0
        )
        tr_idx, val_idx = list(skf.split(X=np.arange(len(X_list)), y=y_int, groups=subjects))[self.fold_idx]

        # ---- Augmenter（マルチモーダル）----
        if self.cfg.aug.no_aug:
            augmenter = None
        else:
            # cfg.aug から安全に取得（未設定はデフォルト）
            aug = self.cfg.aug
            def get(name, default): return getattr(aug, name, default)
            augmenter = AugmentMultiModal(
                idx_imu=self.idx_imu, idx_thm=self.idx_thm, idx_tof=self.idx_tof,
                # 時間操作（全体）
                p_time_shift=get("p_time_shift", 0.7), max_shift_ratio=get("max_shift_ratio", 0.1),
                p_time_warp=get("p_time_warp", 0.5),  warp_min=get("warp_min", 0.9), warp_max=get("warp_max", 1.1),
                p_block_dropout=get("p_block_dropout", 0.5),
                n_blocks=tuple(get("n_blocks", [1,3])), block_len=tuple(get("block_len", [2,6])),
                pad_value=get("pad_value", 0.0),
                # IMU
                p_imu_jitter=get("p_imu_jitter", 0.9), imu_sigma=get("imu_sigma", 0.03),
                p_imu_scale=get("p_imu_scale", 0.5),   imu_scale_sigma=get("imu_scale_sigma", 0.03),
                p_imu_drift=get("p_imu_drift", 0.5),   drift_std=get("drift_std", 0.003), drift_clip=get("drift_clip", 0.3),
                p_imu_small_rot=get("p_imu_small_rot", 0.0), rot_deg=get("rot_deg", 5.0),
                # THM/ToF
                p_thm_ch_drop=get("p_thm_ch_drop", 0.2), thm_drop_frac=get("thm_drop_frac", 0.05), thm_sigma=get("thm_sigma", 0.01),
                p_tof_ch_drop=get("p_tof_ch_drop", 0.2), tof_drop_frac=get("tof_drop_frac", 0.05), tof_sigma=get("tof_sigma", 0.01),
            )

        # ---- Dataset（可変長、全特徴）----
        X_tr  = [X_list[i] for i in tr_idx]
        X_val = [X_list[i] for i in val_idx]
        y_tr  = y_int[tr_idx]
        y_val = y_int[val_idx]

        self.ds_tr  = SequenceDatasetVarLen(X_tr,  y_tr,  augmenter=augmenter)
        self.ds_val = SequenceDatasetVarLen(X_val, y_val, augmenter=None)

        # ---- class_weight（fold内）----
        self.class_weight = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(self.num_classes),
            y=y_int[tr_idx]
        )
        self.steps_per_epoch = math.ceil(len(tr_idx) / self.batch)

    # ---------- DataLoaders ----------
    # 互換のため旧名も維持
    def train_dataloader_imu(self):
        return DataLoader(
            self.ds_tr,
            batch_size=self.batch, shuffle=True,
            collate_fn=mixup_pad_collate_fn(self.mixup_a, return_soft=False),
            drop_last=True, num_workers=4, pin_memory=True, persistent_workers=True
        )

    def val_dataloader_imu(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_val,
            collate_fn=collate_pad,
            num_workers=4, pin_memory=True, persistent_workers=True
        )

    # Lightning 標準名も用意（どちらでも使える）
    def train_dataloader(self):
        return self.train_dataloader_imu()

    def val_dataloader(self):
        return self.val_dataloader_imu()
