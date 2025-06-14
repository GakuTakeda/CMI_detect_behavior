# datamodule_cv.py という新ファイルにすると分かりやすい
import numpy as np, pandas as pd, torch, joblib, pathlib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
import lightning as L
from utils import preprocess_sequence, SequenceDataset, mixup_collate_fn
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
        self.export_dir = (HydraConfig.get().runtime.output_dir
                           / pathlib.Path(cfg.data.export_dir))
        self.batch      = cfg.train.batch_size
        self.batch_val  = cfg.train.batch_size_val
        self.mixup_a    = cfg.train.mixup_alpha

    # prepare_data は最初の fold だけ呼ばれれば OK
    def prepare_data(self):
        df = pd.read_csv(self.raw_dir / "train.csv")
        le = LabelEncoder(); df["gesture_int"] = le.fit_transform(df["gesture"])
        self.export_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.export_dir/"gesture_classes.npy", le.classes_)

        meta = {'gesture','gesture_int','sequence_type','behavior','orientation',
                'row_id','subject','phase','sequence_id','sequence_counter'}
        feat_cols = [c for c in df.columns if c not in meta]
        np.save(self.export_dir/"feature_cols.npy", feat_cols)

        scaler = StandardScaler().fit(
            df[feat_cols].ffill().bfill().fillna(0).values
        )
        joblib.dump(scaler, self.export_dir/"scaler.pkl")

    # fold ごとに train / val を切り分け
    def setup(self, stage="fit"):
        df = pd.read_csv(self.raw_dir / "train.csv")
        le = LabelEncoder(); df["gesture_int"] = le.fit_transform(df["gesture"])
        self.num_classes = len(le.classes_)

        meta = {'gesture','gesture_int','sequence_type','behavior','orientation',
                'row_id','subject','phase','sequence_id','sequence_counter'}
        self.feat_cols = [c for c in df.columns if c not in meta]

        imu_cols = [c for c in self.feat_cols
                    if not (c.startswith("thm_") or c.startswith("tof_"))]
        self.imu_ch = len(imu_cols)
        self.tof_ch = len(self.feat_cols) - self.imu_ch

        scaler = joblib.load(self.export_dir/"scaler.pkl")

        # ---- 全シーケンスをテンソル化 ----
        X_l, y_l, lens = [], [], []
        for _, seq in df.groupby("sequence_id"):
            X_l.append(preprocess_sequence(seq, self.feat_cols, scaler))
            y_l.append(seq["gesture_int"].iloc[0])
            lens.append(len(X_l[-1]))

        pad_len = int(np.percentile(lens, self.cfg.data.pad_percentile))
        np.save(self.export_dir/"sequence_maxlen.npy", pad_len)
        self.pad_len = pad_len

        X_pad = np.zeros((len(X_l), pad_len, len(self.feat_cols)),
                         dtype="float32")
        for i, m in enumerate(X_l):
            X_pad[i, :min(len(m), pad_len)] = m[:pad_len]
        y_int = np.array(y_l)
        y_oh  = np.eye(self.num_classes)[y_int].astype("float32")

        # ---- StratifiedKFold で index を取得 ----
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                              random_state=self.cfg.data.random_seed)
        tr_idx, val_idx = list(skf.split(X_pad, y_int))[self.fold_idx]

        X_tr, X_val = X_pad[tr_idx], X_pad[val_idx]
        y_tr, y_val = y_oh [tr_idx], y_oh [val_idx]

        self.ds_tr  = SequenceDataset(X_tr , y_tr )
        self.ds_val = SequenceDataset(X_val, y_val)

        imu_idx = [self.feat_cols.index(c) for c in imu_cols]   # IMU 列のインデックス
        X_pad_imu = X_pad[..., imu_idx]                         # shape: (N, pad_len, imu_ch)

        self.ds_tr_imu  = SequenceDataset(X_pad_imu[tr_idx], y_tr)
        self.ds_val_imu = SequenceDataset(X_pad_imu[val_idx], y_val)

        # ---- fold 内の class_weight ----
        self.class_weight = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(self.num_classes),
            y=y_int[tr_idx]
        )
        self.steps_per_epoch = math.ceil(len(tr_idx) / self.batch)

    # ---------- DataLoaders ----------
    def train_dataloader(self):
        return DataLoader(self.ds_tr,  batch_size=self.batch, shuffle=True,
                          collate_fn=mixup_collate_fn(self.mixup_a),
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_val,
                          collate_fn=mixup_collate_fn(0.0))

    def train_dataloader_imu(self):
        return DataLoader(self.ds_tr_imu, batch_size=self.batch, shuffle=True,
                          collate_fn=mixup_collate_fn(self.mixup_a),
                          drop_last=True)

    def val_dataloader_imu(self):
        return DataLoader(self.ds_val_imu, batch_size=self.batch_val,
                          collate_fn=mixup_collate_fn(0.0))