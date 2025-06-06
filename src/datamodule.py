import numpy as np, pandas as pd, torch, joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import lightning as L
from utils import preprocess_sequence, SequenceDataset, mixup_collate_fn
from sklearn.utils.class_weight import compute_class_weight

class GestureDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.raw_dir   = Path(cfg.data.raw_dir)
        self.export_dir= Path(cfg.data.export_dir)
        self.batch     = cfg.train.batch_size
        self.batch_val = cfg.train.batch_size_val
        self.mixup_a   = cfg.train.mixup_alpha

    def prepare_data(self):
        # 一度だけ実行：scaler・npys を保存
        df = pd.read_csv(self.raw_dir / "train.csv")
        le = LabelEncoder(); df["gesture_int"] = le.fit_transform(df["gesture"])
        self.export_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.export_dir/"gesture_classes.npy", le.classes_)

        meta = {'gesture','gesture_int','sequence_type','behavior','orientation',
                'row_id','subject','phase','sequence_id','sequence_counter'}
        feat_cols = [c for c in df.columns if c not in meta]
        scaler = StandardScaler().fit(df[feat_cols].ffill().bfill().fillna(0).values)
        joblib.dump(scaler, self.export_dir/"scaler.pkl")

    def setup(self, stage="fit"):
        df  = pd.read_csv(self.raw_dir / "train.csv")
        le  = LabelEncoder(); df["gesture_int"] = le.fit_transform(df["gesture"])
        self.num_classes = len(le.classes_)

        meta = {'gesture','gesture_int','sequence_type','behavior','orientation',
                'row_id','subject','phase','sequence_id','sequence_counter'}
        self.feat_cols = [c for c in df.columns if c not in meta]

        imu_cols = [c for c in self.feat_cols if not (c.startswith("thm_") or c.startswith("tof_"))]
        self.imu_ch = len(imu_cols)
        self.tof_ch = len(self.feat_cols) - self.imu_ch

        scaler = joblib.load(self.export_dir/"scaler.pkl")

        # sequence → tensor list
        X_l, y_l, lens = [], [], []
        for _, seq in df.groupby("sequence_id"):
            X_l.append(preprocess_sequence(seq, self.feat_cols, scaler))
            y_l.append(seq["gesture_int"].iloc[0])   # ← 整数ラベル
            lens.append(len(X_l[-1]))

        pad_len = int(np.percentile(lens, self.cfg.data.pad_percentile))
        self.pad_len = pad_len
        X_pad = np.zeros((len(X_l), pad_len, len(self.feat_cols)), dtype="float32")
        for i, m in enumerate(X_l):  X_pad[i, :min(len(m), pad_len)] = m[:pad_len]
        y_oh  = np.eye(self.num_classes)[y_l].astype("float32")

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_pad, y_oh, test_size=self.cfg.data.val_ratio,
            stratify=y_l, random_state=self.cfg.data.random_seed
        )
        self.ds_tr  = SequenceDataset(X_tr, y_tr)
        self.ds_val = SequenceDataset(X_val, y_val)

        # === ① 整数ラベルを保持 ===
        self.y_int = np.array(y_l)

        # === ② class_weight を計算して保持 ===
        self.class_weight = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(self.num_classes),
            y=self.y_int
        )

    # --- DataLoaders ---
    def train_dataloader(self):
        return DataLoader(self.ds_tr, batch_size=self.batch, shuffle=True,
                          collate_fn=mixup_collate_fn(self.mixup_a), drop_last=True)
    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_val,
                          collate_fn=mixup_collate_fn(0.0))