# datamodule_cv.py という新ファイルにすると分かりやすい
import numpy as np, pandas as pd, torch, joblib, pathlib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
import lightning as L
from utils import (preprocess_sequence, SequenceDataset, phase_to_int, 
                   mixup_collate_fn, Macro_eng, Augment, labeling_for_macro,
                     _scan_lengths, GestureSequence, collate_pad)
from sklearn.utils.class_weight import compute_class_weight
from hydra.core.hydra_config import HydraConfig
import math



class GestureDataModule(L.LightningDataModule):
    def __init__(self, cfg, fold_idx: int):
        super().__init__()
        self.cfg = cfg
        self.fold_idx = fold_idx
        self.n_splits = cfg.data.n_splits

        self.raw_dir = Path(cfg.data.raw_dir)
        self.export_dir = (
            HydraConfig.get().runtime.output_dir / pathlib.Path(cfg.data.export_dir)
        )
        self.batch = cfg.train.batch_size
        self.batch_val = cfg.train.batch_size_val
        self.mixup_a = cfg.train.mixup_alpha

    # ---------------------- prepare_data ----------------------
    def prepare_data(self):
        df = pd.read_csv(self.raw_dir / 'train.csv')
        df = df[df['gesture'].isin(labeling_for_macro('classes'))]
        df = df[df['behavior'].isin(phase_to_int('classes'))]
        df['gesture_int'] = df['gesture'].apply(labeling_for_macro)
        df['behavior_int'] = df['behavior'].apply(phase_to_int)

        df = Macro_eng(df)

        self.export_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.export_dir / 'gesture_classes.npy', labeling_for_macro('classes'))

        meta = {
            'gesture',
            'gesture_int',
            'sequence_type',
            'behavior',
            'behavior_int',
            'orientation',
            'row_id',
            'subject',
            'phase',
            'sequence_id',
            'sequence_counter',
        }
        feat_cols = [c for c in df.columns if c not in meta]
        np.save(self.export_dir / 'feature_cols.npy', feat_cols)

        scaler = StandardScaler().fit(
            df[feat_cols].ffill().bfill().fillna(0).values
        )
        joblib.dump(scaler, self.export_dir / 'scaler.pkl')

        len_move, len_perform = _scan_lengths(df)
        np.save(self.export_dir / 'move_maxlen.npy', len_move)
        np.save(self.export_dir / 'perform_maxlen.npy', len_perform)

    # ---------------------- setup ----------------------
    def setup(self, stage: str = 'fit'):
        df = pd.read_csv(self.raw_dir / 'train.csv')
        df = df[df['gesture'].isin(labeling_for_macro('classes'))]
        df = df[df['behavior'].isin(phase_to_int('classes'))]
        df['gesture_int'] = df['gesture'].apply(labeling_for_macro)
        df['behavior_int'] = df['behavior'].apply(phase_to_int)
        df = Macro_eng(df)

        self.num_classes = len(labeling_for_macro('classes'))

        meta = {
            'gesture',
            'gesture_int',
            'sequence_type',
            'behavior',
            'behavior_int',
            'orientation',
            'row_id',
            'subject',
            'phase',
            'sequence_id',
            'sequence_counter',
        }
        self.feat_cols = [c for c in df.columns if c not in meta]

        imu_cols = [
            c
            for c in self.feat_cols
            if not (c.startswith('thm_') or c.startswith('tof_'))
        ]
        # imu channels + last phase channel
        self.imu_ch = len(imu_cols)
        self.tof_ch = len(self.feat_cols) - self.imu_ch

        self.len_move = int(np.load(self.export_dir / 'move_maxlen.npy'))
        self.len_perform = int(np.load(self.export_dir / 'perform_maxlen.npy'))

        scaler = joblib.load(self.export_dir / 'scaler.pkl')

        # ---- sequence‑level stratified split ----
        seq_df = (
            df.groupby('sequence_id', sort=False).first().reset_index()[['sequence_id', 'gesture_int']]
        )
        y_seq = seq_df['gesture_int'].values
        seq_ids = seq_df['sequence_id'].values

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.cfg.data.random_seed,
        )
        seq_tr_idx, seq_val_idx = list(skf.split(seq_ids, y_seq))[self.fold_idx]

        tr_seq_ids = set(seq_ids[seq_tr_idx])
        val_seq_ids = set(seq_ids[seq_val_idx])

        df_tr = df[df['sequence_id'].isin(tr_seq_ids)]
        df_val = df[df['sequence_id'].isin(val_seq_ids)]

        imu_idx = [self.feat_cols.index(c) for c in imu_cols]
        imu_idx.append(len(self.feat_cols))  # phase channel

        self.ds_tr_imu = GestureSequence(
            df_tr,
            scaler,
            self.feat_cols,
            imu_idx,
            self.len_move,
            self.len_perform
        )
        self.ds_val_imu = GestureSequence(
            df_val,
            scaler,
            self.feat_cols,
            imu_idx,
            self.len_move,
            self.len_perform
        )

        # class weight
        self.class_weight = compute_class_weight(
            class_weight='balanced',
            classes=np.arange(self.num_classes),
            y=y_seq[seq_tr_idx],
        )

        self.steps_per_epoch = math.ceil(len(self.ds_tr_imu) / self.batch)

    # ---------------------- DataLoaders ----------------------
    def train_dataloader_imu(self):
        collate_fn = collate_pad
        return DataLoader(
            self.ds_tr_imu,
            batch_size=self.batch,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader_imu(self):
        collate_fn = collate_pad
        return DataLoader(
            self.ds_val_imu,
            batch_size=self.batch_val,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )