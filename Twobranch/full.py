# cv_main.py
import hydra, lightning as L
from omegaconf import DictConfig
from utils     import LitModel, calc_f1, feature_eng, labeling, _pad, SequenceDataset, mixup_collate_fn, seed_everything
import json
import os
import torch
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
import math
import pathlib, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def run(cfg: DictConfig):
    seed_everything(42)
    n_splits = cfg.data.n_splits        # 例: 5
    caluculate = calc_f1()
    export_dir = (HydraConfig.get().runtime.output_dir/ pathlib.Path(cfg.data.export_dir))
    export_dir.mkdir(parents=True, exist_ok=True)
    pad_len = 127
    imu_ch = 13
    tof_ch = 25
    num_classes = 18

    #scaler.fit
    df = pd.read_csv("../data/train.csv")
    df = feature_eng(df)
    df["gesture_int"] = df["gesture"].apply(labeling)
    np.save(export_dir/"gesture_classes.npy", labeling("classes")) 

    meta = {'gesture','gesture_int','sequence_type','behavior','orientation',
                'row_id','subject','phase','sequence_id','sequence_counter'}
    feat_cols = ['linear_acc_x', 'linear_acc_y', 'linear_acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z', 
                 'linear_acc_mag', 'linear_acc_mag_jerk', 'angular_vel_x', 'angular_vel_y', 'angular_vel_z', 'angular_distance', 
                 'thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5', 'tof_1_mean', 'tof_1_std', 'tof_1_min', 'tof_1_max', 'tof_2_mean', 'tof_2_std', 'tof_2_min', 'tof_2_max', 'tof_3_mean', 'tof_3_std', 'tof_3_min', 'tof_3_max', 'tof_4_mean', 'tof_4_std', 'tof_4_min', 'tof_4_max', 'tof_5_mean', 'tof_5_std', 'tof_5_min', 'tof_5_max']

    scaler = StandardScaler().fit(
            df[feat_cols].ffill().bfill().fillna(0).values
        )
    joblib.dump(scaler, export_dir / "scaler.pkl")

    df[feat_cols] = scaler.transform(df[feat_cols].ffill().bfill().fillna(0).values)
    imu_cols = [c for c in feat_cols
                    if not (c.startswith("thm_") or c.startswith("tof_"))]
    imu_idx = [feat_cols.index(c) for c in imu_cols]

    X = []
    labels = []
    for _, seq in df.groupby("sequence_id"):
        X.append(_pad(seq[feat_cols], pad_len))
        labels.append(seq["gesture_int"].iloc[0])

    skf = StratifiedKFold(n_splits=5, shuffle=True,
                              random_state=42)
    
    X = np.array(X)
    labels = np.array(labels)
    y_oh = np.eye(num_classes)[labels]
    
    dataset = []
    val_dataset = []
    imu_dataset = []
    imu_val_dataset = []

    for tr_idx, val_idx in skf.split(X, labels):
        dataset.append(SequenceDataset(X[tr_idx], y_oh[tr_idx]))
        val_dataset.append(SequenceDataset(X[val_idx], y_oh[val_idx]))
        imu_dataset.append(SequenceDataset(X[tr_idx][...,imu_idx], y_oh[tr_idx]))
        imu_val_dataset.append(SequenceDataset(X[val_idx][...,imu_idx], y_oh[val_idx]))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, labels)):
        print(f"\n===== Fold {fold} / {n_splits-1} =====")

        train_dataloader = DataLoader(dataset[fold],  batch_size=64, shuffle=True,
                          collate_fn=mixup_collate_fn(0.4),
                          drop_last=True)
        val_dataloader = DataLoader(val_dataset[fold], batch_size=64,
                          collate_fn=mixup_collate_fn(0.0))

        imu_dataloader =  DataLoader(imu_dataset[fold], batch_size=64, shuffle=True,
                          collate_fn=mixup_collate_fn(0.4),
                          drop_last=True)

        imu_val_dataloader = DataLoader(imu_val_dataset[fold], batch_size=64,
                          collate_fn=mixup_collate_fn(0.0))
        
        
        class_weight = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(num_classes),
            y=labels[tr_idx]
        )
        steps_per_epoch = math.ceil(len(tr_idx) / 64)

        model = LitModel(imu_ch, tof_ch, num_classes, class_weight=class_weight)
        imu_model = LitModel(imu_ch, tof_ch, num_classes, imu_only=True, class_weight=class_weight)
        # fold ごとに保存ディレクトリを分けると整理しやすい
        ckpt_dir = export_dir 
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        trainer = L.Trainer(
            max_epochs      = 160,
            accelerator     = cfg.train.device,
            callbacks = [
                L.pytorch.callbacks.EarlyStopping(
                    monitor="val/acc",
                    patience=40,
                    mode="max"),
                L.pytorch.callbacks.ModelCheckpoint(
                    dirpath=ckpt_dir,
                    filename=f"best_of_fold_{fold+1}",
                    monitor="val/acc",
                    save_top_k=1,
                    mode="max")
            ],
            log_every_n_steps = 50,
        )
        print("start training")
        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        trainer = L.Trainer(
            max_epochs      = 160,
            accelerator     = cfg.train.device,
            callbacks = [
                L.pytorch.callbacks.EarlyStopping(
                    monitor="val/acc",
                    patience=40,
                    mode="max"),
                L.pytorch.callbacks.ModelCheckpoint(
                    dirpath=ckpt_dir,
                    filename=f"best_of_fold_imu_{fold+1}",
                    monitor="val/acc",
                    save_top_k=1,
                    mode="max")
            ],
            log_every_n_steps = 50,
        )
        print("start training imu")
        trainer.fit(model=imu_model, train_dataloaders=imu_dataloader, val_dataloaders=imu_val_dataloader)
        print("start evaluation")
        gesture_classes = np.load(
            export_dir / "gesture_classes.npy",  # 例: ["hand", "rest", …]
            allow_pickle=True
        )
        with torch.no_grad():
            model = LitModel.load_from_checkpoint(
                    checkpoint_path=os.path.join(ckpt_dir, f"best_of_fold_{fold+1}.ckpt"),
                    imu_ch=imu_ch, 
                    tof_ch=tof_ch,
                    num_classes  = 18,
                )
            model.eval()
            model.to(cfg.train.device)

            imu_model = LitModel.load_from_checkpoint(
                    checkpoint_path=os.path.join(ckpt_dir, f"best_of_fold_imu_{fold+1}.ckpt"),
                    imu_only=True,
                    imu_ch=imu_ch, 
                    tof_ch=tof_ch,
                    num_classes=18,
                )
            imu_model.eval()
            imu_model.to(cfg.train.device)

            submission = []
            submission_imu = []
            solution = []
            solution_imu = []
            for (batch, batch_imu) in zip(val_dataloader, imu_val_dataloader):

                x, y = batch
                x_imu, y_imu = batch_imu
                x = x.to(cfg.train.device, non_blocking=True)
                x_imu = x_imu.to(cfg.train.device, non_blocking=True)

                y_pred = model(x)
                y_pred_imu = imu_model(x_imu)

                for pred, pred_imu_1 in zip(y_pred, y_pred_imu):
                    idx   = pred.argmax(dim=0).cpu().numpy()
                    idx_imu_1 = pred_imu_1.argmax(dim=0).cpu().numpy()
                    submission.append(str(gesture_classes[idx]))
                    submission_imu.append(str(gesture_classes[idx_imu_1]))

                for y_true, y_true_imu in zip(y, y_imu):
                    idx   = y_true.argmax(dim=0).cpu().numpy()
                    idx_imu_1 = y_true_imu.argmax(dim=0).cpu().numpy()
                    solution.append(str(gesture_classes[idx]))
                    solution_imu.append(str(gesture_classes[idx_imu_1]))

            scores = {
                f"binary_score_of_fold_{fold+1}":      caluculate.binary_score(solution, submission),
                f"macro_score_of_fold_{fold+1}":      caluculate.macro_score(solution, submission),
                f"binary_score_of_fold_{fold+1}_imu":      caluculate.binary_score(solution_imu, submission_imu),
                f"macro_score_of_fold_{fold+1}_imu":      caluculate.macro_score(solution_imu, submission_imu),
            }
            with open(export_dir / f"scores_{fold+1}.json", "w") as f:
                json.dump(scores, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    run()
