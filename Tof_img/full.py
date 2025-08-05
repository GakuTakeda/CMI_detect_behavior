# cv_main.py
import hydra, lightning as L
from omegaconf import DictConfig
from utils     import LitModel, calc_f1, feature_eng, labeling, _pad, SequenceDataset_for_tof, mixup_collate_fn, seed_everything
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
    num_classes = 18

    #scaler.fit
    df = pd.read_csv("../data/train.csv")
    df = feature_eng(df)
    df["gesture_int"] = df["gesture"].apply(labeling)
    np.save(export_dir/"gesture_classes.npy", labeling("classes")) 

    meta = {'gesture','gesture_int','sequence_type','behavior','orientation',
                'row_id','subject','phase','sequence_id','sequence_counter'}
    
    feat_cols = [c for c in df.columns if c not in meta]

    scaler = StandardScaler().fit(
            df[feat_cols].ffill().bfill().fillna(0).values
        )
    joblib.dump(scaler, export_dir / "scaler.pkl")

    df[feat_cols] = scaler.transform(df[feat_cols].ffill().bfill().fillna(0).values)
    imu_cols = [c for c in feat_cols
                    if not (c.startswith("thm_") or c.startswith("tof_"))]
    imu_ch = len(imu_cols)
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

    tof_cols = [c for c in feat_cols if c.startswith("tof_")]

    tof_idx = [feat_cols.index(c) for c in tof_cols]
    X_tof = X[..., tof_idx]    
    tof_img = []
    for m in X_tof:
        tof_img.append(m.reshape(-1, 5, 8, 8))

    tof_img = np.array(tof_img)
    X = X[..., imu_idx]

    dataset = []
    val_dataset = []

    for tr_idx, val_idx in skf.split(X, labels):
        dataset.append(SequenceDataset_for_tof(X[tr_idx], tof_img[tr_idx], y_oh[tr_idx]))
        val_dataset.append(SequenceDataset_for_tof(X[val_idx], tof_img[val_idx], y_oh[val_idx]))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, labels)):
        print(f"\n===== Fold {fold} / {n_splits-1} =====")

        train_dataloader = DataLoader(dataset[fold],  batch_size=64, shuffle=True,
                          collate_fn=mixup_collate_fn(0.4),
                          drop_last=True)
        val_dataloader = DataLoader(val_dataset[fold], batch_size=64,
                          collate_fn=mixup_collate_fn(0.0))

        class_weight = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(num_classes),
            y=labels[tr_idx]
        )
        steps_per_epoch = math.ceil(len(tr_idx) / 64)

        model = LitModel(imu_ch, num_classes, class_weight=class_weight)
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

        print("start evaluation")
        gesture_classes = np.load(
            export_dir / "gesture_classes.npy",  # 例: ["hand", "rest", …]
            allow_pickle=True
        )
        with torch.no_grad():
            model = LitModel.load_from_checkpoint(
                    checkpoint_path=os.path.join(ckpt_dir, f"best_of_fold_{fold+1}.ckpt"),
                    imu_dim=imu_ch, 
                    num_classes  = 18,
                )
            model.eval()
            model.to(cfg.train.device)

            submission = []
            solution = []
            for batch in val_dataloader:

                x, img, y = batch
                x = x.to(cfg.train.device, non_blocking=True)
                img = img.to(cfg.train.device, non_blocking=True)

                y_pred = model(x, img)

                for pred in y_pred:
                    idx   = pred.argmax(dim=0).cpu().numpy()
                    submission.append(str(gesture_classes[idx]))

                for y_true in y:
                    idx   = y_true.argmax(dim=0).cpu().numpy()
                    solution.append(str(gesture_classes[idx]))

            scores = {
                f"binary_score_of_fold_{fold+1}":      caluculate.binary_score(solution, submission),
                f"macro_score_of_fold_{fold+1}":      caluculate.macro_score(solution, submission),}
            with open(export_dir / f"scores_{fold+1}.json", "w") as f:
                json.dump(scores, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    run()
