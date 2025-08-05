# cv_main.py
import hydra, lightning as L
from omegaconf import DictConfig
from datamodule import GestureDataModule
from lit_model     import LitModelVariantConf
import json
import os
import torch
import numpy as np
from CMI_2025 import score
from lightning.pytorch.loggers import TensorBoardLogger

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def run(cfg: DictConfig):
    n_splits = cfg.data.n_splits        # 例: 5
    for fold in range(n_splits):
        print(f"\n===== Fold {fold} / {n_splits-1} =====")

        dm = GestureDataModule(cfg, fold_idx=fold)
        if fold == 0:        # 1回だけ実行
            dm.prepare_data()
        dm.setup()

        imu_model = LitModelVariantConf(
            num_classes=dm.num_classes,
            lr_init=cfg.train.lr_init,
            weight_decay=cfg.train.weight_decay,
            dense_drop=cfg.train.dense_drop,
            conv_drop=cfg.train.conv_drop,
            noise_std=cfg.train.noise_std,
            class_weight=dm.class_weight
        )

        # fold ごとに保存ディレクトリを分けると整理しやすい
        ckpt_dir = dm.export_dir 
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        trainer = L.Trainer(
            max_epochs      = cfg.train.epochs,
            accelerator     = cfg.train.device,
            callbacks = [
                L.pytorch.callbacks.EarlyStopping(
                    monitor="val/loss",
                    patience=cfg.train.patience,
                    mode="min"),
                L.pytorch.callbacks.ModelCheckpoint(
                    dirpath=ckpt_dir,
                    filename=f"best_of_fold_{fold+1}",
                    monitor="val/loss",
                    save_top_k=1,
                    mode="min")
            ],
            log_every_n_steps = 50,
        )
        print("start training")

        tb_logger = TensorBoardLogger(
            save_dir=os.path.join(dm.export_dir, "lightning_logs"),   # ここにサブフォルダを自動生成
            name=f"imu_{fold+1}",                  # → lightning_logs/imu/version_0/
        )

        trainer = L.Trainer(
            max_epochs      = cfg.train.epochs,
            accelerator     = cfg.train.device,
            logger          = tb_logger, 
            callbacks = [
                L.pytorch.callbacks.EarlyStopping(
                    monitor="val/loss",
                    patience=cfg.train.patience,
                    mode="min"),
                L.pytorch.callbacks.ModelCheckpoint(
                    dirpath=ckpt_dir,
                    filename=f"best_of_fold_imu_{fold+1}",
                    monitor="val/loss",
                    save_top_k=1,
                    mode="min")
            ],
            log_every_n_steps = 50,
        )
        print("start training imu")
        trainer.fit(model=imu_model, train_dataloaders=dm.train_dataloader_imu(), val_dataloaders=dm.val_dataloader_imu())
        print("start evaluation")
        gesture_classes = np.load(
            dm.export_dir / "gesture_classes.npy",  # 例: ["hand", "rest", …]
            allow_pickle=True
        )
        with torch.no_grad():

            imu_model = LitModelVariantConf.load_from_checkpoint(
                    checkpoint_path=os.path.join(ckpt_dir, f"best_of_fold_imu_{fold+1}.ckpt"),
                    num_classes=18,
                    lr_init=cfg.train.lr_init,
                    weight_decay=cfg.train.weight_decay,
                    class_weight=dm.class_weight
                )
            imu_model.eval()
            imu_model.to(cfg.train.device)

            submission_imu = []
            solution_imu = []
            for batch_imu in dm.val_dataloader_imu():

                x_imu, y_imu = batch_imu
                x_imu = x_imu.to(cfg.train.device, non_blocking=True)

                y_pred_imu_1 = imu_model(x_imu)

                for pred_imu_1 in y_pred_imu_1:
                    idx_imu_1 = pred_imu_1.argmax(dim=0).cpu().numpy()
                    submission_imu.append(str(gesture_classes[idx_imu_1]))
                for y_true_imu in y_imu:
                    idx_imu_1 = y_true_imu.argmax(dim=0).cpu().numpy()
                    solution_imu.append(str(gesture_classes[idx_imu_1]))

            scores = {
                f"score_of_fold_{fold+1}_imu":  score(solution_imu,   submission_imu),
            }

            with open(dm.export_dir / f"scores_{fold+1}.json", "w") as f:
                json.dump(scores, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    run()