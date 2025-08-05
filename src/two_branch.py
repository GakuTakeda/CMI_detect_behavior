# cv_main.py
import hydra, lightning as L
from omegaconf import DictConfig
from datamodule import GestureDataModule
from lit_model     import LitTwoBranch, LitModelVariantGRU
import json
import os
import torch
import numpy as np
from CMI_2025 import score

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def run(cfg: DictConfig):
    n_splits = cfg.data.n_splits        # 例: 5
    for fold in range(n_splits):
        print(f"\n===== Fold {fold} / {n_splits-1} =====")

        dm = GestureDataModule(cfg, fold_idx=fold)
        if fold == 0:        # 1回だけ実行
            dm.prepare_data()
        dm.setup()

        model = LitTwoBranch(cfg,
                                     dm.imu_ch, 320,
                                     dm.num_classes,
                                     class_weight=dm.class_weight,
                                     steps_per_epoch=dm.steps_per_epoch)

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
        trainer.fit(model=model, train_dataloaders=dm.train_dataloader_imu_tof(), val_dataloaders=dm.val_dataloader_imu_tof())
        print("start evaluation")
        gesture_classes = np.load(
            dm.export_dir / "gesture_classes.npy",  # 例: ["hand", "rest", …]
            allow_pickle=True
        )
        with torch.no_grad():
            model = LitTwoBranch.load_from_checkpoint(
                    checkpoint_path=os.path.join(ckpt_dir, f"best_of_fold_{fold+1}.ckpt"),
                    cfg          = 0,
                    imu_ch       = 15,
                    tof_ch       = 320,
                    num_classes  = 18,
                )
            model.eval()
            model.to(cfg.train.device)
            submission = []
            solution = []
            for batch in dm.val_dataloader_imu_tof():

                x, y = batch
                x = x.to(cfg.train.device, non_blocking=True)

                y_pred = model(x)

                for pred, y_true in zip(y_pred, y):
                    idx   = pred.argmax(dim=0).cpu().numpy()
                    true_idx = y_true.argmax(dim=0).cpu().numpy()
                    submission.append(gesture_classes[idx])
                    solution.append(gesture_classes[true_idx])

            scores = {
                f"score_of_fold_{fold+1}":      score(solution,       submission),
            }

            with open(dm.export_dir / f"scores_{fold+1}.json", "w") as f:
                json.dump(scores, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    run()
