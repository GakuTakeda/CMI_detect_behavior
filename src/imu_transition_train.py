# cv_main.py
import hydra, lightning as L
from omegaconf import DictConfig
from Macrodatamodule import GestureDataModule
from lit_model     import LitModel_for_Macro_Imu
import json
import os
import torch
import numpy as np
from CMI_2025 import score
from utils import set_seed, calc_f1
from sklearn.metrics import f1_score

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def run(cfg: DictConfig):

    caluculate = calc_f1()

    set_seed(cfg.data.random_seed)
    n_splits = cfg.data.n_splits        # 例: 5
    for fold in range(n_splits):
        print(f"\n===== Fold {fold} / {n_splits-1} =====")

        dm = GestureDataModule(cfg, fold_idx=fold)
        if fold == 0:        # 1回だけ実行
            dm.prepare_data()
        dm.setup()

        model = LitModel_for_Macro_Imu(cfg, num_classes=6)
        ckpt_dir = dm.export_dir 
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        trainer = L.Trainer(
            max_epochs      = cfg.train.epochs,
            accelerator     = cfg.train.device,
            callbacks = [
                L.pytorch.callbacks.EarlyStopping(
                    monitor="val/acc",
                    patience=cfg.train.patience,
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
        print("start training")
        trainer.fit(model=model, train_dataloaders=dm.train_dataloader_imu(), val_dataloaders=dm.val_dataloader_imu())
    
        print("start evaluation")
        gesture_classes = np.load(
            dm.export_dir / "gesture_classes.npy",  # 例: ["hand", "rest", …]
            allow_pickle=True
        )
        with torch.no_grad():

            model = LitModel_for_Macro_Imu(cfg, num_classes=6)
            model.eval()
            model.to(cfg.train.device)

            submission = []
            solution = []
            for batch in dm.val_dataloader_imu():
                x, y = batch
                x = x.to(cfg.train.device, non_blocking=True)

                y_pred = model(x)

                for pred, y_true in zip(y_pred, y):
                    idx = pred.argmax(dim=0).cpu().numpy()
                    true_idx = y_true.argmax(dim=0).cpu().numpy()
                    submission.append(gesture_classes[idx])
                    solution.append(gesture_classes[true_idx])

            np.save(dm.export_dir/f"fold{fold+1}submission.npy", submission)
            np.save(dm.export_dir/f"fold{fold+1}solution.npy", solution)

            scores = {
                f"f1_score_of_fold_{fold+1}_imu":     f1_score(solution, submission, average='macro',zero_division=0),
            }

            with open(dm.export_dir / f"scores_{fold+1}.json", "w") as f:
                json.dump(scores, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    run()

