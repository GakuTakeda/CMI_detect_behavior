# cv_main_tinycnn.py
import os, json, yaml
import hydra, lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
import warnings
warnings.filterwarnings("ignore")   

# 既存ユーティリティ
from utils import litmodel, GestureDataModule, calc_f1, seed_everything

avg = []
@hydra.main(config_path="config", config_name="tof_img", version_base="1.3")
def run(cfg: DictConfig):
    # 再現性
    seed_everything(cfg.data.random_seed)
    scorer = calc_f1()

    run_dir = Path(HydraConfig.get().runtime.output_dir)  # .../YYYY-MM-DD/HH-MM-SS
    date_part = run_dir.parent.name                       # YYYY-MM-DD
    time_part = run_dir.name                              # HH-MM-SS

    n_splits = cfg.data.n_splits
    print(f"CV n_splits = {n_splits}")

    for fold in range(n_splits):
        print(f"\n===== Fold {fold} / {n_splits-1} =====")
        tb_logger = TensorBoardLogger(
            save_dir=str(run_dir),        # 例: outputs/tof_img/2025-08-22/09-05-11
            name="tb",                    # 例: .../tb/
            version=f"{date_part}_{time_part}_fold{fold+1}",  # 例: .../tb/2025-08-22_09-05-11_fold1
            default_hp_metric=False,
        )

        # ---- DataModule 準備（固定長化は DataModule 内で完結・mask なし）----
        dm = GestureDataModule(cfg, fold_idx=fold)
        if fold == 0:
            dm.prepare_data()
        dm.setup()

        # 入力次元・クラス情報を cfg に反映
        imu_dim = len(np.load(dm.export_dir / "imu_cols.npy", allow_pickle=True))
        cfg.imu_dim = imu_dim

        gesture_classes = np.load(dm.export_dir / "gesture_classes.npy", allow_pickle=True)
        cfg.num_classes = int(len(gesture_classes))

        # ---- モデル ----
        model = litmodel(
            cfg,
            lr_init=cfg.train.lr_init,
            min_lr=cfg.train.min_lr,
            weight_decay=cfg.train.weight_decay,
            class_weight=dm.class_weight,
        )

        # ---- コールバック / Trainer ----
        ckpt_dir = dm.export_dir
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        early_stop = EarlyStopping(
            monitor="val/f1_avg",
            patience=cfg.train.patience,
            mode="max",
        )
        checkpoint = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"best_of_fold_tinycnn_{fold+1}",
            monitor="val/f1_avg",
            save_top_k=1,
            mode="max",
        )

        trainer = L.Trainer(
            max_epochs=cfg.train.epochs,
            accelerator=getattr(cfg.train, "device", "auto"),
            devices=getattr(cfg.train, "devices", 1),
            precision=getattr(cfg.train, "precision", "32-true"),
            gradient_clip_val=1.0,
            callbacks=[early_stop, checkpoint],
            log_every_n_steps=50,
            logger=tb_logger,
        )

        # ---- 学習 ----
        print("start training")
        trainer.fit(
            model=model,
            train_dataloaders=dm.train_dataloader(),
            val_dataloaders=dm.val_dataloader(),
        )

        # ---- 評価（ベストCKPT）----
        print("start evaluation")
        best_ckpt = checkpoint.best_model_path
        if not best_ckpt or not os.path.exists(best_ckpt):
            raise FileNotFoundError("Best checkpoint not found. Check ModelCheckpoint settings/logs.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = litmodel.load_from_checkpoint(
            checkpoint_path=best_ckpt,
            cfg=cfg,
            lr_init=cfg.train.lr_init,
            min_lr=cfg.train.min_lr,
            weight_decay=cfg.train.weight_decay,
            class_weight=dm.class_weight,
        ).to(device)
        model.eval()

        submission, solution = [], []
        with torch.no_grad():
            for x_imu, x_tof, y in dm.val_dataloader():
                x_imu = x_imu.to(device, non_blocking=True)    # [B, L, C_imu]
                x_tof = x_tof.to(device, non_blocking=True)    # [B, L, Ct, H, W]
                y      = y.to(device, non_blocking=True)       # [B] (long)

                logits = model(x_imu, x_tof)                   # [B, num_classes]
                pred_ids = logits.argmax(dim=1).cpu().numpy()
                true_ids = y.cpu().numpy()

                submission.extend([gesture_classes[i] for i in pred_ids])
                solution.extend([gesture_classes[i] for i in true_ids])

        scores = {
            f"binary_score_of_fold_{fold+1}_tinycnn": scorer.binary_score(solution, submission),
            f"macro_score_of_fold_{fold+1}_tinycnn":  scorer.macro_score(solution, submission),
        }
        with open(dm.export_dir / f"scores_{fold+1}.json", "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)
        avg.append(scores[f"binary_score_of_fold_{fold+1}_tinycnn"] + scores[f"macro_score_of_fold_{fold+1}_tinycnn"])

        print("Fold scores:", scores)
        pd.DataFrame({"true": solution, "pred": submission}).to_csv(
            dm.export_dir / f"val_preds_fold_{fold+1}.csv", index=False
        )
        del trainer, model
        torch.cuda.empty_cache()

    with open(os.path.join(dm.export_dir, "config_resolved.yaml"), "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))
    print("Average scores across folds:", np.mean(avg))
    with open(dm.export_dir / "avg_scores.json", "w", encoding="utf-8") as f:
        json.dump({"average_score": np.mean(avg)}, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    run()
