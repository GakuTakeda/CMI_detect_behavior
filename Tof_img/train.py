# cv_main_tinycnn.py
import os, json, yaml
import hydra, lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig
import torch
import numpy as np

# 既存ユーティリティ
from utils import litmodel, GestureDataModule, calc_f1, seed_everything


@hydra.main(config_path="config", config_name="tof_img", version_base="1.3")
def run(cfg: DictConfig):
    # 再現性
    seed_everything(cfg.data.random_seed)
    scorer = calc_f1()

    n_splits = cfg.data.n_splits
    print(f"CV n_splits = {n_splits}")

    for fold in range(n_splits):
        print(f"\n===== Fold {fold} / {n_splits-1} =====")

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
            monitor="val/acc",
            patience=cfg.train.patience,
            mode="max",
        )
        checkpoint = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"best_of_fold_tinycnn_{fold+1}",
            monitor="val/acc",
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

        print("Fold scores:", scores)

    with open(os.path.join(dm.export_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, indent=2, allow_unicode=True, sort_keys=False)

if __name__ == "__main__":
    run()
