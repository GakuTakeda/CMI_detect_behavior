# cv_main_tinycnn.py
import os, json, math, pathlib
import hydra, lightning as L
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np

# 既存ユーティリティの想定
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

        dm = GestureDataModule(cfg, fold_idx=fold)
        if fold == 0:
            dm.prepare_data()
        dm.setup()

        imu_path = dm.export_dir / "imu_cols.npy"
        imu_dim = len(np.load(imu_path, allow_pickle=True))
        cfg.imu_dim = imu_dim

        # num_classes を export から取得
        gesture_classes_path = dm.export_dir / "gesture_classes.npy"
        gesture_classes = np.load(gesture_classes_path, allow_pickle=True)
        cfg.num_classes = len(gesture_classes)

        # ------ モデル ------
        model = litmodel(
            cfg,
            lr_init     = cfg.train.lr_init,
            min_lr      = cfg.train.min_lr,
            weight_decay= cfg.train.weight_decay,
            class_weight= dm.class_weight,
        )

        # ------ Trainer / Callbacks ------
        ckpt_dir = dm.export_dir
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        early_stop = L.pytorch.callbacks.EarlyStopping(
            monitor="val/acc",
            patience=cfg.train.patience,
            mode="max",
        )
        checkpoint = L.pytorch.callbacks.ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"best_of_fold_tinycnn_{fold+1}",
            monitor="val/acc",
            save_top_k=1,
            mode="max",
        )

        trainer = L.Trainer(
            max_epochs=cfg.train.epochs,
            accelerator=getattr(cfg.train, "device", "auto"),  # "gpu"/"cpu"/"auto"
            devices=getattr(cfg.train, "devices", 1),
            precision=getattr(cfg.train, "precision", "32-true"),
            gradient_clip_val=1.0,
            callbacks=[early_stop, checkpoint],
            log_every_n_steps=50,
        )

        print("start training")
        trainer.fit(
            model=model,
            train_dataloaders=dm.train_dataloader(),
            val_dataloaders=dm.val_dataloader(),
        )

        # ------ 評価（fold のベスト ckptで） ------
        print("start evaluation")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_ckpt = os.path.join(ckpt_dir, f"best_of_fold_tinycnn_{fold+1}.ckpt")

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
            for batch in dm.val_dataloader():
                # 期待フォーマット:
                # (x_imu, x_tof, lengths, mask, y)  or
                # (x_imu, x_tof, lengths, mask, y, _, _)  or
                # （推論のみなら y なし: 今回のvalでは想定しない）
                if len(batch) == 5:
                    x_imu, x_tof, lengths, mask, y = batch
                elif len(batch) == 7:
                    x_imu, x_tof, lengths, mask, y, _, _ = batch
                else:
                    raise RuntimeError(f"Unexpected val batch format: len={len(batch)}")

                x_imu   = x_imu.to(device, non_blocking=True)        # [B,T,C_imu]
                x_tof   = x_tof.to(device, non_blocking=True)        # [B,T,C_toF,H,W]
                lengths = lengths.to(device, non_blocking=True)      # [B]
                mask    = mask.to(device, non_blocking=True)         # [B,T] (bool/byte)

                logits = model(x_imu, x_tof, lengths, mask)          # [B,num_classes]
                pred_ids = logits.argmax(dim=1).cpu().numpy()

                if y.ndim == 2:
                    true_ids = y.argmax(dim=1).cpu().numpy()
                else:
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


if __name__ == "__main__":
    run()
