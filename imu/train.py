# cv_main.py
import hydra, lightning as L
from omegaconf import DictConfig
from datamodule import GestureDataModule
from utils import litmodel
import json
import os
import torch
import numpy as np
from utils import seed_everything, calc_f1
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
from omegaconf import OmegaConf
import warnings
warnings.filterwarnings("ignore")

@hydra.main(config_path="config", config_name="config", version_base="1.3")
def run(cfg: DictConfig):
    seed_everything(cfg.data.random_seed)
    scorer = calc_f1()

    # Hydra のラン出力ディレクトリ（…/YYYY-MM-DD/HH-MM-SS）
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    date_part = run_dir.parent.name
    time_part = run_dir.name

    n_splits = cfg.data.n_splits
    print(f"CV n_splits = {n_splits}")

    avg_scores = []

    for fold in range(n_splits):
        print(f"\n===== Fold {fold} / {n_splits-1} =====")

        # ---- Logger（foldごとに一意のversion名）----
        tb_logger = TensorBoardLogger(
            save_dir=str(run_dir),
            name="tb",
            version=f"{date_part}_{time_part}_fold{fold+1}",
            default_hp_metric=False,
        )

        # ---- DataModule（固定長化は内部で完結）----
        dm = GestureDataModule(cfg, fold_idx=fold)
        if fold == 0:
            dm.prepare_data()
        dm.setup()
        cfg.num_classes = dm.num_classes  # データモジュールからクラス数を取得
        cfg.num_channels = dm.imu_ch  # 特徴量の数を設定
        cfg.data.pad_len = dm.pad_len  # パディング長を設定
        # ---- モデル ----
        model = litmodel(
            cfg,
            lr_init=cfg.train.lr_init,
            weight_decay=cfg.train.weight_decay,
        )

        # ---- コールバック / Trainer ----
        ckpt_dir = dm.export_dir
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        early_stop = EarlyStopping(monitor="val/f1_avg", patience=cfg.train.patience, mode="max")
        checkpoint = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"best_of_fold_imu_{fold+1}",
            monitor="val/f1_avg",
            save_top_k=1,
            mode="max",
        )

        trainer = L.Trainer(
            max_epochs=cfg.train.epochs,
            accelerator=cfg.train.device,
            devices=cfg.train.devices,
            precision=cfg.train.precision,
            gradient_clip_val=1.0,
            callbacks=[early_stop, checkpoint],
            log_every_n_steps=50,
            logger=tb_logger,
        )

        # ---- 学習 ----
        print("start training")
        trainer.fit(
            model=model,
            train_dataloaders=dm.train_dataloader_imu(),
            val_dataloaders=dm.val_dataloader_imu(),
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
            num_classes=dm.num_classes,
            lr_init=cfg.train.lr_init,
            weight_decay=cfg.train.weight_decay,
            class_weight=dm.class_weight,
        ).to(device)
        model.eval()

        gesture_classes = np.load(dm.export_dir / "gesture_classes.npy", allow_pickle=True)

        submission, solution = [], []
        with torch.no_grad():
            for batch in dm.val_dataloader_imu():
                # 検証は (x, y) 前提。万一 mixup collate が来ても y_a を使う
                if len(batch) == 2:
                    x, y = batch
                elif len(batch) == 4:
                    x, y_a, y_b, lam = batch
                    y = y_a
                else:
                    raise RuntimeError(f"Unexpected batch format: len={len(batch)}")

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits = model(x)  # 固定長: lengths/mask なし
                pred_ids = logits.argmax(dim=1).cpu().numpy()
                true_ids = (y.argmax(dim=1).cpu().numpy() if y.ndim == 2 else y.cpu().numpy())

                submission.extend([gesture_classes[i] for i in pred_ids])
                solution.extend([gesture_classes[i] for i in true_ids])

        scores = {
            f"binary_score_of_fold_{fold+1}_imu": scorer.binary_score(solution, submission),
            f"macro_score_of_fold_{fold+1}_imu":  scorer.macro_score(solution, submission),
        }
        with open(dm.export_dir / f"scores_{fold+1}.json", "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)

        avg_scores.append(scores[f"binary_score_of_fold_{fold+1}_imu"] + scores[f"macro_score_of_fold_{fold+1}_imu"])
        print("Fold scores:", scores)

        # 予測CSV
        (Path(dm.export_dir) / f"val_preds_fold_{fold+1}.csv").write_text(
            "\n".join(["true,pred"] + [f"{t},{p}" for t, p in zip(solution, submission)]),
            encoding="utf-8"
        )

        # メモリ掃除
        del trainer, model
        torch.cuda.empty_cache()

    # ---- 解決済み設定の保存 & 平均スコア ----
    with open(os.path.join(dm.export_dir, "config_resolved.yaml"), "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))

    avg_mean = float(np.mean(avg_scores)) if len(avg_scores) > 0 else float("nan")
    print("Average scores across folds:", avg_mean)
    with open(dm.export_dir / "avg_scores.json", "w", encoding="utf-8") as f:
        json.dump({"average_score": avg_mean}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    run()