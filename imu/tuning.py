# tune_aug_tinycnn.py
import os
import json
import math
import hydra
import torch
import optuna
import numpy as np
import lightning as L
from copy import deepcopy
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from optuna.integration import PyTorchLightningPruningCallback

# 既存ユーティリティ（IMU-only 版）
from utils import litmodel, GestureDataModule, calc_f1, seed_everything


def _set_aug_from_trial(cfg_trial: DictConfig, trial: optuna.trial.Trial) -> None:
    """
    trial から cfg_trial.aug.* を上書き
    - warp_min <= 1.0 <= warp_max になるようにサンプリング
    - n_blocks, block_len は [lo, hi] の配列を保証
    """
    # ---- 時間方向系 ----
    cfg_trial.aug.p_time_shift    = trial.suggest_float("p_time_shift", 0.0, 0.9)
    cfg_trial.aug.max_shift_ratio = trial.suggest_float("max_shift_ratio", 0.0, 0.5)

    cfg_trial.aug.p_time_warp = trial.suggest_float("p_time_warp", 0.0, 0.9)
    warp_min = trial.suggest_float("warp_min", 0.8, 1.0)
    warp_max = trial.suggest_float("warp_max", 1.0, 1.2)
    if warp_min > warp_max:
        warp_min, warp_max = warp_max, warp_min
    cfg_trial.aug.warp_min = warp_min
    cfg_trial.aug.warp_max = warp_max

    cfg_trial.aug.p_block_dropout = trial.suggest_float("p_block_dropout", 0.0, 0.8)
    # n_blocks と block_len は [lo, hi]
    nb_hi = trial.suggest_int("n_blocks_hi", 0, 3)
    nb_lo = trial.suggest_int("n_blocks_lo", 0, nb_hi)
    cfg_trial.aug.n_blocks = [nb_lo, nb_hi]

    bl_hi = trial.suggest_int("block_len_hi", 0, 10)
    bl_lo = trial.suggest_int("block_len_lo", 0, bl_hi)
    cfg_trial.aug.block_len = [bl_lo, bl_hi]

    # ---- IMU系列 ----
    cfg_trial.aug.p_imu_jitter   = trial.suggest_float("p_imu_jitter", 0.0, 1.0)
    cfg_trial.aug.imu_sigma      = trial.suggest_float("imu_sigma", 0.0, 0.08)

    cfg_trial.aug.p_imu_scale    = trial.suggest_float("p_imu_scale", 0.0, 1.0)
    cfg_trial.aug.imu_scale_sigma= trial.suggest_float("imu_scale_sigma", 0.0, 0.15)

    cfg_trial.aug.p_imu_drift    = trial.suggest_float("p_imu_drift", 0.0, 1.0)
    cfg_trial.aug.drift_std      = trial.suggest_float("drift_std", 0.0, 0.01)
    cfg_trial.aug.drift_clip     = trial.suggest_float("drift_clip", 0.0, 0.5)

    cfg_trial.aug.p_imu_small_rot= trial.suggest_float("p_imu_small_rot", 0.0, 1.0)
    cfg_trial.aug.rot_deg        = trial.suggest_float("rot_deg", 0.0, 10.0)


def objective(trial: optuna.trial.Trial, cfg: DictConfig) -> float:
    scorer = calc_f1()
    seed_everything(cfg.data.random_seed)

    cfg_trial = deepcopy(cfg)
    _set_aug_from_trial(cfg_trial, trial)

    # （任意）学習系の軽いチューニングも可
    cfg_trial.train.lr_init      = trial.suggest_float("lr_init", 1e-4, 3e-3, log=True)
    cfg_trial.train.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    # 早期停止のパティエンスも調整（幅はお好みで）
    cfg_trial.train.patience     = trial.suggest_int("patience", 5, 20)

    n_splits = int(cfg_trial.data.n_splits)
    accelerator = getattr(cfg_trial.train, "device", "auto")
    devices     = getattr(cfg_trial.train, "devices", 1)
    precision   = getattr(cfg_trial.train, "precision", "32-true")

    fold_scores = []

    for fold in range(n_splits):
        print(f"\n===== Fold {fold} / {n_splits-1} (trial #{trial.number}) =====")

        dm = GestureDataModule(cfg_trial, fold_idx=fold)
        if fold == 0:
            dm.prepare_data()
        dm.setup()

        # IMU入次元/クラスをcfgに反映
        imu_dim = int(len(np.load(dm.export_dir / "imu_cols.npy", allow_pickle=True)))
        cfg_trial.imu_dim = imu_dim
        gesture_classes = np.load(dm.export_dir / "gesture_classes.npy", allow_pickle=True)
        cfg_trial.num_classes = int(len(gesture_classes))

        model = litmodel(
            cfg_trial,
            lr_init=cfg_trial.train.lr_init,
            min_lr=getattr(cfg_trial.train, "min_lr", 1e-6),
            weight_decay=cfg_trial.train.weight_decay,
        )

        # trial / fold ごとに ckpt 保存先を分ける
        ckpt_dir = dm.export_dir / "optuna_aug" / f"trial_{trial.number}" / f"fold_{fold+1}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        callbacks = [
            EarlyStopping(monitor="val/f1_avg", patience=cfg_trial.train.patience, mode="max"),
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename=f"best_of_fold_imu_{fold+1}",
                monitor="val/f1_avg",
                save_top_k=1,
                mode="max",
            ),
            PyTorchLightningPruningCallback(trial, monitor="val/f1_avg"),
        ]

        trainer = L.Trainer(
            max_epochs=cfg_trial.train.epochs,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            log_every_n_steps=50,
            callbacks=callbacks,
            deterministic=False,
            enable_progress_bar=True,
            enable_checkpointing=True,
            logger=False,  # ログが不要なら False（必要なら TensorBoardLogger を使ってください）
        )

        print("start training")
        trainer.fit(
            model=model,
            train_dataloaders=dm.train_dataloader(),  # IMU-only (拡張は DataModule 内で適用)
            val_dataloaders=dm.val_dataloader(),
        )

        # ===== 評価 =====
        print("start evaluation")
        best_ckpt = callbacks[1].best_model_path  # ModelCheckpoint
        if not best_ckpt or not os.path.exists(best_ckpt):
            # 監視指標が一度も更新されずに終了したケース等
            raise optuna.TrialPruned()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = litmodel.load_from_checkpoint(
            checkpoint_path=best_ckpt,
            cfg=cfg_trial,
            lr_init=cfg_trial.train.lr_init,
            min_lr=getattr(cfg_trial.train, "min_lr", 1e-6),
            weight_decay=cfg_trial.train.weight_decay,
        ).to(device)
        model.eval()

        submission, solution = [], []
        with torch.no_grad():
            for x_imu, y in dm.val_dataloader():
                x_imu = x_imu.to(device, non_blocking=True)
                y     = y.to(device, non_blocking=True)

                logits = model(x_imu)  # [B, num_classes]
                pred_ids = logits.argmax(dim=1).cpu().numpy()
                true_ids = y.cpu().numpy()

                submission.extend([gesture_classes[i] for i in pred_ids])
                solution.extend([gesture_classes[i] for i in true_ids])

        fold_score = (
            scorer.binary_score(solution, submission) +
            scorer.macro_score(solution, submission)
        )
        fold_scores.append(float(fold_score))

        # Optuna へ途中経過をレポート → プルーニング判定
        trial.report(float(np.mean(fold_scores)), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # キャッシュ解放
        del trainer, model
        torch.cuda.empty_cache()

    mean_score = float(np.mean(fold_scores))
    trial.set_user_attr("fold_scores", fold_scores)
    return mean_score


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def run(cfg: DictConfig):
    # 研究名/DB は必要に応じて変更してください
    study = optuna.create_study(
        study_name="imu_tinycnn_aug_tuning",
        storage="sqlite:///tinycnn_aug_tuning.db",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True, group=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
    )
    study.optimize(lambda t: objective(t, cfg), n_trials=50, gc_after_trial=True)

    print("\n===== Best Result =====")
    print("Best value :", study.best_value)
    print("Best params:", study.best_trial.params)

    # 実行フォルダにベストを保存（Hydra のカレントは run ディレクトリ）
    out = {
        "best_value": study.best_value,
        "best_params": study.best_trial.params,
        "best_user_attrs": study.best_trial.user_attrs,
    }
    with open("best_aug_params.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    run()
