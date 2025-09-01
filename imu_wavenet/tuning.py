# cv_main.py (tuning / Optuna)
import os
import json
import hydra
import torch
import lightning as L
import numpy as np
from copy import deepcopy
from omegaconf import DictConfig
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from datamodule import GestureDataModule
from utils import litmodel, seed_everything, calc_f1

import warnings
warnings.filterwarnings("ignore")
KS_CANDIDATES = {
    "k357":   (3, 5, 7),
    "k3711":  (3, 7, 11),
    "k579":   (5, 7, 9),
}
def objective(trial, cfg) -> float:
    scorer = calc_f1()
    seed_everything(cfg.data.random_seed)

    cfg_trial = deepcopy(cfg)

    # ----- meta -----
    cfg_trial.model.meta.proj_dim = trial.suggest_categorical("meta_proj_dim", [16, 32, 64, 128])
    cfg_trial.model.meta.dropout  = trial.suggest_float("meta_dropout", 0.0, 0.5)

    # ----- cnn.multiscale -----
    cfg_trial.model.cnn.multiscale.out_per_kernel = trial.suggest_categorical(
        "ms_out_per_kernel", [8, 12, 16, 24, 32]
    )
    ks_key = trial.suggest_categorical("ms_kernel_key", list(KS_CANDIDATES.keys()))
    cfg_trial.model.cnn.multiscale.kernel_sizes = list(KS_CANDIDATES[ks_key])

    # ----- cnn.se -----
    cfg_trial.model.cnn.se.out_channels = trial.suggest_categorical(
        "se_out_channels", [32, 48, 64, 96]
    )
    cfg_trial.model.cnn.se.drop = trial.suggest_float("se_drop", 0.0, 0.5)

    # ----- rnn -----
    cfg_trial.model.rnn.bidirectional = trial.suggest_categorical("rnn_bidir", [True, False])
    base_hidden = trial.suggest_int("rnn_hidden_size", 64, 256, step=32)
    # 片方向のときは少し底上げ（情報量を補うため）
    if not cfg_trial.model.rnn.bidirectional:
        base_hidden = max(base_hidden, 96)
    cfg_trial.model.rnn.hidden_size = base_hidden

    cfg_trial.model.rnn.num_layers = trial.suggest_int("rnn_num_layers", 1, 3)
    cfg_trial.model.rnn.dropout    = trial.suggest_float("rnn_dropout", 0.0, 0.6)

    # ----- noise -----
    cfg_trial.model.noise.std = trial.suggest_float("noise_std", 0.0, 0.15)

    # ----- head -----
    cfg_trial.model.head.hidden  = trial.suggest_int("head_hidden", 256, 1024, step=64)
    cfg_trial.model.head.dropout = trial.suggest_float("head_dropout", 0.2, 0.7)

    # ----- 学習系（任意だが推奨）-----
    cfg_trial.train.lr_init      = trial.suggest_float("lr_init", 1e-4, 3e-3, log=True)
    cfg_trial.train.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    n_splits = cfg_trial.data.n_splits
    device   = cfg_trial.train.device
    fold_scores: list[float] = []

    # ---- K-Fold ループ ----
    for fold in range(n_splits):
        print(f"\n===== Fold {fold} / {n_splits-1} (trial #{trial.number}) =====")

        dm = GestureDataModule(cfg_trial, fold_idx=fold)
        if fold == 0:
            dm.prepare_data()  # 特徴列・スケーラ等を作成
        dm.setup()

        model = litmodel(
            cfg=cfg_trial,
            num_classes=18,
            lr_init=cfg_trial.train.lr_init,
            weight_decay=cfg_trial.train.weight_decay,
            class_weight=dm.class_weight,
        )

        # trial / fold ごとに ckpt 保存パスを分ける（上書き回避）
        ckpt_dir = dm.export_dir / f"trial_{trial.number}" / f"fold_{fold+1}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        callbacks = [
            L.pytorch.callbacks.EarlyStopping(
                monitor="val/acc", patience=cfg_trial.train.patience, mode="max"
            ),
            L.pytorch.callbacks.ModelCheckpoint(
                dirpath=ckpt_dir,
                filename=f"best_of_fold_imu_{fold+1}",
                monitor="val/acc",
                save_top_k=1,
                mode="max",
            ),
            # Optuna のプルーニング（val/acc を監視）
            PyTorchLightningPruningCallback(trial, monitor="val/acc"),
        ]

        trainer = L.Trainer(
            max_epochs=cfg_trial.train.epochs,
            accelerator=device,                      # "gpu" / "cpu" / "auto"
            devices=1 if device != "cpu" else None, # GPU1枚想定
            log_every_n_steps=50,
            callbacks=callbacks,
            deterministic=False,
        )

        print("start training")
        trainer.fit(
            model=model,
            train_dataloaders=dm.train_dataloader_imu(),  # MixUp + Pad + Mask
            val_dataloaders=dm.val_dataloader_imu(),      # Pad + Mask
        )

        # ====== 評価（foldスコアを計算） ======
        print("start evaluation")
        gesture_classes = np.load(dm.export_dir / "gesture_classes.npy", allow_pickle=True)

        with torch.no_grad():
            best_ckpt = os.path.join(ckpt_dir, f"best_of_fold_imu_{fold+1}.ckpt")
            model = litmodel.load_from_checkpoint(
                checkpoint_path=best_ckpt,
                cfg=cfg_trial,
                num_classes=18,
                lr_init=cfg_trial.train.lr_init,
                weight_decay=cfg_trial.train.weight_decay,
                class_weight=dm.class_weight,
            ).eval().to(device)

            submission, solution = [], []
            for batch in dm.val_dataloader_imu():
                # 検証は通常 (x, lengths, mask, y)。万一 train 用collateが来ても耐性あり
                if len(batch) == 4:
                    x, lengths, mask, y = batch
                elif len(batch) == 6:
                    x, lengths, mask, y, _, _ = batch  # mixup の y_b / lam は無視
                else:
                    raise RuntimeError(f"Unexpected batch format: len={len(batch)}")

                x = x.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                logits = model(x, lengths, mask)               # [B, num_classes]
                pred_ids = logits.argmax(dim=1).cpu().numpy()
                true_ids = (y.argmax(dim=1).cpu().numpy() if y.ndim == 2 else y.cpu().numpy())

                submission.extend([gesture_classes[i] for i in pred_ids])
                solution.extend([gesture_classes[i] for i in true_ids])

            # 目的関数：2指標の和（必要なら重み付け可）
            fold_score = (
                scorer.binary_score(solution, submission) +
                scorer.macro_score(solution, submission)
            )
            fold_scores.append(float(fold_score))

        # ---- プルーニング判定（fold の途中経過）----
        trial.report(float(np.mean(fold_scores)), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # ---- 最終：fold平均を返す / 併せて user_attr に全foldを保存 ----
    mean_score = float(np.mean(fold_scores))
    trial.set_user_attr("fold_scores", fold_scores)  # list[float]（可視化でそのまま使える）
    return mean_score


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def run(cfg: DictConfig):
    study = optuna.create_study(
        study_name="imu_tuning_3",
        storage="sqlite:///lstmgru_tuning.db",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True, group=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=2),
    )
    study.optimize(lambda t: objective(t, cfg), n_trials=50, gc_after_trial=True)

    print("\n===== Best Result =====")
    print("Best value :", study.best_value)
    print("Best params:", study.best_trial.params)


if __name__ == "__main__":
    run()
