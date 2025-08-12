# cv_main.py
import hydra, lightning as L
from omegaconf import DictConfig
from datamodule import GestureDataModule
from utils import litmodel
import json
import os
import torch
import numpy as np
from utils import set_seed, calc_f1
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

# --- objective() を置き換え ---
def objective(trial, cfg) -> float:
    caluculate = calc_f1()
    set_seed(cfg.data.random_seed)
    cfg_trial = deepcopy(cfg)

    # ← サーチ空間（必要に応じて増やしてOK）
    cfg_trial.model.cnn.se.drop      = trial.suggest_float("se_drop", 0.1, 0.5)
    cfg_trial.model.rnn.num_layers   = trial.suggest_int("rnn_num_layers", 1, 3)
    cfg_trial.model.rnn.hidden_size  = trial.suggest_int("rnn_hidden_size", 32, 256, step=32)
    cfg_trial.model.rnn.dropout      = trial.suggest_float("rnn_dropout", 0.1, 0.5)
    cfg_trial.model.lmu.hidden_size  = trial.suggest_int("lmu_hidden_size", 32, 256, step=32)
    cfg_trial.model.lmu.memory_size  = trial.suggest_int("lmu_memory_size", 32, 256, step=32)
    cfg_trial.model.lmu.theta        = trial.suggest_int("lmu_theta", 32, 256, step=32)
    cfg_trial.model.head.hidden      = trial.suggest_int("head_hidden", 256, 768, step=32)
    cfg_trial.train.lr_init          = trial.suggest_float("lr_init", 1e-4, 3e-3, log=True)
    cfg_trial.train.weight_decay     = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    fold_scores = []

    # お好みで cfg_trial.data.n_splits を使ってもOK
    for fold in range(5):
        print(f"\n===== Fold {fold} / {5-1} =====")

        dm = GestureDataModule(cfg_trial, fold_idx=fold)
        if fold == 0:
            dm.prepare_data()
        dm.setup()

        model = litmodel(
            cfg_trial,
            num_classes=18,
            lr_init=cfg_trial.train.lr_init,
            weight_decay=cfg_trial.train.weight_decay,
            class_weight=dm.class_weight
        )

        # ★ trial/ fold ごとに分離（上書き回避）
        ckpt_dir = (dm.export_dir / f"trial_{trial.number}" / f"fold_{fold+1}")
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        callbacks = [
            L.pytorch.callbacks.EarlyStopping(
                monitor="val/acc",
                patience=cfg_trial.train.patience,
                mode="max"
            ),
            L.pytorch.callbacks.ModelCheckpoint(
                dirpath=ckpt_dir,
                filename=f"best_of_fold_imu_{fold+1}",
                monitor="val/acc",
                save_top_k=1,
                mode="max"
            ),
            # ★ Lightning のログ "val/acc" を見て epoch 途中で prune
            PyTorchLightningPruningCallback(trial, monitor="val/acc"),
        ]

        trainer = L.Trainer(
            max_epochs=cfg_trial.train.epochs,
            accelerator=cfg_trial.train.device,   # "gpu"/"cpu"/"auto"
            devices=1 if cfg_trial.train.device != "cpu" else None,
            log_every_n_steps=50,
            callbacks=callbacks,
            deterministic=False,
        )

        print("start training")
        trainer.fit(
            model=model,
            train_dataloaders=dm.train_dataloader_imu(),
            val_dataloaders=dm.val_dataloader_imu()
        )

        # ====== 評価（foldスコアを計算） ======
        gesture_classes = np.load(dm.export_dir / "gesture_classes.npy", allow_pickle=True)

        with torch.no_grad():
            best_ckpt = os.path.join(ckpt_dir, f"best_of_fold_imu_{fold+1}.ckpt")
            model = litmodel.load_from_checkpoint(
                checkpoint_path=best_ckpt,
                cfg=cfg_trial,
                num_classes=18,
                lr_init=cfg_trial.train.lr_init,
                weight_decay=cfg_trial.train.weight_decay,
                class_weight=dm.class_weight
            ).eval().to(cfg_trial.train.device)

            submission, solution = [], []
            for batch in dm.val_dataloader_imu():
                x, y = batch
                x = x.to(cfg_trial.train.device, non_blocking=True)
                y_pred = model(x)
                for pred, y_true in zip(y_pred, y):
                    idx = pred.argmax(dim=0).item()
                    true_idx = y_true.argmax(dim=0).item()
                    submission.append(gesture_classes[idx])
                    solution.append(gesture_classes[true_idx])

            # 目的関数：2指標の和（必要なら重み付けを）
            fold_score = (
                caluculate.binary_score(solution, submission)
                + caluculate.macro_score(solution, submission)
            )
            fold_scores.append(float(fold_score))

        # （オプション）fold ごとの進捗を Optuna に記録
        trial.report(np.mean(fold_scores), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    mean_score = float(np.mean(fold_scores))
    trial.set_user_attr("fold_scores", fold_scores)  # 後で可視化に便利
    return mean_score

@hydra.main(config_path="config", config_name="config", version_base="1.3")
def run(cfg: DictConfig):
    study = optuna.create_study(
        study_name="imu_tuning",
        storage="sqlite:///lstmgru_tuning.db",     # ← storage_name ではなく storage
        load_if_exists=True, 
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True, group=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=2),
    )
    study.optimize(lambda trial: objective(trial, cfg), n_trials=50, gc_after_trial=True)
    print("Best value:", study.best_value)
    print("Best params:", study.best_trial.params)

if __name__ == "__main__":
    run()