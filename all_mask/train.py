# cv_main.py
import hydra, lightning as L
from omegaconf import DictConfig, OmegaConf
from datamodule import GestureDataModule
from utils import litmodel_mask
import json
import os
import torch
import numpy as np
from utils import seed_everything, calc_f1

@hydra.main(config_path="config", config_name="all_mask", version_base="1.3")
def run(cfg: DictConfig):

    caluculate = calc_f1()

    seed_everything(cfg.data.random_seed)
    n_splits = cfg.data.n_splits        # 例: 5
    avg = []
    submission, solution = [], []
    for fold in range(n_splits):
        print(f"\n===== Fold {fold} / {n_splits-1} =====")

        dm = GestureDataModule(cfg, fold_idx=fold)
        if fold == 0:        # 1回だけ実行
            dm.prepare_data()  # "18_class" or "8_class"
        dm.setup()
        cfg.model.model.num_classes = dm.num_classes
        cfg.model.model.num_channels = dm.imu_ch + dm.tof_ch  # (= THM+ToF の合計)
        cfg.model.modalities.imu_dim = dm.imu_ch
        cfg.model.modalities.tof_dim = dm.tof_ch  # (= THM+ToF の合計)

        model = litmodel_mask(
            cfg,
            num_classes=18,
            lr_init=cfg.train.lr_init,
            weight_decay=cfg.train.weight_decay,
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
            device = cfg.train.device

            model = litmodel_mask.load_from_checkpoint(
                checkpoint_path=os.path.join(ckpt_dir, f"best_of_fold_imu_{fold+1}.ckpt"),
                cfg=cfg,
                num_classes=18,
                lr_init=cfg.train.lr_init,
                weight_decay=cfg.train.weight_decay,
                class_weight=dm.class_weight,
            )
            model.eval().to(device)

            for batch in dm.val_dataloader_imu():
                # 検証は通常 (x, lengths, mask, y)。万一 train 用 collate が来ても耐性を持たせる
                if len(batch) == 4:
                    x, lengths, mask, y = batch
                elif len(batch) == 6:
                    x, lengths, mask, y, _, _ = batch  # mixup の y_b/lam は無視
                else:
                    raise RuntimeError(f"Unexpected batch format: len={len(batch)}")

                x = x.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                logits = model(x, lengths, mask)              # [B, num_classes]
                pred_ids = logits.argmax(dim=1).cpu().numpy() # 予測クラスID
                true_ids = (y.argmax(dim=1).cpu().numpy() if y.ndim == 2 else y.cpu().numpy())

                submission.extend([gesture_classes[i] for i in pred_ids])
                solution.extend([gesture_classes[i] for i in true_ids])


        del model, trainer
        if fold != n_splits - 1:
            del dm
        torch.cuda.empty_cache()

    with open(os.path.join(dm.export_dir, "config_resolved.yaml"), "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))
    scores = {
        f"binary_score_of_fold_mask":  caluculate.binary_score(solution, submission),
        f"macro_score_of_fold_mask":   caluculate.macro_score(solution, submission),
    }
    print("Scores:", scores)

if __name__ == "__main__":
    run()