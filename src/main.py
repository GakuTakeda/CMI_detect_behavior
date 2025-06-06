import hydra, lightning as L
from omegaconf import DictConfig
from datamodule   import GestureDataModule
from lit_model   import LitGestureClassifier

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def run(cfg: DictConfig):
    dm = GestureDataModule(cfg); dm.prepare_data(); dm.setup()
    lit = LitGestureClassifier(cfg, dm.imu_ch, dm.tof_ch,
                           dm.num_classes, class_weight=dm.class_weight)


    trainer = L.Trainer(
        max_epochs      = cfg.train.epochs,
        accelerator     = cfg.train.device,
        callbacks = [
            L.pytorch.callbacks.EarlyStopping(
                monitor="val_loss", patience=cfg.train.patience, mode="min"),
            L.pytorch.callbacks.ModelCheckpoint(
                dirpath=dm.export_dir, filename="best", monitor="val_loss",
                save_top_k=1, mode="min")
        ],
        log_every_n_steps = 50,
    )
    trainer.fit(lit, dm)                    # ← たった 1 行
    # best_model.ckpt は自動保存

if __name__ == "__main__":
    run()
