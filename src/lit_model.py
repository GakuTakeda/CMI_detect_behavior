# src/lit_module.py
import torch, lightning as L, numpy as np
from sklearn.utils.class_weight import compute_class_weight
from utils import TwoBranchModel
from omegaconf import OmegaConf

class LitGestureClassifier(L.LightningModule):
    def __init__(self, cfg, imu_ch, tof_ch,
                 num_classes, class_weight, conv_ch=64, lstm_units=128,
                 lr_init=1e-3, weight_decay=1e-4, t0_factor=5):
        super().__init__()

        # 1) モデル本体
        self.model = TwoBranchModel(
            imu_ch, tof_ch, num_classes,
            conv_ch   = cfg.model.conv_ch,
            lstm_units= cfg.model.lstm_units,
        )

        self.save_hyperparameters()   

        # 3) その他
        self.lr = lr_init
        self.wd = weight_decay
        self.t0_factor = t0_factor
        self.register_buffer(
            "class_weight",
            torch.tensor(class_weight, dtype=torch.float32)
        )
    # -------- forward ----------
    def forward(self, x): return self.model(x)

    # -------- shared --------
    def _step(self, batch, stage):
        X, y = batch
        logits = self(X)
        loss = torch.nn.functional.cross_entropy(
            logits, y.argmax(1), weight=self.class_weight, reduction="mean"
        )
        self.log(f"{stage}_loss", loss, prog_bar=True)
        return loss

    def training_step  (self, batch, _): return self._step(batch, "train")
    def validation_step(self, batch, _): return self._step(batch, "val")

    # -------- optim ----------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=self.t0_factor*len(self.trainer.datamodule.train_dataloader())
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
