# src/lit_module.py
import torch, lightning as L, numpy as np
from sklearn.utils.class_weight import compute_class_weight
from utils import MacroImuModel
from omegaconf import OmegaConf
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam
from torchmetrics.classification import MulticlassAccuracy


class LitModel_for_Macro_Imu(L.LightningModule):
    def __init__(self, cfg, num_classes, len_move, len_perform, 
                 lr_init=1e-3, weight_decay=1e-4, t0_factor=5, cls_loss_weight=1.0, class_weight=None):
        super().__init__()

        self.model = MacroImuModel(num_classes=num_classes, len_move=len_move, len_perform=len_perform)
        self.save_hyperparameters()

        # logging 用
        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.val_acc   = MulticlassAccuracy(num_classes=num_classes, average="macro")
        
        self.register_buffer(
            "class_weight",
            torch.tensor(class_weight, dtype=torch.float32)
        )

    def forward(self, x):
        return self.model(x)   
    def _shared_step(self, batch, stage: str):
        """
        共通処理（train/val/test）
        """

        x, y_cls = batch

        logits = self(x)
        # --- 損失計算 ------------------------------------------------------
        ce = nn.CrossEntropyLoss(weight=self.class_weight)
        loss_cls = ce(logits, y_cls)
        loss = self.hparams.cls_loss_weight * loss_cls

        # --- ログ ----------------------------------------------------------
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        if stage == "train":
            self.train_acc(preds, y_cls.argmax(dim=1) if y_cls.ndim == 2 else y_cls)
            self.log_dict(
                {
                    "train/loss": loss,
                    "train/acc": self.train_acc,
                },
                on_step=False, on_epoch=True, prog_bar=True,
            )
        elif stage == "val":
            self.val_acc(preds, y_cls.argmax(dim=1) if y_cls.ndim == 2 else y_cls)
            self.log_dict(
                {
                    "val/loss": loss,
                    "val/acc": self.val_acc,
                },
                on_step=False, on_epoch=True, prog_bar=True,
            )
        else:  # test
            self.log_dict(
                {"test/loss": loss, "test/loss_cls": loss_cls},
                on_step=False, on_epoch=True,
            )
        return loss

    # --------------------------------------------------------------------- #
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    # --------------------------------------------------------------------- #
    def configure_optimizers(self):
        opt = Adam(
            self.parameters(),
            lr=self.hparams.lr_init,
            weight_decay=self.hparams.weight_decay,
        )

        # --- ① 余弦減衰（エポック終了時に呼び出す） --------------------
        # T_max: 何エポックで最小 lr（eta_min）に到達するか
        # eta_min: 最小 lr（デフォルト 0）を少し残したい場合は設定
        scheduler = CosineAnnealingLR(
            opt,
            T_max=self.trainer.max_epochs,      # 例: 総エポック数
            eta_min=self.hparams.lr_init * 0.01 # 例: 初期 lr の 1%
        )

        # --- ② Warm Restarts を使いたい場合 ----------------------------
        # scheduler = CosineAnnealingWarmRestarts(
        #     opt,
        #     T_0=10,          # 初回サイクル長（エポック数）
        #     T_mult=2,        # サイクルを回すごとに長さを 2 倍
        #     eta_min=self.hparams.lr_init * 0.01,
        # )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",   # 1 エポックごとに step する
                "frequency": 1,        # 毎エポック
            }
        }
 