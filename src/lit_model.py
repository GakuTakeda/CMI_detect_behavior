# src/lit_module.py
import torch, lightning as L, numpy as np
from sklearn.utils.class_weight import compute_class_weight
from utils import TwoBranchModel, ModelVariant_GRU
from omegaconf import OmegaConf
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from torchmetrics.classification import MulticlassAccuracy

class LitGestureClassifier(L.LightningModule):
    def __init__(self, cfg, imu_ch, tof_ch,
                 num_classes, class_weight, steps_per_epoch, imu_only=False, conv_ch=64, lstm_units=128,
                 lr_init=1e-3, weight_decay=1e-4, t0_factor=5,):
        super().__init__()

        if imu_only == True:
            self.model = ModelVariant_GRU(num_classes=9)
        else:
            self.model = TwoBranchModel(
                imu_ch, tof_ch, num_classes,
                conv_ch   = conv_ch,
                lstm_units= lstm_units,
            )

        self.save_hyperparameters()   

        # 3) その他
        self.lr = lr_init
        self.wd = weight_decay
        self.steps_per_epoch = steps_per_epoch
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
        logits = self(X)                       # 形状は (N, L, C) か (N, C, L)

        if logits.dim() == 3:
            # --- shape に応じて平均化方向を自動判定 ---
            if logits.shape[-1] == self.num_classes:      # (N, L, C)
                logits = logits.mean(dim=1)               # → (N, C)
            elif logits.shape[1] == self.num_classes:     # (N, C, L)
                logits = logits.mean(dim=2)               # → (N, C)
            else:
                raise RuntimeError(f"Unexpected logits shape {logits.shape}")
        elif logits.dim() != 2:
            raise RuntimeError(f"Unsupported logits ndim={logits.dim()}")

        targets = y.argmax(dim=1)              # (N,)

        loss = torch.nn.functional.cross_entropy(
            logits, targets,
            weight=self.class_weight,          # 長さ = self.num_classes
            reduction="mean"
        )
        self.log(f"{stage}/loss", loss, prog_bar=True)
        return loss

    def training_step  (self, batch, _): return self._step(batch, "train")
    def validation_step(self, batch, _): return self._step(batch, "val")
    def test_step(self, batch, _): return self._step(batch, "test")

    # -------- optim ----------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=self.t0_factor*self.steps_per_epoch
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

class LitModelVariantGRU(L.LightningModule):
    """
    LightningModule wrapping ModelVariant_GRU
    -----------------------------------------
    * 入力        : x         … (B, L, C)    float32
    * 分類ラベル  : y_cls     … (B,)         int64  or one-hot
    * 回帰ターゲット: y_reg   … (B,)         float32  (オプション)
    """
    def __init__(
        self,
        num_classes: int,
        lr_init: float = 1e-3,
        weight_decay: float = 1e-4,
        cls_loss_weight: float = 1.0,
        reg_loss_weight: float = 0.1,
        class_weight: torch.Tensor | None = None,   # CE のクラス重み
    ):
        super().__init__()
        self.save_hyperparameters()

        # --- core model -----------------------------------------------------
        self.model = ModelVariant_GRU(num_classes=num_classes)

        # 損失関数
        self.mse = nn.MSELoss()   # 回帰用

        # logging 用
        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.val_acc   = MulticlassAccuracy(num_classes=num_classes, average="macro")
        
        self.register_buffer(
            "class_weight",
            torch.tensor(class_weight, dtype=torch.float32)
        )

    # --------------------------------------------------------------------- #
    # forward はそのまま
    def forward(self, x):
        return self.model(x)   # -> (logits, regression)

    # --------------------------------------------------------------------- #
    def _shared_step(self, batch, stage: str):
        """
        共通処理（train/val/test）
        """
        if len(batch) == 3:
            x, y_cls, y_reg = batch
        else:
            # 回帰ターゲット無しなら None に
            x, y_cls = batch
            y_reg = None

        logits, pred_reg = self(x)
        # --- 損失計算 ------------------------------------------------------
        ce = nn.CrossEntropyLoss(weight=self.class_weight)
        loss_cls = ce(logits, y_cls.argmax(dim=1) if y_cls.ndim == 2 else y_cls)
        loss = self.hparams.cls_loss_weight * loss_cls

        if y_reg is not None:
            loss_reg = self.mse(pred_reg.squeeze(-1), y_reg.float())
            loss = loss + self.hparams.reg_loss_weight * loss_reg
        else:
            loss_reg = torch.tensor(0.0, device=self.device)

        # --- ログ ----------------------------------------------------------
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        if stage == "train":
            self.train_acc(preds, y_cls.argmax(dim=1) if y_cls.ndim == 2 else y_cls)
            self.log_dict(
                {
                    "train/loss": loss,
                    "train/loss_cls": loss_cls,
                    "train/loss_reg": loss_reg,
                    "train/acc": self.train_acc,
                },
                on_step=False, on_epoch=True, prog_bar=True,
            )
        elif stage == "val":
            self.val_acc(preds, y_cls.argmax(dim=1) if y_cls.ndim == 2 else y_cls)
            self.log_dict(
                {
                    "val/loss": loss,
                    "val/loss_cls": loss_cls,
                    "val/loss_reg": loss_reg,
                    "val/acc": self.val_acc,
                },
                on_step=False, on_epoch=True, prog_bar=True,
            )
        else:  # test
            self.log_dict(
                {"test/loss": loss, "test/loss_cls": loss_cls, "test/loss_reg": loss_reg},
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
        opt = Adam(self.parameters(),
                   lr=self.hparams.lr_init,
                   weight_decay=self.hparams.weight_decay)

        # Plateau で LR 半減
        scheduler = {
            "scheduler": ReduceLROnPlateau(opt,
                                           mode="min",
                                           patience=2,
                                           factor=0.5),
            "monitor": "val/loss",
        }
        return {"optimizer": opt, "lr_scheduler": scheduler}