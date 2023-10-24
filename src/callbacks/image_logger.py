from typing import Any
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer import Trainer
from lightning import LightningModule
import torch
import wandb


class ImagePredictionLogger(Callback):
    def __init__(self, num_samples: int = 32):
        super().__init__()
        self.val_imgs = None
        self.val_labels = None
        self.num_samples = num_samples
    
    def on_validation_batch_start(
            self, trainer: Trainer,
            pl_module: LightningModule,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0):
        if batch_idx == 0:
            # print("save the first batch for image logging")
            X, y = batch
            self.val_imgs = X.clone()
            self.val_labels = y.clone()
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        # print("on_validation_epoch_end")
        if self.val_imgs is None or self.val_labels is None:
            print("no images to log")
            return
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        # Log image to wandb
        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.log({
                "examples": [wandb.Image(
                    x,
                    caption=f"Pred:{pred}, Label:{y}")
                    for x, pred, y in zip(
                        val_imgs[:self.num_samples],
                        preds[:self.num_samples],
                        val_labels[:self.num_samples]
                    )
                ]
            })
        else:
            print("logger or logger.experiment not found")
        