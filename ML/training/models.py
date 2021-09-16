import pytorch_lightning as pl
import torchvision.models as models
from ML.training import config
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.metrics.functional import accuracy
import torch.nn as nn
import torch

class ProductModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.efficient_net = EfficientNet.from_pretrained(
        #     'efficientnet-b2', num_classes=config.CLASSES)
        self.model=models.resnet18(pretrained=True)
        in_features = self.model.c.in_features
        self.model._fc = nn.Linear(in_features, config.CLASSES)

    def forward(self,x):
        out = self.model(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=1e-4, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=2, gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)
        self.log("train_acc", acc, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)
        self.log("val_acc", acc, prog_bar=True, logger=True),
        self.log("val_loss", loss, prog_bar=True, logger=True)