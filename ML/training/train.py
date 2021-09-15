from ML.training.data import ProductDataset,ProductDataModule
from ML.training.models import ProductModel
from ML.training import config

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

def train_model():
    pl.seed_everything(42)

    # Callbacks
    model_checkpoint = ModelCheckpoint(monitor="val_loss",
                                    verbose=True,
                                    filename="{epoch}_{val_loss:.4f}")
    early_stopping = EarlyStopping('val_loss', patience=4)


    dm = ProductDataModule()
    cassava_model = ProductModel()
    trainer = pl.Trainer(gpus=-1, max_epochs=30,
                        callbacks=[model_checkpoint, early_stopping])
    trainer.fit(cassava_model, dm)

    # manually you can save best checkpoints -
    trainer.save_checkpoint(config.MODEL_NAME)
    # torch.save(dm.state_dict(), config.MODEL_NAME)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    train_model()