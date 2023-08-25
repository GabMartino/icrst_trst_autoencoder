import gc
import glob
import os
import pickle
import torch

import hydra

from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
import os

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

import gc

from src.dataloaders.RandomMNISTDataLoader import RandomMNISTDataloader
from src.dataloaders.TotalRandomMNISTDataLoader import TotalRandomMNISTDataloader
from src.models.AE2 import AE2

gc.collect()


def savePrediction(path, predictions):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + "prediction.pkl", "wb") as f:
        pickle.dump(predictions, f, -1)


torch.cuda.empty_cache()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    for p in list(cfg.p):
        print("_" * 15, str(p), "_" * 15)

        datamodule = None
        if cfg.datamodule == "mnist_mnist_m":
            if cfg.datamodule_path.mnist_mnist_m.totalRandom:
                datamodule = TotalRandomMNISTDataloader(
                    cfg.batch_size,
                    cfg.datamodule_path.mnist_mnist_m.mnist.train_path,
                    p=p,
                )
            else:
                datamodule = RandomMNISTDataloader(
                    cfg.batch_size,
                    cfg.datamodule_path.mnist_mnist_m.mnist.train_path,
                    p=p,
                )

        checkpoint_path = (
            cfg.checkpoint_path
            + "/"
            + cfg.model
            + "/"
            + cfg.checkpoint_datamodule
            + "/"
            + str(p)
            + "/"
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss_epoch",
            dirpath=checkpoint_path,
            filename="{epoch:02d}-{val_loss:.2f}",
            save_last=True,
            mode="min",
        )

        model = None
        if cfg.model == "AE2":
            if cfg.restore_from_checkpoint:
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                list_of_files = glob.glob(
                    checkpoint_path + "*"
                )  # * means all if need specific format then *.csv
                latest_checkpoint = max(list_of_files, key=os.path.getctime)
                model = AE2.load_from_checkpoint(latest_checkpoint)
            else:
                model = AE2(
                    cfg.input_channels,
                    cfg.latent_space_size,
                    cfg.hidden_dims,
                    cfg.kernel_sizes,
                    cfg.strides,
                    cfg.input_size,
                    cfg.lr,
                    cfg.pooling,
                )

        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=cfg.logdir,
            name=cfg.model
            + "/"
            + str(cfg.datamodule)
            + "/"
            + str(p)
            + "/"
            + str(cfg.input_size)
            + "/"
            + str(cfg.batch_size)
            + "/"
            + str(cfg.lr),
        )

        trainer = pl.Trainer(
            max_epochs=cfg.epochs,
            accelerator="gpu",
            devices=[0],
            precision=32,
            logger=tb_logger,
            log_every_n_steps=1,
            detect_anomaly=True,
            callbacks=[
                EarlyStopping("val_loss_epoch", patience=cfg.early_stopping_patience),
                checkpoint_callback,
                LearningRateMonitor(),
            ],
        )
        if cfg.Train:
            trainer.fit(model, datamodule)

        if cfg.Test:
            predictions = trainer.predict(model, datamodule)
            path = (
                cfg.prediction_logdir
                + "/"
                + cfg.model
                + "/"
                + cfg.datamodule
                + "_source/"
                + str(p)
                + "/"
            )
            savePrediction(path, predictions)


if __name__ == "__main__":
    main()
