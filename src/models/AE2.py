from copy import deepcopy
from typing import List
import pytorch_lightning as pl
import torch.nn as nn
import numpy as np
import torch.optim as optim


class AE2(pl.LightningModule):
    def __init__(
        self,
        input_channels: int,
        latent_space_size: int,
        hidden_dims: List,
        kernel_sizes: List,
        strides: List,
        input_size: int,
        lr: float,
        pooling: True,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.latent_dim = latent_space_size
        self.in_channels = input_channels
        self.hidden_dims = hidden_dims
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.input_size = input_size
        self.lr = lr
        self.pooling = pooling

        modules = []

        """
            Encoder
        """
        self.last_feature_map_size = self.input_size
        in_channels = self.in_channels
        # print("input_channels", in_channels, input_size)
        for h_dim, kernel, stride in zip(
            self.hidden_dims, self.kernel_sizes, self.strides
        ):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=kernel,
                        stride=stride,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            if self.pooling:
                modules.append(nn.MaxPool2d((2, 2)))
            modules.append(nn.Dropout())
            in_channels = h_dim
            self.last_feature_map_size = (
                self.last_feature_map_size - (kernel - 1) - 1
            ) / stride + 1
            self.last_feature_map_size /= 2 if self.pooling else 1

        self.last_feature_map_size = int(self.last_feature_map_size)
        self.encoder = nn.Sequential(*modules)
        self.avgGlobalPooling = nn.AvgPool2d(kernel_size=self.last_feature_map_size)

        # print("last size ", self.last_feature_map_size, (1, self.in_channels, self.input_size, self.input_size))

        # summary(self.encoder, (1, self.in_channels, self.input_size, self.input_size))
        # Build Decoder
        modules = []
        # print("init", self.hidden_dims[-1], self.last_feature_map_size)

        """
            Let's make the first block of the decoder equals to the last of the encoder
        """
        decoder_hidden_dims = self.hidden_dims.copy()
        decoder_hidden_dims.append(decoder_hidden_dims[-1])
        decoder_hidden_dims.reverse()

        decoder_kernel_size = self.kernel_sizes.copy()
        decoder_kernel_size.reverse()

        reverse_strides = self.strides.copy()
        reverse_strides.reverse()

        out_size = deepcopy(self.last_feature_map_size)
        for i, k, s in zip(
            range(len(decoder_hidden_dims) - 1), decoder_kernel_size, reverse_strides
        ):
            if self.pooling:
                modules.append(nn.Upsample(scale_factor=2))

            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        decoder_hidden_dims[i],
                        decoder_hidden_dims[i + 1],
                        kernel_size=k,
                        stride=s,
                        padding=0,
                        output_padding=0,
                    ),
                    nn.BatchNorm2d(decoder_hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                    nn.Dropout(),
                )
            )
            out_size *= 2 if self.pooling else 1
            out_size = (out_size - 1) * s + (k - 1) + 1

        self.decoder = nn.Sequential(*modules)
        # print(out_size - self.input_size + 1)
        # out_kernel_size = (abs(out_size - self.input_size + 1)) + (
        #    2 if self.pooling else 0
        # )
        # print(out_size, out_kernel_size)
        # print((1, self.hidden_dims[-1], self.last_feature_map_size, self.last_feature_map_size))
        # summary(self.decoder, (1, self.hidden_dims[-1], self.last_feature_map_size, self.last_feature_map_size))
        self.final_layer = nn.Sequential(
            nn.Conv2d(
                decoder_hidden_dims[-1], out_channels=self.in_channels, kernel_size=1
            ),
            nn.Sigmoid(),
        )

    def generate(self, input):
        # input = input.reshape(1, 512, 1, 1)
        out = self.decoder(input)
        out = self.final_layer(out)
        return out

    def forward(self, input):
        # print(input.shape)
        out = self.encoder(input)
        out = self.decoder(out)
        out = self.final_layer(out)
        return out

    def getLatentVector(self, input):
        out = self.encoder(input)
        avg = self.avgGlobalPooling(out)
        return avg, out

    """

                    TRAINING METHODS

    """

    def training_step(self, batch, batch_idx):
        input_img, output_img, labels = batch
        out = self.forward(input_img)
        train_loss = nn.MSELoss()(output_img, out)
        self.log("train_loss", train_loss, sync_dist=True)
        return train_loss

    def training_epoch_end(self, outputs):
        mean_loss = np.mean([v["loss"].item() for v in outputs])
        self.log("train_loss_epoch", mean_loss, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        input_img, output_img, labels = batch
        out = self.forward(input_img)
        val_loss = nn.MSELoss()(input_img, out)
        self.log("val_loss", val_loss, sync_dist=True)
        return {"val_loss": val_loss, "labels": labels}

    def validation_epoch_end(self, outputs):
        mean_loss = np.mean([v["val_loss"].item() for v in outputs])
        self.log("val_loss_epoch", mean_loss, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        input_img, output, labels = batch
        out, flat = self.getLatentVector(input_img)

        return out, labels

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
