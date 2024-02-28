import torch
import torch.nn as nn
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        # Copied from cgan.py
        self.model = nn.Sequential(
            nn.Linear(opt.n_classes + int(opt.series_length), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            # We do not use tahn in WGAN-GP, it is a soft critic
        )

    def forward(self, series, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((series.view(series.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity