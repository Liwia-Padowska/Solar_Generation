import torch
import torch.nn as nn
from torch import Tensor
import argparse


class Generator(nn.Module):
    def __init__(self, opt: argparse.Namespace):
        """
        Generator class for Conditional Wasserstein Generative Adversarial Network with Gradient Penalty(cWGAN-GP).

        Args:
            opt (argparse.Namespace): Configuration options.

        Attributes:
            opt (argparse.Namespace): Configuration options (from scripts/train).
            label_emb (torch.nn.Embedding): Embedding layer for label information.
            model (torch.nn.Sequential): Generator model architecture.
        """
        super(Generator, self).__init__()
        self.opt = opt

        # Embedding layer for label information
        self.label_emb = nn.Embedding(opt.n_classes, opt.latent_dim)

        def block(feat_in, fear_out, normalize=True):
            layers = [nn.Linear(feat_in, fear_out)]
            if normalize:
                layers.append(nn.LayerNorm(fear_out))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Define the generator model
        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.latent_dim, 256, normalize=True),
            *block(256, 128),
            nn.Linear(128, opt.series_length),
            nn.Tanh()
        )

    def forward(self, z: Tensor, labels: Tensor) -> Tensor:
        """
        Forward pass of the generator.

        Args:
            z (torch.Tensor): Input noise vector.
            labels (torch.Tensor): Input labels.

        Returns:
            torch.Tensor: Generated series data.
        """
        # Obtain label embeddings
        label_embedding = self.label_emb(labels=labels)

        # Concatenate label embeddings with noise vector
        gen_input = torch.cat((label_embedding, z), -1)

        # Generate series data
        series = self.model(input=gen_input)
        return series