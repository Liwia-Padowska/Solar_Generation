import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, opt):
        """
        Discriminator class for Conditional Wasserstein Generative Adversarial Network with Gradient Penalty (cWGAN-GP).

        Args:
            opt (Namespace): Namespace containing configuration options.

        Attributes:
            label_embedding (nn.Embedding): Embedding layer for label information.
            model (nn.Sequential): Discriminator model architecture.
        """
        super(Discriminator, self).__init__()

        # Embedding layer for label information
        self.label_embedding = nn.Embedding(opt.n_classes, opt.latent_dim)

        self.model = nn.Sequential(
            nn.Linear(opt.series_length + opt.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            # No tanh in WGAN-GP, as it is a soft critic
        )

    def forward(self, series, labels):
        """
        Forward pass of the discriminator.
        Args:
            series (torch.Tensor): Input time series data.
            labels (torch.Tensor): Label information.
        Returns:
            real_probability (torch.Tensor): Discriminator output (probability that a sample comes from real data, number between [0,1])).
        """
        # Obtain label embeddings
        label_emb = self.label_embedding(labels)
        # Concatenate label embeddings with time series data
        d_in = torch.cat((series.view(series.size(0), -1), label_emb), -1)
        # Compute discriminator output
        return self.model(d_in)
