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
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm(128),
            nn.Linear(128, 1),
            # No tanh in WGAN-GP, as it is a soft critic
            # Batch normalisation introduces learnable parameters (scale and shift) for each feature,
            # allowing the network to learn the optimal normalisation for each layer during training,
            # but it may lead to correlations within a batch since it changes the problem of mapping a single input
            # to single output to mapping from an entire batch of inputs to a batch of outputs - but here we
            # penalize the norm of the critic's gradient with respect to each input independently, hence the WGAN-GP
            # paper recommends the layer normalisation.

            # Layer Norm operates on the feature dimension.
            # It normalizes the activations of a layer by computing the mean and variance
            # across the features for each individual sample.
            # The normalization process is applied independently to each sample,
            # which makes LN less sensitive to batch size variations compared to BN.
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
