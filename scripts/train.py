import argparse
import numpy as np
import torch
import torch.autograd as autograd
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

from models.custom_dataset import join_resample_dataframes
from models.generator import Generator
from models.discriminator import Discriminator
from models.custom_dataset import OpenPowerDataset


def train_cwgan_gp(opt: argparse.Namespace, generator: Generator, discriminator: Discriminator,
                   dataloader: torch.utils.data.DataLoader, device: torch.device):
    """
    Train the conditional WGAN-GP model.

    Args:
        opt (argparse.Namespace): Command line arguments.
        generator (Generator): The generator model.
        discriminator (Discriminator): The discriminator model.
        dataloader (DataLoader): The data loader.
        device (torch.device): The device to be used for training.

    Returns:
        None
    """
    # Loss weight for gradient penalty
    lambda_gp = 10

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if device.type == 'cuda' else torch.LongTensor

    batches_done = 0
    for epoch in range(opt.n_epochs):
        for i, (sequence, labels) in enumerate(dataloader):
            batch_size = sequence.shape[0]

            # Move to GPU if necessary
            real_sequence = sequence.type(Tensor)
            labels = labels.type(LongTensor)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Sample noise and labels as generator input
            z = Tensor(np.random.normal(0, 1, (sequence.shape[0], opt.latent_dim)))

            # Generate a batch of sequences
            fake_sequence = generator(z, labels)

            # Real sequences
            real_validity = discriminator(real_sequence, labels)
            # Fake sequences
            fake_validity = discriminator(fake_sequence, labels)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(
                discriminator, real_sequence.data, fake_sequence.data, labels.data, device)

            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:
                # Generate a batch of sequences
                fake_sequence = generator(z, labels)
                # Train on fake sequences
                fake_validity = discriminator(fake_sequence, labels)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )
                if batches_done % opt.sample_interval == 0:
                    titles = [f'Label: {label}' for i, label in enumerate(labels)]
                    plot_time_series(fake_sequence.data, int(np.sqrt(opt.batch_size)), titles, batch_number=batches_done)

                batches_done += opt.n_critic


def compute_gradient_penalty(D: Discriminator, real_samples: torch.Tensor, fake_samples: torch.Tensor,
                             labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Calculates the gradient penalty loss for WGAN GP.

    Args:
        D (Discriminator): The discriminator model.
        real_samples (torch.Tensor): Real samples.
        fake_samples (torch.Tensor): Fake samples.
        labels (torch.Tensor): Labels.
        device (torch.device): The device to be used.

    Returns:
        torch.Tensor: The gradient penalty.
    """
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1), device=device)
    labels = labels.to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    d_interpolates = D(interpolates, labels)
    fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(device)
    fake.requires_grad = False
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def plot_time_series(data: torch.Tensor, n_row: int, titles: List[str], batch_number: int) -> None:
    """
    Plots a grid of time series data.

    Args:
        data (torch.Tensor): The time series data.
        n_row (int): Number of rows in the grid.
        titles (List[str]): Titles for each time series.
        batch_number (int): The batch number.

    Returns:
        None
    """
    plt.figure(figsize=(15, 15))
    for i in range(n_row ** 2):
        plt.subplot(n_row, n_row, i + 1)
        plt.plot(data[i].cpu().numpy())
        plt.title(titles[i], fontsize=15)
        plt.axis('off')
    plt.suptitle(f"Generated Time Series (Batch {batch_number})", fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=150, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="number of training instances in one batch")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--series_length", type=int, default=96, help="length of input time series, equal to length of generated outputs")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iteration")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
    parser.add_argument("--dataset", type=str, choices=['open_power'], default='open_power', help="dataset to use")
    parser.add_argument("--sample_interval", type=int, default=1000, help="interval between sampling of generated time series")
    opt = parser.parse_args()

    opt.n_classes = 5

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print("CUDA available:", cuda)

    # Initialize generator and discriminator
    generator = Generator(opt).to(device)
    discriminator = Discriminator(opt).to(device)

    # Configure data loader
    if opt.dataset == 'open_power':
        solar_power_file = 'data/open_power/solar_15min.csv'
        weather_file = 'data/open_power/weather_data.csv'
        solar_power_data = pd.read_csv(solar_power_file)
        weather_data = pd.read_csv(weather_file)
        data = join_resample_dataframes(solar_power_data, weather_data, "utc_timestamp")
        open_power_dataset = OpenPowerDataset(data, n_classes=opt.n_classes)
        dataloader = torch.utils.data.DataLoader(
            open_power_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            drop_last=True
        )

    train_cwgan_gp(opt, generator, discriminator, dataloader, device)
