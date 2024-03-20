import argparse
import numpy as np
import torch
import torch.autograd as autograd
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from scipy.stats import pearsonr

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
    lambda_gp = 10  # as in WGAN-GP paper: https://arxiv.org/pdf/1704.00028.pdf

    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    tensor = torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor
    long_tensor = torch.cuda.LongTensor if device.type == 'cuda' else torch.LongTensor

    discriminator_losses = []
    generator_losses = []

    # Plot a sample from the original dataset
    plot_original_samples(dataloader, opt.n_classes)

    batches_done = 0
    for epoch in range(opt.n_epochs):
        diversity_results = []
        for i, (sequence, labels) in enumerate(dataloader):

            real_sequence = sequence.type(tensor)
            labels = labels.type(long_tensor)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_d.zero_grad()

            # Sample noise and labels as generator input
            z = tensor(np.random.normal(0, 1, (sequence.shape[0], opt.latent_dim)))

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
            loss_c = torch.mean(fake_validity) - torch.mean(real_validity) + lambda_gp * gradient_penalty
            # print(torch.mean(fake_validity))
            # print(torch.mean(real_validity))
            # print(lambda_gp * gradient_penalty)

            loss_c.backward()
            optimizer_d.step()

            discriminator_losses.append(loss_c.item())

            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:
                optimizer_g.zero_grad()

                # Generate a batch of sequences
                fake_sequence = generator(z, labels)
                # Train on fake sequences
                fake_validity = discriminator(fake_sequence, labels)
                loss_g = (-1.0) * torch.mean(fake_validity)

                loss_g.backward()
                optimizer_g.step()

                generator_losses.append(loss_g.item())

                diversity = diversity_metrics(fake_sequence)
                diversity_results.append(diversity)

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), loss_c.item(), loss_g.item())
                )
                if (batches_done % opt.sample_interval == 0) and (epoch % 10 == 0):
                    titles = [f'Label: {label}' for i, label in enumerate(labels)]
                    plot_time_series(fake_sequence.data, int(np.sqrt(opt.batch_size)), titles,
                                     batch_number=batches_done)

                batches_done += opt.n_critic

        if diversity_results:
            avg_diversity = {metric: np.mean([result[metric] for result in diversity_results]) for metric in
                             diversity_results[0]}
            print(f"Epoch [{epoch}/{opt.n_epochs}], Average Diversity: {avg_diversity}")

    plot_losses(discriminator_losses, generator_losses)


def plot_losses(discriminator_losses, generator_losses):
    """
    Plot discriminator and generator losses across epochs.

    Args:
        discriminator_losses (list): List of discriminator losses.
        generator_losses (list): List of generator losses.

    Returns:
        None
    """
    plt.figure(figsize=(10, 5))
    plt.plot(discriminator_losses, label='Discriminator Loss', alpha=0.7)
    plt.plot(generator_losses, label='Generator Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Discriminator and Generator Losses')
    plt.legend()
    plt.show()


def compute_gradient_penalty(D: Discriminator, real_samples: torch.Tensor, fake_samples: torch.Tensor,
                             labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Calculates the gradient penalty for WGAN GP.
    It is required to enforce 1-Lipshitz continuity (L2 norm of the gradient needs to be <= to 1)
    It adds a regularisation term to critic's gradient by penalising whenever the gradient norm is higher than 1.
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
    epsilon = torch.rand((real_samples.size(0), 1), device=device)
    labels = labels.to(device)
    # Get random interpolation between real and fake samples
    interpolates = (epsilon * real_samples + (1 - epsilon) * fake_samples).requires_grad_(True)

    # Get logits of interpolated series
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
    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradient_penalty = ((gradients_norm - 1) ** 2).mean()
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


def plot_original_samples(dataloader: torch.utils.data.DataLoader, n_samples_per_class: int = 3) -> None:
    """
    Plots a few samples from the original dataset for each class.

    Args:
        dataloader (torch.utils.data.DataLoader): The data loader.
        n_samples_per_class (int): Number of samples to plot for each class. Default is 3.

    Returns:
        None
    """
    plt.figure(figsize=(15, 15))

    original_samples = torch.Tensor()
    original_labels = torch.Tensor()
    for batch_idx, (data, labels) in enumerate(dataloader):
        for class_idx in range(opt.n_classes):
            class_indices = (labels == class_idx).nonzero().squeeze(1)
            class_samples = data[class_indices[:n_samples_per_class]]
            original_samples = torch.cat((original_samples, class_samples), dim=0)
            original_labels = torch.cat((original_labels, labels[class_indices[:n_samples_per_class]]), dim=0)

    for i in range(opt.n_classes * n_samples_per_class):
        plt.subplot(opt.n_classes, n_samples_per_class, i + 1)
        plt.plot(original_samples[i].cpu().numpy())
        plt.title(f'Label: {original_labels[i]}', fontsize=15)
        plt.axis('off')

    plt.suptitle("Original Samples", fontsize=16)
    plt.tight_layout()
    plt.show()


def diversity_metrics(data: torch.Tensor) -> dict:
    """
    Calculate diversity metrics for generated time series.
    Args:
        data (torch.Tensor): Generated time series data.
    Returns:
        dict: Dictionary containing diversity metrics.
    """
    metrics = {}
    # Standard Deviation
    std_dev = torch.std(data, dim=1).mean().item()
    metrics['Standard Deviation'] = std_dev

    # Entropy
    data_normalized = (data - torch.min(data)) / (torch.max(data) - torch.min(data) + 1e-9)
    entropy = -torch.sum(data_normalized * torch.log(data_normalized + 1e-9), dim=1).mean().item()
    metrics['Entropy'] = entropy

    psd = torch.abs(torch.fft.fft(data)) ** 2
    psd_normalized = psd / torch.sum(psd, dim=1, keepdim=True)  # Normalize PSD
    entropy = -torch.sum(psd_normalized * torch.log(psd_normalized + 1e-9), dim=1).mean().item()
    metrics['Spectral_Entropy'] = entropy

    # Auto-correlation
    auto_corr_values = [np.correlate(seq.cpu().detach().numpy(), seq.cpu().detach().numpy(), mode='full') for seq in
                        data]
    auto_corr = np.mean(auto_corr_values)
    metrics['Auto-correlation'] = auto_corr

    # KL Divergence (Assuming normal distribution for simplicity)
    # It equals 0 iff the two distributions are the same, there is no upper bound
    expected_distribution = torch.abs(torch.randn_like(data)) + 1e-9  # Adding small constant to avoid zeros
    data_bounded = torch.abs(data) + 1e-9  # Adding small constant to avoid zeros
    kl_divergence = torch.mean(
        expected_distribution * (torch.log(expected_distribution) - torch.log(data_bounded))).item()
    metrics['KL Divergence'] = kl_divergence

    # Pearson Correlation Coefficient
    pearson_corr_values = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            corr, _ = pearsonr(data[i].cpu().detach().numpy(), data[j].cpu().detach().numpy())
            pearson_corr_values.append(corr)
    avg_pearson_corr = np.mean(pearson_corr_values)
    metrics['Pearson Correlation'] = avg_pearson_corr

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="number of training instances in one batch")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--series_length", type=int, default=96,
                        help="length of input time series, equal to length of generated outputs")
    parser.add_argument("--n_critic", type=int, default=5,
                        help="number of training steps for discriminator per iteration")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
    parser.add_argument("--dataset", type=str, choices=['open_power'], default='open_power', help="dataset to use")
    parser.add_argument("--sample_interval", type=int, default=1000,
                        help="interval between sampling of generated time series")
    opt = parser.parse_args()

    opt.n_classes = 5

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print("CUDA available:", cuda)

    generator = Generator(opt).to(device)
    discriminator = Discriminator(opt).to(device)

    if opt.dataset == 'open_power':
        solar_power_file = 'data/open_power/solar_15min.csv'
        weather_file = 'data/open_power/weather_data.csv'
        solar_power_data = pd.read_csv(solar_power_file)
        weather_data = pd.read_csv(weather_file)
        data = join_resample_dataframes(solar_power_data, weather_data, "utc_timestamp")
        # open_power_dataset = OpenPowerDataset(data, n_classes=opt.n_classes)
        open_power_dataset = OpenPowerDataset(data, n_classes=3, threshold_mode="manual", manual_thresholds=[10, 11.5])
        dataloader = torch.utils.data.DataLoader(
            open_power_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            drop_last=True
        )

    train_cwgan_gp(opt, generator, discriminator, dataloader, device)
