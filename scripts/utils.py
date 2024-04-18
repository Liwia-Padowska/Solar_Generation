import torch
import torch.autograd as autograd
import matplotlib.pyplot as plt
from typing import List
from scipy.stats import pearsonr
import numpy as np

def plot_losses(discriminator_losses: List[float], generator_losses: List[float]) -> None:
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


def plot_diversity_metrics(diversity_metrics_per_epoch: dict, epoch: int) -> None:
    """
    Plot diversity metrics per epoch.

    Args:
        diversity_metrics_per_epoch (dict): Dictionary containing diversity metrics per epoch.
        epoch (int): The current epoch.

    Returns:
        None
    """
    plt.figure(figsize=(10, 5))
    for metric, values in diversity_metrics_per_epoch.items():
        plt.plot(values, label=metric, alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Diversity Metric')
    plt.title('Diversity Metrics per Epoch')
    plt.legend()
    plt.savefig(f'diversity_metrics_epoch_{epoch}.png')
    plt.close()


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