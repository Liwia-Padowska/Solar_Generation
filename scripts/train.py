import sys
import os
import wandb
import argparse
import numpy as np
import torch
import torch.autograd as autograd
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from scipy.stats import pearsonr
import subprocess
import importlib
from torch.optim.lr_scheduler import ReduceLROnPlateau

module_path = '/mnt/batch/tasks/shared/LS_root/mounts/clusters/lpa1/code/Users/lpa/thesis_liwia/'
sys.path.append(module_path)
from models.custom_dataset import join_resample_dataframes
from models.generator import Generator
from models.discriminator import Discriminator
from models.custom_dataset import OpenPowerDataset
from utils import plot_losses, plot_diversity_metrics, plot_time_series, plot_original_samples, diversity_metrics
from config import N_EPOCHS, BATCH_SIZE, LR, B1, B2, N_CPU, LATENT_DIM, SERIES_LENGTH, N_CRITIC, CLIP_VALUE, DATASET, SAMPLE_INTERVAL, N_CLASSES

wandb.login(key="e07c6a046e65f891be99ba00784e2c18ae35e6a9")

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
    # Initialize Weights & Biases run
    wandb.init(project="your-project-name", config=vars(opt))

    # Log discriminator and generator architectures
    wandb.watch(generator)
    wandb.watch(discriminator)

    # Loss weight for gradient penalty
    lambda_gp = 10  # as in WGAN-GP paper: https://arxiv.org/pdf/1704.00028.pdf

    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Learning rate schedulers
    scheduler_g = ReduceLROnPlateau(optimizer_g, 'min', patience=10, factor=0.1, verbose=True)
    scheduler_d = ReduceLROnPlateau(optimizer_d, 'min', patience=10, factor=0.1, verbose=True)

    tensor = torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor
    long_tensor = torch.cuda.LongTensor if device.type == 'cuda' else torch.LongTensor

    discriminator_losses = []
    generator_losses = []

    diversity_metrics_per_epoch = {'Standard Deviation': [], 'Entropy': [], 'Spectral_Entropy': [],
                                   'Auto-correlation': [], 'KL Divergence': [], 'Pearson Correlation': []}

    avg_losses_per_epoch = {'Generator': [], 'Discriminator': []}

    # Plot a sample from the original dataset
    plot_original_samples(dataloader, opt.n_classes, 3)

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

            loss_c.backward()
            optimizer_d.step()

            discriminator_losses.append(loss_c.item())
            wandb.log({"discriminator_loss": loss_c.item()})

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
                wandb.log({"generator_loss": loss_g.item()})

                diversity = diversity_metrics(fake_sequence)
                diversity_results.append(diversity)

                if (batches_done % opt.sample_interval == 0) and (epoch % 10 == 0):
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch, opt.n_epochs, i, len(dataloader), loss_c.item(), loss_g.item())
                    )
                    if (epoch % 10 == 0):
                        titles = [f'Label: {label}' for i, label in enumerate(labels)]
                        plot_time_series(fake_sequence.data, int(np.sqrt(opt.batch_size)), titles,
                                         batch_number=batches_done)
                        # Log generated time series plots
                        wandb.log({"generated_time_series": wandb.Image(plt)})  # Assuming plt has the generated plot
                batches_done += opt.n_critic

        # Update learning rate
        scheduler_g.step(loss_g.item())
        scheduler_d.step(loss_c.item())

        if diversity_results:
            avg_diversity = {metric: np.mean([result[metric] for result in diversity_results]) for metric in
                             diversity_results[0]}
            print(f"Epoch [{epoch}/{opt.n_epochs}], Average Diversity: {avg_diversity}")

            # Log diversity metrics
            wandb.log(avg_diversity)

            # Save diversity metrics per epoch
            for metric, values in avg_diversity.items():
                diversity_metrics_per_epoch[metric].append(values)

        # Calculate average losses for the epoch
        avg_gen_loss = sum(generator_losses) / len(generator_losses)
        avg_disc_loss = sum(discriminator_losses) / len(discriminator_losses)

        # Save average losses per epoch
        avg_losses_per_epoch['Generator'].append(avg_gen_loss)
        avg_losses_per_epoch['Discriminator'].append(avg_disc_loss)

        # Log average losses to wandb
        wandb.log({'Average Generator Loss': avg_gen_loss, 'Average Discriminator Loss': avg_disc_loss})

        # Clear the losses lists for the next epoch
        generator_losses.clear()
        discriminator_losses.clear()

        # Plot and save diversity metrics per epoch
        plot_diversity_metrics(diversity_metrics_per_epoch, epoch)

    # Log discriminator and generator losses
    wandb.log({"discriminator_loss": discriminator_losses, "generator_loss": generator_losses})

    # Log final losses
    wandb.run.summary["final_discriminator_loss"] = discriminator_losses[-1]
    wandb.run.summary["final_generator_loss"] = generator_losses[-1]
    # Log final diversity metrics
    final_diversity_metrics = {metric: values[-1] for metric, values in diversity_metrics_per_epoch.items()}
    wandb.run.summary.update(final_diversity_metrics)

    plot_losses(discriminator_losses, generator_losses)

    # Finish Weights & Biases run
    wandb.finish()


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


def install(package: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def import_or_install(package: str) -> None:
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"{package} is not installed. Installing...")
        install(package)
        importlib.import_module(package)
    finally:
        print(f"{package} is installed and imported.")


if __name__ == '__main__':
    packages = [
        "argparse",
        "numpy",
        "torch",
        "torch.autograd",
        "pandas",
        "matplotlib.pyplot",
        "scipy.stats",
        "models.custom_dataset",
        "models.generator",
        "models.discriminator",
        "models.custom_dataset",
        "torch.utils.data",
        "wandb",
        "config",
    ]

    print(f"{os.getcwd()}", flush=True)
    for package in packages:
        import_or_install(package)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=N_EPOCHS, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="number of training instances in one batch")
    parser.add_argument("--lr", type=float, default=LR, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=B1, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=B2, help="adam: decay of second order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=N_CPU, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=LATENT_DIM, help="dimensionality of the latent space")
    parser.add_argument("--series_length", type=int, default=SERIES_LENGTH,
                        help="length of input time series, equal to length of generated outputs")
    parser.add_argument("--n_critic", type=int, default=N_CRITIC,
                        help="number of training steps for discriminator per iteration")
    parser.add_argument("--clip_value", type=float, default=CLIP_VALUE, help="lower and upper clip value for disc. weights")
    parser.add_argument("--dataset", type=str, choices=['open_power'], default=DATASET, help="dataset to use")
    parser.add_argument("--sample_interval", type=int, default=SAMPLE_INTERVAL,
                        help="interval between sampling of generated time series")
    parser.add_argument("--n_classes", type=int, default=N_CLASSES,
                        help="number of classes to classify time series")
    opt = parser.parse_args()

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print("CUDA available:", cuda)

    generator = Generator(opt).to(device)
    discriminator = Discriminator(opt).to(device)

    if opt.dataset == 'open_power':
        os.chdir(module_path)
        print("Current Directory:", os.getcwd())
        solar_power_file = 'data/open_power/solar_15min.csv'
        weather_file = 'data/open_power/weather_data.csv'
        solar_power_data = pd.read_csv(solar_power_file)
        weather_data = pd.read_csv(weather_file)
        data = join_resample_dataframes(solar_power_data, weather_data, "utc_timestamp")
        open_power_dataset = OpenPowerDataset(data, n_classes=3, threshold_mode="manual", manual_thresholds=[10, 11.5])
        dataloader = torch.utils.data.DataLoader(
            open_power_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            drop_last=True
        )

    train_cwgan_gp(opt, generator, discriminator, dataloader, device)
