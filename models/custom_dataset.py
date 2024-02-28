import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import pandas as pd
import numpy as np

class OpenPowerDataset(Dataset):
    def __init__(self, root_dir, target_col, group_ids=None, time_varying_unknown_reals=None, max_encoder_length=None,
                 max_prediction_length=None, min_prediction_length=None, train=True, transform=None):
        # Load your dataset here, assuming it's stored in CSV format
        self.data = pd.read_csv(root_dir)
        # Assuming 'weather_labels' contains weather labels as one-hot encoded vectors
        self.weather_labels = self.data.iloc[:,
                              -num_labels:].values  # Assuming the last columns are one-hot encoded labels
        # Additional parameters for TimeSeriesDataset
        if group_ids is None:
            group_ids = self.data[
                'group_id'].unique()  # Assuming 'group_id' is a column identifying individual time series
        if time_varying_unknown_reals is None:
            time_varying_unknown_reals = [self.data.columns.difference(
                ['weather_labels', 'group_id'])]  # Assuming other columns are time-varying features
        super().__init__(
            target_col=target_col,
            group_ids=group_ids,
            time_varying_unknown_reals=time_varying_unknown_reals,
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            min_prediction_length=min_prediction_length,
            train=train
        )

    def __getitem__(self, idx):
        # Get the data slice for the given index
        sample = self.data.iloc[idx]
        # Extract features and convert to PyTorch tensor
        features = sample.drop(columns=['weather_labels']).values.astype(np.float32)
        # Apply transformations if specified
        if self.transform:
            features = self.transform(features)
        # Get the corresponding weather labels for the sample
        labels = self.weather_labels[idx]
        return features, labels


# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])