import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class PVDataset(Dataset):
    def __init__(self, csv_file, column, transform=None):
        self.data = pd.read_csv(csv_file)[f'{column}'].values.astype(float)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


# Transform function to normalize the data
class Normalize(object):
    def __call__(self, sample):
        return (sample - sample.mean()) / sample.std()
