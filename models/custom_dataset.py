import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np


class OpenPowerDataset(Dataset):
    """
    Dataset class for the OpenPower dataset.

    Args:
        data (pd.DataFrame): DataFrame containing the dataset.
        n_classes (int): Number of classes for label categorization.
    """

    def __init__(self, data, n_classes):
        """
        Initializes the dataset.

        Extracts time series sequences and their corresponding labels from the given DataFrame.

        Args:
            data (pd.DataFrame): DataFrame containing the dataset.
            n_classes (int): Number of classes for label categorization.
        """
        sequences, labels = extract_time_series_with_labels(data, "NL_solar_generation_actual", "NL_temperature",
                                                            4 * 24, n_classes)
        self.sequences_tensor = torch.tensor(sequences)
        self.labels_tensor = torch.tensor(labels)
        # print("Sequences Tensor:")
        # print(self.sequences_tensor[0])
        # print("Labels Tensor:")
        # print(self.labels_tensor[0])
        # check_tensor_compatibility(self.sequences_tensor, self.labels_tensor)

    def __len__(self):
        return len(self.sequences_tensor)

    def __getitem__(self, idx):
        return self.sequences_tensor[idx], self.labels_tensor[idx]


def join_resample_dataframes(df1, df2, timestamp_column):
    """
    Joins and resamples two DataFrames based on a common timestamp column.

    Args:
        df1 (pd.DataFrame): First DataFrame.
        df2 (pd.DataFrame): Second DataFrame.
        timestamp_column (str): Name of the timestamp column.

    Returns:
        pd.DataFrame: Joined and resampled DataFrame.
    """
    df1[timestamp_column] = pd.to_datetime(df1[timestamp_column])
    df2[timestamp_column] = pd.to_datetime(df2[timestamp_column])
    df1.set_index(timestamp_column, inplace=True)
    df2.set_index(timestamp_column, inplace=True)
    df2_resampled = df2.resample('15T').mean()
    joined_df = df1.join(df2_resampled, how='inner', lsuffix='_df1', rsuffix='_df2')
    null_count = joined_df.isnull().sum().sum()
    # print("Number of null values before filling:", null_count)
    joined_df.fillna(joined_df.mean(), inplace=True)
    null_count_after_filling = joined_df.isnull().sum().sum()
    # print("Number of null values after filling:", null_count_after_filling)
    return joined_df


def check_tensor_compatibility(tensor1, tensor2):
    """
    Checks if two tensors are compatible.

    Args:
        tensor1 (torch.Tensor): First tensor.
        tensor2 (torch.Tensor): Second tensor.

    Returns:
        bool: True if tensors are compatible, False otherwise.
    """
    if tensor1.size(0) != tensor2.size(0):
        print("Number of samples in tensor1:", tensor1.size(0))
        print("Number of samples in tensor2:", tensor2.size(0))
        print("Tensors have different numbers of samples.")
        return False
    else:
        print("Number of samples in tensor1 and tensor2:", tensor1.size(0))
        print("Tensors are compatible.")
        return True


def extract_time_series_with_labels(dataframe, data_column_name, label_column_name, lookback_window, n_classes):
    """
    Extracts time series sequences and their corresponding labels from the DataFrame.

    Args:
        dataframe (pd.DataFrame): DataFrame containing the dataset.
        data_column_name (str): Name of the column containing time series data.
        label_column_name (str): Name of the column containing labels.
        lookback_window (int): Size of the window for each sequence.
        n_classes (int): Number of classes for label categorization.

    Returns:
        np.ndarray: Array of time series sequences.
        np.ndarray: Array of corresponding labels.
    """
    time_series_data = dataframe[data_column_name].values
    label_series_data = dataframe[label_column_name].values
    max_temp = np.max(label_series_data)
    min_temp = np.min(label_series_data)
    temp_range = max_temp - min_temp
    interval = temp_range / n_classes
    sequences = []
    labels = []
    for i in range(len(time_series_data) - lookback_window + 1):
        window = time_series_data[i:i + lookback_window]
        avg_temp = np.mean(label_series_data[i:i + lookback_window])
        class_label = min(int((avg_temp - min_temp) / interval), n_classes - 1)
        sequences.append(window)
        labels.append(class_label)
    sequences_array = np.array(sequences)
    labels_array = np.array(labels)
    return sequences_array, labels_array


if __name__ == '__main__':
    solar_power_file = 'data/open_power/solar_15min.csv'
    weather_file = 'data/open_power/weather_data.csv'
    solar_power_data = pd.read_csv(solar_power_file)
    weather_data = pd.read_csv(weather_file)
    data = join_resample_dataframes(solar_power_data, weather_data, "utc_timestamp")
    dataset = OpenPowerDataset(data, 5)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    for batch in data_loader:
        solar_power_batch, weather_data_batch = batch
        # Print batch shapes
        print("Solar power batch shape:", solar_power_batch.shape)
        print("Weather data batch shape:", weather_data_batch.shape)
        for sequence, label in zip(solar_power_batch, weather_data_batch):
            print("Sequence:", sequence)
            print("Label:", label)
        break
