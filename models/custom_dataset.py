import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class OpenPowerDataset(Dataset):
    """
    Dataset class for the OpenPower dataset.

    Args:
        data (pd.DataFrame): DataFrame containing the dataset.
        n_classes (int): Number of classes for label categorization.
        threshold_mode (str): Mode for setting temperature thresholds. Options are "auto" or "manual".
        manual_thresholds (list): List of manual thresholds for each class. Only used if threshold_mode is set to "manual".
    """

    def __init__(self, data, n_classes, threshold_mode="auto", manual_thresholds=None):
        """
        Initializes the dataset.
        Extracts time series sequences and their corresponding labels from the given DataFrame.

        Args:
            data (pd.DataFrame): DataFrame containing the dataset.
            n_classes (int): Number of classes for label categorization.
            threshold_mode (str): Mode for setting temperature thresholds. Options are "auto" or "manual".
            manual_thresholds (list): List of manual thresholds for each class. Only used if threshold_mode is set to "manual".
        """
        sequences, labels = extract_time_series_with_labels(data, "NL_solar_generation_actual", "NL_temperature",
                                                            4 * 24, n_classes, threshold_mode, manual_thresholds)
        self.sequences_tensor = torch.tensor(sequences)
        self.labels_tensor = torch.tensor(labels)
        self.n_classes = n_classes
        self.data = data

    def __len__(self):
        return len(self.sequences_tensor)

    def __getitem__(self, idx):
        return self.sequences_tensor[idx], self.labels_tensor[idx]

    def label_histogram(self):
        """
        Plots a histogram showing the distribution of labels in the dataset.
        """
        label_counts = {i: 0 for i in range(self.n_classes)}
        for label in self.labels_tensor:
            label_counts[label.item()] += 1

        plt.bar(label_counts.keys(), label_counts.values())
        plt.xlabel('Label')
        plt.ylabel('Frequency')
        plt.title('Histogram of Labels')
        plt.show()

    def print_statistics(self):
        """
        Prints descriptive statistics of the dataset and plots the distributions of each variable.
        """
        print("Descriptive Statistics:")
        print(self.data.describe())

        print("\nDistributions of Variables:")
        for column in self.data.columns:
            if column not in ['utc_timestamp', 'NL_solar_generation_actual', 'NL_temperature']:
                plt.figure(figsize=(8, 6))
                plt.hist(self.data[column], bins=20, color='skyblue', edgecolor='black')
                plt.title(f"Distribution of {column}")
                plt.xlabel(column)
                plt.ylabel('Frequency')
                plt.show()


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


def extract_time_series_with_labels(dataframe, data_column_name, label_column_name, lookback_window, n_classes,
                                    threshold_mode="auto", manual_thresholds=None):
    """
    Extracts time series sequences and their corresponding labels from the DataFrame.

    Args:
        dataframe (pd.DataFrame): DataFrame containing the dataset.
        data_column_name (str): Name of the column containing time series data.
        label_column_name (str): Name of the column containing labels.
        lookback_window (int): Size of the window for each sequence.
        n_classes (int): Number of classes for label categorization.
        threshold_mode (str): Mode for setting temperature thresholds. Options are "auto" or "manual".
        manual_thresholds (list): List of manual thresholds for each class. Only used if threshold_mode is set to "manual".

    Returns:
        np.ndarray: Array of time series sequences.
        np.ndarray: Array of corresponding labels.
    """
    time_series_data = dataframe[data_column_name].values
    label_series_data = dataframe[label_column_name].values

    if threshold_mode == "manual":
        if len(manual_thresholds) != n_classes - 1:
            raise ValueError("Number of manual thresholds should be equal to number of classes minus one.")
        thresholds = manual_thresholds
    else:
        max_temp = np.max(label_series_data)
        min_temp = np.min(label_series_data)
        temp_range = max_temp - min_temp
        interval = temp_range / n_classes
        thresholds = [min_temp + interval * (i + 1) for i in range(n_classes - 1)]

    sequences = []
    labels = []
    all_means = []

    for i in range(len(time_series_data) - lookback_window + 1):
        window = time_series_data[i:i + lookback_window]
        avg_temp = np.mean(label_series_data[i:i + lookback_window])
        all_means.append(avg_temp)

        # Assign label based on temperature thresholds
        for idx, threshold in enumerate(thresholds):
            if avg_temp <= threshold:
                class_label = idx
                break
        else:
            class_label = n_classes - 1

        sequences.append(window)
        labels.append(class_label)

    sequences_array = np.array(sequences)
    labels_array = np.array(labels)
    all_means_mean = np.mean(all_means)
    all_means_std = np.std(all_means)
    print("Mean of all means across different days:", all_means_mean)
    print("Standard deviation of all means across different days:", all_means_std)
    return sequences_array, labels_array



if __name__ == '__main__':
    solar_power_file = 'data/open_power/solar_15min.csv'
    weather_file = 'data/open_power/weather_data.csv'
    solar_power_data = pd.read_csv(solar_power_file)
    weather_data = pd.read_csv(weather_file)
    data = join_resample_dataframes(solar_power_data, weather_data, "utc_timestamp")
    dataset = OpenPowerDataset(data, n_classes=3, threshold_mode="manual", manual_thresholds=[10, 11.5])
    dataset.label_histogram()
    dataset.print_statistics()
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

