import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

weather_data_path = 'data/open_power/weather_data.csv'
power_data_path = 'data/open_power/solar_15min.csv'
time_col = 'utc_timestamp'

# Time series

# [0.1, 02, ......] 96 values: 1 Day of solar power production - 4*24
# [1, 3 , 1] 3 - 6
# [
#
# [1, 2, 3, 4] #temperature in 4 bins temperatur bin1
# [cc


class OpenPowerDataset(Dataset):
    def __init__(self, combined_df, scaler_feature, scaler_condition, transform=None):
        self.df = combined_df
        # self.features = torch.tensor(scaler_feature.fit_transform(combined_df[["NL_solar_generation_actual"]]), dtype=torch.float32)
        # self.conditions = torch.tensor(
        #     scaler_condition.fit_transform(combined_df[['weather_conditions']]), dtype=torch.float32)
        # self.

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.conditions[idx], self.features[idx]


def categorize_columns_with_prefix(df, prefix, time_column, num_bins):
    """
    Select columns with the specified prefix from the DataFrame and apply group_into_bins function to each.
    Parameters:
    df (DataFrame): The input DataFrame.
    prefix (str): The prefix to filter columns.
    num_bins (int): The number of bins to create.
    Returns:
    DataFrame: A new DataFrame with selected columns categorized into bins.
    """
    df = df.copy()
    selected_columns = [col for col in df.columns if col.startswith(prefix)]
    categorized_df = pd.DataFrame(index=df.index)
    for col in selected_columns:
        binned_col = group_column_into_bins(df, col, num_bins)
        categorized_df[col] = binned_col
    categorized_df[time_column] = df[time_column]
    categorized_df.set_index(time_column, inplace=True)
    return categorized_df


def group_column_into_bins(df, column_name, num_bins):
    """
    Group the values in the specified column of the DataFrame into the specified number of bins.
    Parameters:
    df (DataFrame): The input DataFrame.
    column_name (str): The name of the column to group.
    num_bins (int): The number of bins to create.
    Returns:
    DataFrame: A new DataFrame with the column values grouped into bins.
    """
    df = df.copy()
    column_values = df[column_name]
    bin_edges = pd.cut(column_values, bins=num_bins, labels=False) + 1
    binned_df = pd.DataFrame({column_name: bin_edges}, index=df.index)
    return binned_df


# def load_and_preprocess_data(solar_data_path, weather_data_path):
#     solar_df = pd.read_csv(solar_data_path)
#     weather_df = pd.read_csv(weather_data_path)
#
#     combined_df = pd.merge(solar_df, weather_df, on='utc_timestamp')
#     # print(combined_df['NL_solar_generation_actual'])
#
#     # Normalize data
#     scaler_feature = MinMaxScaler(feature_range=(0, 1))
#     scaler_condition = MinMaxScaler(feature_range=(0, 1))
#
#     # Here add train/test split
#
#     dataset = SolarWeatherDataset(combined_df, scaler_feature, scaler_condition)
#     loader = DataLoader(dataset, batch_size=32, shuffle=True)
#
#     return loader

def one_hot_encode_columns(df):
    """
    Perform one-hot encoding on each column of the DataFrame.
    Parameters:
    df (DataFrame): The input DataFrame.
    Returns:
    DataFrame: A new DataFrame with each column one-hot encoded.
    """
    encoded_columns = []
    for col in df.columns:
        encoded_col = pd.get_dummies(df[col], prefix=col)
        encoded_columns.append(encoded_col)
    encoded_df = pd.concat(encoded_columns, axis=1)
    return encoded_df

if __name__ == '__main__':
    df_weather = pd.read_csv(weather_data_path)
    df_weather = categorize_columns_with_prefix(df_weather, 'NL', time_col, 5)
    print(df_weather)
    print(one_hot_encode_columns(df_weather))

