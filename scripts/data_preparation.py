import pandas as pd
from pandas import DataFrame

from config import SOLAR_POWER_FILE, WEATHER_FILE, COLUMN_NAME

weather_data_path = 'data/open_power/weather_data.csv'
power_data_path = 'data/open_power/solar_15min.csv'
time_col = 'utc_timestamp'


def categorize_columns_with_prefix(df: DataFrame, prefix: str, time_column: str, num_bins: int) -> DataFrame:
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


def group_column_into_bins(df: DataFrame, column_name: str, num_bins: int) -> DataFrame:
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


def classify_into_above_below_median_bins(df: DataFrame, column_name: str, aggregation: str = 'mean') -> DataFrame:
    """
    Group the values in the specified column of the DataFrame based on specified aggregation.
    Parameters:
    df (DataFrame): The input DataFrame.
    column_name (str): The name of the column to group.
    aggregation (str): Aggregation method. Can be 'mean', 'median', etc.
    Returns:
    DataFrame: A new DataFrame with the column values grouped into bins.
    """
    df = df.copy()
    column_values = df[column_name]

    if aggregation == 'mean':
        threshold = column_values.mean()
    elif aggregation == 'median':
        threshold = column_values.median()
    else:
        raise ValueError("Aggregation method not supported")

    bin_edges = (column_values > threshold).astype(int)
    binned_df = pd.DataFrame({column_name: bin_edges}, index=df.index)
    return binned_df


def one_hot_encode_columns(df: DataFrame) -> DataFrame:
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
    df_weather = pd.read_csv(WEATHER_FILE)
    df_weather = categorize_columns_with_prefix(df_weather, 'NL', time_col, 5)
    df_power = pd.read_csv(SOLAR_POWER_FILE)
    print(df_weather.columns)
    print(one_hot_encode_columns(df_weather))
