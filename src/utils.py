import os
import glob

import pandas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def retrive_data(save=False):
    base_directory = os.getcwd()
    csv_files = glob.glob(os.path.join(base_directory, '**/emissions_detailed.csv'), recursive=True)
    df_list = []

    for file in csv_files:
        df = pd.read_csv(file)
        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True)

    if save:
        merged_csv_path = os.path.join(base_directory, 'raw_merged_emissions.csv')
        merged_df.to_csv(merged_csv_path, index=False)

    return merged_df


def plot(df, x_axis, y_axis, plotType="barPlot"):
    plt.figure(figsize=(10, 6))

    if plotType == "boxPlot":
        sns.boxplot(x=x_axis, y=y_axis, data=df)
    elif plotType == "barPlot":
        sns.barplot(x=x_axis, y=y_axis, data=df, errorbar=None)
    elif plotType == "violinPlot":
        sns.violinplot(x=x_axis, y=y_axis, data=df)
    else:
        raise ValueError(f"Invalid plot type '{plotType}'. Valid options are 'barPlot', 'boxPlot', or 'violinPlot'.")

    plt.title(f'{y_axis} by {x_axis}')
    plt.ylabel(f'{y_axis}')
    plt.xlabel(f'{x_axis}')
    plt.xticks(rotation=45)
    plt.show()


def mean_unique_triplets(df: pd.DataFrame, *args: str):

    missing_columns = [col for col in args if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns not found in DataFrame: {', '.join(missing_columns)}")

    result = df.groupby(['algorithm', 'dataset', 'language'])[list(args)].mean().reset_index()

    return result


def saveCsv(df: pandas.DataFrame, name):
    df.to_csv(f"processedDatasets/{name}")


import pandas as pd


def mean_group_by(df: pd.DataFrame, group_by: str, *args):

    # Check if group_by is valid
    if group_by not in ['algorithm', 'dataset', 'language']:
        raise ValueError("group_by must be one of 'algorithm', 'dataset', or 'language'")

    # Check if args are valid feature names
    for feature in args:
        if feature not in df.columns:
            raise ValueError(f"'{feature}' is not a valid column name in the DataFrame")

    # Group by the specified column and calculate the mean of specified features
    result_df = df.groupby(group_by)[list(args)].mean().reset_index()

    return result_df
