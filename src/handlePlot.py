import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob


def plot(df, x_axis, y_axis, plotType="barPlot"):
    plt.figure(figsize=(10, 6))

    if plotType == "boxPlot":
        sns.boxplot(x=x_axis, y=y_axis, data=df)
    elif plotType == "barPlot":
        sns.barplot(x=x_axis, y=y_axis, data=df, ci=None)
    elif plotType == "violinPlot":
        sns.violinplot(x=x_axis, y=y_axis, data=df)
    else:
        raise ValueError(f"Invalid plot type '{plotType}'. Valid options are 'barPlot', 'boxPlot', or 'violinPlot'.")

    plt.title(f'{y_axis} by {x_axis}')
    plt.ylabel(f'{y_axis}')
    plt.xlabel(f'{x_axis}')
    plt.xticks(rotation=45)
    plt.show()


def retrive_data(save=True):
    base_directory = os.getcwd()
    csv_files = glob.glob(os.path.join(base_directory, '**/emissions_detailed.csv'), recursive=True)
    csv_files2 = glob.glob(os.path.join(base_directory, '**/emissions_detailed.csv'), recursive=True)

    df_list = []

    for file in csv_files:
        df = pd.read_csv(file)
        print(df.columns)
        df_list.append(df)

    for file in csv_files2:
        df = pd.read_csv(file)
        print(df.columns)
        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True)

    if save:
        merged_csv_path = os.path.join(base_directory, 'merged_emissions.csv')
        merged_df.to_csv(merged_csv_path, index=False)

    return merged_df


df = pd.read_csv('cpp/emissions/emissions_detailed.csv')
x_axis = "algorithm"
y_axis = "energy_consumed"

plot(df, x_axis, y_axis, "boxPlot")
df = retrive_data(save=True)
print(df.columns)
