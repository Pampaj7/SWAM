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


def plot(df, x_axis=None, y_axis=None, plotType="barPlot", save_path=None, hist_cols=None, pairplot_cols=None):
    """
    Funzione per generare vari tipi di grafici e salvarli in un file.

    Args:
    - df (DataFrame): Il dataset su cui basare i grafici.
    - x_axis (str): Nome della colonna da usare per l'asse x (per barPlot e boxPlot).
    - y_axis (str): Nome della colonna da usare per l'asse y (per barPlot e boxPlot).
    - plotType (str): Tipo di grafico da creare ('barPlot', 'boxPlot', 'violinPlot', 'histogram', 'pairplot').
    - save_path (str): Percorso del file in cui salvare il grafico.
    - hist_cols (list): Lista delle colonne da usare per gli istogrammi.
    - pairplot_cols (list): Lista delle colonne da usare per il pairplot.
    """
    if plotType == "histogram":
        # Plotting unificato per gli istogrammi
        plt.figure(figsize=(18, 6))
        for i, col in enumerate(hist_cols):
            plt.subplot(1, len(hist_cols), i + 1)
            plt.hist(df[col], bins=30, edgecolor='black')
            plt.title(f'Distribuzione di {col}')
            plt.xlabel(col)
            plt.ylabel('Frequenza')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    elif plotType == "boxPlot":
        # Plotting unificato per i boxplot
        plt.figure(figsize=(18, 6))
        for i, col in enumerate(hist_cols):
            plt.subplot(1, len(hist_cols), i + 1)
            sns.boxplot(x=df[col], color='skyblue' if i == 0 else 'orange' if i == 1 else 'green')
            plt.title(f'Distribuzione di {col}')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    elif plotType == "pairplot":
        # Pair plot
        sns.pairplot(df[pairplot_cols])
        if save_path:
            plt.savefig(save_path)
        plt.show()

    elif plotType == "heatmap":
        # Heatmap della matrice di correlazione
        numeric_cols = df.select_dtypes(include=['number']).columns
        correlation_matrix = df[numeric_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Matrice di Correlazione')
        if save_path:
            plt.savefig(save_path)
        plt.show()

    elif plotType == "barPlot":
        # Bar plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x=x_axis, y=y_axis, data=df, errorbar=None)
        plt.title(f'{y_axis} by {x_axis}')
        plt.ylabel(f'{y_axis}')
        plt.xlabel(f'{x_axis}')
        plt.xticks(rotation=45)
        if save_path:
            plt.savefig(save_path)
        plt.show()

    elif plotType == "violinPlot":
        # Violin plot
        plt.figure(figsize=(12, 8))
        sns.violinplot(x=x_axis, y=y_axis, data=df)
        plt.title(f'{y_axis} by {x_axis}')
        plt.ylabel(f'{y_axis}')
        plt.xlabel(f'{x_axis}')
        plt.xticks(rotation=45)
        if save_path:
            plt.savefig(save_path)
        plt.show()

    else:
        raise ValueError(
            f"Invalid plot type '{plotType}'. Valid options are 'barPlot', 'boxPlot', 'violinPlot', 'histogram', 'pairplot', 'heatmap'.")


def mean_unique_triplets(df: pd.DataFrame, *args: str):
    missing_columns = [col for col in args if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns not found in DataFrame: {', '.join(missing_columns)}")

    result = df.groupby(['algorithm', 'dataset', 'language'])[list(args)].mean().reset_index()

    return result


def saveCsv(df: pandas.DataFrame, name):
    df.to_csv(f"processedDatasets/{name}")


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
