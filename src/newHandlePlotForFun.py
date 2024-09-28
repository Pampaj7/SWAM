import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr, kendalltau


def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df


def plot_execution_time_by_language_and_algorithm(df, language_col, algorithm_col, time_col, type_col, type_value,
                                                  title, excluded_language):
    if excluded_language:
        df = df[df[language_col] != excluded_language]
    filtered_df = df[df[type_col] == type_value]
    grouped_data = filtered_df.groupby([language_col, algorithm_col])[time_col].sum().unstack()
    grouped_data['total'] = grouped_data.sum(axis=1)
    grouped_data = grouped_data.sort_values(by='total', ascending=True)
    grouped_data = grouped_data.drop(columns='total')
    grouped_data.plot(kind='bar', stacked=True, colormap='tab20', figsize=(12, 8))
    plt.title(title)
    plt.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    plt.xlabel('Language')
    plt.grid(True)
    plt.ylabel('CPU Energy (Joules)')
    plt.savefig("graphics/" + title)
    plt.show()


def plot_duration_vs_emissions(df, duration_col, co2_col, title="Durata_vs_Emissioni_di_CO₂"):
    # Calcolo dei quartili
    Q1 = df[co2_col].quantile(0.25)
    Q3 = df[co2_col].quantile(0.75)
    IQR = Q3 - Q1

    # Filtrare i dati per rimuovere gli outlier
    df_filtered = df[~((df[co2_col] < (Q1 - 1.5 * IQR)) | (df[co2_col] > (Q3 + 1.5 * IQR)))]

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=duration_col, y=co2_col, data=df_filtered, hue='language', palette='tab20')
    plt.grid(True)
    plt.title("Duration vs Emissions (Outlier Removed)")
    plt.xlabel('CPU Energy (Joules)')
    plt.ylabel('Emissions KG(CO₂)')
    plt.savefig("graphics/" + title)
    plt.show()


def plot_duration_by_algorithm_and_dataset(df, algorithm_col, duration_col, dataset_col,
                                           title="Durata_per_Algoritmo_diviso_per_Dataset"):
    plt.figure(figsize=(14, 8))
    palette = sns.color_palette('tab20', n_colors=df[dataset_col].nunique())
    sns.barplot(data=df, x=algorithm_col, y=duration_col, hue=dataset_col, palette=palette,
                ci=None)  # Rimuovi le barre di errore
    plt.title("Duration by Algorithm divided by Dataset")
    plt.xlabel('Algorithms')
    plt.ylabel('Total Execution Time(s)')
    plt.legend(title=dataset_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("graphics/" + title)
    plt.show()


def plot_performance_heatmap(df, language_col, algorithm_col, duration_col,
                             title="Heatmap_delle_Performance_per_Linguaggio_e_Algoritmo"):
    # Create figure and axes with vertical subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 16), sharex=True, sharey=True)

    for ax, phase in zip(axes, ['train', 'test']):
        # Filter for the phase
        phase_df = df[df['phase'] == phase]

        # Pivot table for heatmap
        pivot_table = phase_df.pivot_table(values=duration_col, index=algorithm_col, columns=language_col,
                                           aggfunc='mean')
        pivot_table *= 1000  # Convert seconds to milliseconds



        #pivot_table_log = np.log10(pivot_table)

        # Plot heatmap
        sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="viridis", ax=ax,
                    cbar_kws={'label': 'Milliseconds'})

        # Customize the axis
        ax.set_title(f"Performance Heatmap for Phase: {phase}")
        ax.set_xlabel('Languages')
        ax.set_ylabel('Algorithms')

    plt.tight_layout()
    # Save the plot as an image file
    plt.savefig(f"graphics/{title}_log.png")

    # Show the plot
    plt.show()


def plot_execution_time_by_dataset(df, dataset_col, time_col, language_col, title="Execution_Time_by_Dataset"):
    plt.figure(figsize=(12, 8))
    grouped_data = df.groupby([dataset_col, language_col])[time_col].sum().unstack()
    grouped_data.plot(kind='bar', stacked=True, colormap='tab20', figsize=(12, 8))
    plt.title("Total execution Time by Dataset divided by Language")
    plt.xlabel('Dataset')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.ylabel('Total Execution Time(s)')
    plt.savefig("graphics/" + title)
    plt.show()


def plot_correlation_by_phase(df, cpu_energy_col, emissions_col, phase_col, language_col,
                              title_prefix="Correlation_by_Phase"):
    phases = df[phase_col].unique()

    fig, axes = plt.subplots(nrows=len(phases), ncols=1, figsize=(10, 5 * len(phases)), sharex=True)

    for ax, phase in zip(axes, phases):
        phase_df = df[df[phase_col] == phase]

        sns.scatterplot(x=phase_df[cpu_energy_col], y=phase_df[emissions_col], hue=phase_df[language_col],
                        palette='tab20', s=50, alpha=0.4, ax=ax)

        # Aggiunta di una regressione polinomiale (grado 2) opzionale
        sns.regplot(x=phase_df[cpu_energy_col], y=phase_df[emissions_col], scatter=False, color='green', order=2,
                    line_kws={"linewidth": 1.5, "linestyle": "--"}, ax=ax)

        # Calcolo delle correlazioni
        pearson_corr = phase_df[[cpu_energy_col, emissions_col]].corr().iloc[0, 1]
        spearman_corr, _ = spearmanr(phase_df[cpu_energy_col], phase_df[emissions_col])
        kendall_corr, _ = kendalltau(phase_df[cpu_energy_col], phase_df[emissions_col])

        # Calcolo di R^2
        X = phase_df[[cpu_energy_col]]
        y = phase_df[emissions_col]
        model = LinearRegression().fit(X, y)
        r_squared = model.score(X, y)

        ax.set_title(f'{phase} Phase')
        ax.set_xlabel('CPU Energy (Joules)')
        if emissions_col == 'emissions':
            ax.set_ylabel('Emissions Kg(CO₂)')
        if emissions_col == 'duration':
            ax.set_ylabel('Duration (s)')
        ax.annotate(f'Pearson: {pearson_corr:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10, color='red')
        ax.annotate(f'Spearman: {spearman_corr:.2f}', xy=(0.05, 0.90), xycoords='axes fraction', fontsize=10,
                    color='blue')
        ax.annotate(f'Kendall: {kendall_corr:.2f}', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=10,
                    color='green')
        ax.annotate(f'R²: {r_squared:.2f}', xy=(0.05, 0.80), xycoords='axes fraction', fontsize=10, color='purple')

        if phase_df[cpu_energy_col].min() > 0:
            ax.set_xscale('log')
        if phase_df[emissions_col].min() > 0:
            ax.set_yscale('log')

        ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)

    fig.tight_layout()
    plt.legend(title=language_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(f"graphics/{title_prefix}_Train_vs_Test.png")
    plt.show()


# we can use this function to remove outliers from the dataset and return a new dataset without outliers
def remove_outliers(df, threshold=3):
    # threshold (float): Il valore z-score sopra il quale un punto dati è considerato un outlier.

    z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
    mask = (z_scores < threshold).all(axis=1)
    df_clean = df[mask]
    return df_clean


def plot_emissions_by_algorithm(df, algorithm_col, emissions_col, title="Emissions_by_Algorithm"):
    # Calcolo delle emissioni medie per algoritmo
    emissions_summary = df.groupby(algorithm_col)[emissions_col].mean().sort_values()

    # Creazione del grafico a barre
    plt.figure(figsize=(12, 8))
    sns.barplot(x=emissions_summary.index, y=emissions_summary.values, palette='tab20')
    plt.title("Average Emissions by Algorithm")
    plt.xlabel('Algorithm')
    plt.ylabel('Average Emissions (kg CO₂)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("graphics/" + title)
    plt.show()


def plot_time_series(df, language, dataset, algorithm, date_col, value_col, phase_col='phase'):
    # Filter the DataFrame based on selected language, dataset, and algorithm
    df_filtered = df[(df['language'] == language) &
                     (df['dataset'] == dataset) &
                     (df['algorithm'] == algorithm)].copy()  # Create a copy to avoid SettingWithCopyWarning

    # Check if filtering resulted in non-empty DataFrame
    if df_filtered.empty:
        print(f"No data available for language: {language}, dataset: {dataset}, algorithm: {algorithm}")
        return

    # Convert the timestamp column to datetime
    df_filtered[date_col] = pd.to_datetime(df_filtered[date_col], errors='coerce')

    # Check for any conversion errors or NaT values
    if df_filtered[date_col].isna().any():
        print("There are invalid dates in the timestamp column.")

    # Drop rows with invalid dates
    df_filtered = df_filtered.dropna(subset=[date_col])

    # Ensure the value column contains numeric data
    df_filtered[value_col] = pd.to_numeric(df_filtered[value_col], errors='coerce')

    # Drop rows with invalid or missing values in the value column
    df_filtered = df_filtered.dropna(subset=[value_col])

    # Check if there's still data to plot
    if df_filtered.empty:
        print("No valid data available for plotting.")
        return

    # Plot the time series
    plt.figure(figsize=(14, 8))

    # Separate the data into train and test phases
    # Separate the data into train and test phases
    for phase in df_filtered[phase_col].unique():
        df_phase = df_filtered[df_filtered[phase_col] == phase]

        plt.plot(df_phase[date_col], df_phase[value_col], marker='o', linestyle='-',
                 label=f'{phase} phase')

    plt.title(f'Emissions Over Time for {algorithm} in {language} on Dataset {dataset}')
    plt.xlabel('Date')
    plt.ylabel('Emissions (kg CO2)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend(title='Phase')
    plt.tight_layout()

    # Save the plot as an image file
    plt.savefig(f'graphics/{algorithm}_emissions_time_series_{language}_{dataset}.png')

    # Show the plot
    plt.show()


def plot_time_series_all_algorithms(df, language, date_col, value_col, phase_col='phase', phase='train', dataset="breastCancer"):
    """
    Plot time series data for all algorithms for a specific language and dataset in a given phase.

    Parameters:
    - df: DataFrame containing the data
    - language: The programming language to filter
    - dataset: The dataset to filter
    - date_col: Column name for date/time
    - value_col: Column name for the values to plot
    - phase_col: Column name for phase (optional, default is 'phase')
    - phase: Phase to filter (optional, default is 'test')
    """
    # Filter the DataFrame based on the selected language, dataset, and phase
    df_filtered = df[(df['language'] == language) &
                     (df['dataset'] == dataset) &
                     (df[phase_col] == phase)].copy()  # Create a copy to avoid SettingWithCopyWarning

    # Check if filtering resulted in non-empty DataFrame
    if df_filtered.empty:
        print(f"No data available for language: {language}, dataset: {dataset} in phase: {phase}")
        return

    # Convert the timestamp column to datetime
    df_filtered[date_col] = pd.to_datetime(df_filtered[date_col], errors='coerce')

    # Check for any conversion errors or NaT values
    if df_filtered[date_col].isna().any():
        print("There are invalid dates in the timestamp column.")

    # Drop rows with invalid dates
    df_filtered = df_filtered.dropna(subset=[date_col])

    # Ensure the value column contains numeric data
    df_filtered[value_col] = pd.to_numeric(df_filtered[value_col], errors='coerce')

    # Drop rows with invalid or missing values in the value column
    df_filtered = df_filtered.dropna(subset=[value_col])

    # Check if there's still data to plot
    if df_filtered.empty:
        print("No valid data available for plotting.")
        return

    # Plot the time series
    plt.figure(figsize=(14, 8))

    # Get unique algorithms
    algorithms = df_filtered['algorithm'].unique()

    # Generate a color palette for the algorithms
    palette = sns.color_palette("tab20", n_colors=len(algorithms))

    # Plot each algorithm
    for algorithm, color in zip(algorithms, palette):
        df_algo = df_filtered[df_filtered['algorithm'] == algorithm]

        plt.plot(df_algo[date_col], df_algo[value_col], marker='o', linestyle='-', color=color,
                 label=f'{algorithm}')

    plt.title(f'Emissions Over Time for All Algorithms in {language} on Dataset {dataset} (Phase: {phase})')
    plt.xlabel('Date')
    plt.ylabel('Emissions (kg CO2)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend(title='Algorithm')
    plt.tight_layout()

    # Save the plot as an image file
    plt.savefig(f'graphics/emissions_time_series_{language}_{dataset}_{phase}.png')

    # Show the plot
    plt.show()


def plot_execution_time_by_language(df, language_col, algorithm_col, time_col, type_col,
                                    selected_algorithm, title):
    # Filter the DataFrame based on the selected algorithm
    filtered_df = df[df[algorithm_col] == selected_algorithm]

    if filtered_df.empty:
        print(f"No data available for algorithm: {selected_algorithm}")
        return

    # Create subplots for train and test phases
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 16), sharex=True)

    # Plot for the Train phase
    train_df = filtered_df[filtered_df[type_col] == 'train']
    if not train_df.empty:
        grouped_train = train_df.groupby([language_col, 'dataset'])[time_col].sum().unstack()
        grouped_train.plot(kind='bar', stacked=True, colormap='tab20', ax=axes[0])
        axes[0].set_title(f'{title} - Train Phase for Algorithm: {selected_algorithm}')
        axes[0].set_ylabel('CPU Energy (Joules)')
        axes[0].grid(True)

    # Plot for the Test phase
    test_df = filtered_df[filtered_df[type_col] == 'test']
    if not test_df.empty:
        grouped_test = test_df.groupby([language_col, 'dataset'])[time_col].sum().unstack()
        grouped_test.plot(kind='bar', stacked=True, colormap='tab20', ax=axes[1])
        axes[1].set_title(f'{title} - Test Phase for Algorithm: {selected_algorithm}')
        axes[1].set_ylabel('CPU Energy (Joules)')
        axes[1].grid(True)

    axes[1].set_xlabel('Language')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot as an image file
    plt.savefig(f'graphics/{title}_{selected_algorithm}.png')

    # Show the plot
    plt.show()


def plot_time_series_subplot(df, combinations, date_col, value_col, phase_col='phase'):
    """
    Plot time series data for multiple combinations in a grid of subplots, handling different scales.

    Parameters:
    - df: DataFrame containing the data
    - combinations: List of tuples with (language, dataset, algorithm)
    - date_col: Column name for date/time
    - value_col: Column name for the values to plot
    - phase_col: Column name for phase (optional, default is 'phase')
    """
    num_combinations = len(combinations)
    num_cols = 2  # Number of columns for subplots
    num_rows = (num_combinations + num_cols - 1) // num_cols  # Calculate number of rows needed

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows), sharex=True)
    axes = axes.flatten()  # Flatten the 2D array of axes to make indexing easier

    for idx, (language, dataset, algorithm) in enumerate(combinations):
        # Filter the DataFrame based on the current combination
        df_filtered = df[(df['language'] == language) &
                         (df['dataset'] == dataset) &
                         (df['algorithm'] == algorithm)].copy()  # Create a copy to avoid SettingWithCopyWarning

        if df_filtered.empty:
            print(f"No data available for language: {language}, dataset: {dataset}, algorithm: {algorithm}")
            continue

        # Convert the timestamp column to datetime
        df_filtered[date_col] = pd.to_datetime(df_filtered[date_col], errors='coerce')

        # Drop rows with invalid dates
        df_filtered = df_filtered.dropna(subset=[date_col])

        # Ensure the value column contains numeric data
        df_filtered[value_col] = pd.to_numeric(df_filtered[value_col], errors='coerce')

        # Drop rows with invalid or missing values in the value column
        df_filtered = df_filtered.dropna(subset=[value_col])

        if df_filtered.empty:
            print(f"No valid data available for language: {language}, dataset: {dataset}, algorithm: {algorithm}")
            continue

        # Plot the time series
        ax = axes[idx]
        for phase in df_filtered[phase_col].unique():
            df_phase = df_filtered[df_filtered[phase_col] == phase]
            ax.plot(df_phase[date_col], df_phase[value_col], marker='o', linestyle='-', label=f'{phase} phase')

        ax.set_title(f'{algorithm} in {language} on Dataset {dataset}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Emissions (kg CO2)')
        ax.grid(True)
        ax.legend(title='Phase')

        # Set individual y-axis limits if necessary
        y_min, y_max = df_filtered[value_col].min(), df_filtered[value_col].max()
        ax.set_ylim(y_min, y_max)

    # Remove any unused subplots
    for ax in axes[num_combinations:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.xticks(rotation=45)

    # Save the plot as an image file
    plt.savefig(f'graphics/time_series_subplot.png')
    plt.show()


df = load_dataset('processedDatasets/meanEmissionsNew.csv')
df_raw = load_dataset('raw_merged_emissions.csv')
df_clean = remove_outliers(df)

"""plot_execution_time_by_language_and_algorithm(df, 'language', 'algorithm', 'cpu_energy', 'phase', 'test',
                                              title="CPU energy distribution by language and algorithm (Test)",
                                              excluded_language='None')

plot_execution_time_by_language_and_algorithm(df, 'language', 'algorithm', 'cpu_energy', 'phase', 'train',
                                              title="CPU energy distribution by language and algorithm (Train)",
                                              excluded_language='None')
"""
plot_performance_heatmap(df, 'language', 'algorithm', 'duration', 'phase', )

"""plot_execution_time_by_dataset(df, 'dataset', 'emissions', 'language')

plot_correlation_by_phase(df, 'cpu_energy', 'emissions', 'phase', 'language',
                          title_prefix="Correlation_by_Phase_emissions")

plot_correlation_by_phase(df, 'cpu_energy', 'duration', 'phase', 'language', title_prefix="Correlation_by_Phase_time")

plot_correlation_by_phase(df_raw, 'cpu_energy', 'emissions', 'phase', 'language',
                          title_prefix="Correlation_by_Phase_emissions_raw")

plot_correlation_by_phase(df_raw, 'cpu_energy', 'duration', 'phase', 'language',
                          title_prefix="Correlation_by_Phase_time_raw")

plot_time_series(df_raw, 'matlab', 'breastCancer', 'randomForest', 'timestamp', 'cpu_energy')

plot_execution_time_by_language(df, 'language', 'algorithm', 'cpu_energy', 'phase', "KNN",
                                "CPU energy "
                                "distribution by language")

plot_execution_time_by_language(df, 'language', 'algorithm', 'cpu_energy', 'phase', "decisionTree",
                                "CPU energy "
                                "distribution by language")



plot_time_series_all_algorithms(df_raw, 'cpp', 'timestamp', 'emissions')"""