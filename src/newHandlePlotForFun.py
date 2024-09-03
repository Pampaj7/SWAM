import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df


def plot_execution_time_by_language_and_algorithm(df, language_col, algorithm_col, time_col, type_col, type_value,
                                                  title, excluded_language):
    if excluded_language:
        df = df[df[language_col] != excluded_language]
    filtered_df = df[df[type_col] == type_value]
    grouped_data = filtered_df.groupby([language_col, algorithm_col])[time_col].sum().unstack()
    grouped_data.plot(kind='bar', stacked=True, colormap='tab20', figsize=(12, 8))
    plt.title(title)
    plt.xlabel('Language')
    plt.ylabel('Total Execution Time')
    plt.savefig("graphics/" + title)
    plt.show()


def plot_co2_distribution_by_language(df, language_col, co2_col, title="CO₂_Emissions_Distribution_by_Language"):
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=language_col, y=co2_col, data=df, palette='Set3', hue='phase')
    plt.title(title)
    plt.xlabel('language')
    plt.ylabel('emissions')
    plt.savefig("graphics/" + title)
    plt.show()


def plot_duration_vs_emissions(df, duration_col, co2_col, title="Durata_vs_Emissioni_di_CO₂"):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=duration_col, y=co2_col, data=df, hue='language', palette='Set1')
    plt.title(title)
    plt.xlabel('Durata')
    plt.ylabel('Emissioni di CO₂')
    plt.savefig("graphics/" + title)
    plt.show()


def plot_co2_distribution_by_algorithm(df, algorithm_col, co2_col,
                                       title="Distribuzione_delle_Emissioni_di_CO₂_per_Algoritmo"):
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=algorithm_col, y=co2_col, data=df, palette='Set2', hue='phase')
    plt.title(title)
    plt.xlabel('Algoritmo')
    plt.ylabel('Emissioni di CO₂')
    plt.savefig("graphics/" + title)
    plt.show()


def plot_comparison_by_phase(df, duration_col, co2_col, phase_col, title="Confronto_tra_Fasi_(Test_vs_Train)"):
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=phase_col, y=duration_col, data=df, palette='Set1', hue='language')
    plt.title(f"Distribuzione della Durata per {title}")
    plt.xlabel('Fase')
    plt.ylabel('Durata')
    plt.show()
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=phase_col, y=co2_col, data=df, palette='Set2', hue='language')
    plt.title(f"Distribuzione delle Emissioni di CO₂ per {title}")
    plt.xlabel('Fase')
    plt.ylabel('Emissioni di CO₂')
    plt.savefig("graphics/" + title)
    plt.show()


def plot_temporal_trends(df, time_col, duration_col, co2_col, title="Tendenze_Temporali"):
    df[time_col] = pd.to_datetime(df[time_col])
    df.sort_values(by=time_col, inplace=True)
    plt.figure(figsize=(12, 8))
    plt.plot(df[time_col], df[duration_col], label='Durata', color='blue')
    plt.plot(df[time_col], df[co2_col], label='Emissioni di CO₂', color='green')
    plt.title(title)
    plt.xlabel('Data')
    plt.ylabel('Valori')
    plt.legend()
    plt.savefig("graphics/" + title)
    plt.show()


def plot_duration_by_algorithm_and_dataset(df, algorithm_col, duration_col, dataset_col,
                                           title="Durata_per_Algoritmo_diviso_per_Dataset"):
    plt.figure(figsize=(14, 8))
    palette = sns.color_palette('husl', n_colors=df[dataset_col].nunique())
    sns.barplot(data=df, x=algorithm_col, y=duration_col, hue=dataset_col, palette=palette)
    plt.title(title)
    plt.xlabel('Algoritmo')
    plt.ylabel('Durata')
    plt.legend(title=dataset_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("graphics/" + title)
    plt.show()


def plot_avg_co2_by_language(df, language_col, co2_col, title="Emissioni_di_CO₂_Medie_per_Linguaggio"):
    plt.figure(figsize=(12, 8))
    mean_co2 = df.groupby(language_col)[co2_col].mean()
    std_co2 = df.groupby(language_col)[co2_col].std()
    mean_co2.plot(kind='bar', yerr=std_co2, capsize=4, color='skyblue')
    plt.title(title)
    plt.xlabel('Linguaggio')
    plt.ylabel('Media delle Emissioni di CO₂')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("graphics/" + title)
    plt.show()


def plot_performance_heatmap(df, language_col, algorithm_col, duration_col,
                             title="Heatmap_delle_Performance_per_Linguaggio_e_Algoritmo"):
    pivot_table = df.pivot_table(values=duration_col, index=algorithm_col, columns=language_col, aggfunc='mean')
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt=".2f")
    plt.title(title)
    plt.xlabel('Linguaggio')
    plt.ylabel('Algoritmo')
    plt.tight_layout()
    plt.savefig("graphics/" + title)
    plt.show()


def plot_execution_time_by_dataset(df, dataset_col, time_col, language_col, title="Execution_Time_by_Dataset"):
    plt.figure(figsize=(12, 8))
    grouped_data = df.groupby([dataset_col, language_col])[time_col].sum().unstack()
    grouped_data.plot(kind='bar', stacked=True, colormap='tab20', figsize=(12, 8))
    plt.title(title)
    plt.xlabel('Dataset')
    plt.ylabel('Total Execution Time')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("graphics/" + title)
    plt.show()


def plot_execution_time_by_phase(df, dataset_col, time_col, phase_col, title="Execution_Time_by_Train_and_Test_Phases"):
    plt.figure(figsize=(12, 8))

    grouped_data = df.groupby([dataset_col, phase_col])[time_col].sum().unstack()
    grouped_data.plot(kind='bar', stacked=False, colormap='Set2', figsize=(12, 8))

    plt.title(title)
    plt.xlabel('Dataset')
    plt.ylabel('Total Execution Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("graphics/" + title)
    plt.show()


def plot_correlation_cpu_energy_emissions(df, cpu_energy_col, emissions_col,
                                          title="Correlation_between_CPU_Energy_and_Emissions"):
    plt.figure(figsize=(10, 6))

    # Scatter plot
    sns.scatterplot(x=df[cpu_energy_col], y=df[emissions_col], color='blue', s=100)

    # Linea di tendenza (trendline)
    sns.regplot(x=df[cpu_energy_col], y=df[emissions_col], scatter=False, color='red', line_kws={"linewidth": 2})

    plt.title(title)
    plt.xlabel('CPU Energy (Joules)')
    plt.ylabel('Emissions (CO₂)')

    # Calcolo del coefficiente di correlazione
    correlation = df[[cpu_energy_col, emissions_col]].corr().iloc[0, 1]
    plt.annotate(f'Correlation: {correlation:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, color='red')

    plt.tight_layout()
    plt.savefig("graphics/" + title)
    plt.show()


def plot_multiple_correlations(df, columns, title="Multiple_Correlations"):
    sns.pairplot(df[columns], kind='reg', plot_kws={'line_kws': {'color': 'red'}, 'scatter_kws': {'alpha': 0.5}})

    plt.suptitle(title, y=1.02)  # Adjust the title position
    plt.tight_layout()
    plt.savefig("graphics/" + title)
    plt.show()


# we can use this function to remove outliers from the dataset and return a new dataset without outliers
def remove_outliers(df, threshold=3):
    # threshold (float): Il valore z-score sopra il quale un punto dati è considerato un outlier.

    z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
    mask = (z_scores < threshold).all(axis=1)
    df_clean = df[mask]
    return df_clean


def benchmark_by_language(df, language_col, algorithm_col, dataset_col, metrics, title="Benchmark_by_Language"):
    # Raggruppamento dei dati per linguaggio, algoritmo e dataset
    grouped_df = df.groupby([language_col, algorithm_col, dataset_col])[metrics].mean().reset_index()

    # Creazione del plot per ogni metrica
    for metric in metrics:
        plt.figure(figsize=(14, 8))
        sns.barplot(x=algorithm_col, y=metric, hue=language_col, data=grouped_df, errorbar=None, palette="Set3")
        plt.title(f"{title}: {metric.capitalize()}")
        plt.xlabel('Algorithm')
        plt.ylabel(metric.capitalize())
        plt.legend(title='Language')
        plt.savefig("graphics/" + metric + title)
        plt.show()


def plot_emissions_by_algorithm(df, algorithm_col, emissions_col, title="Emissions_by_Algorithm"):
    # Calcolo delle emissioni medie per algoritmo
    emissions_summary = df.groupby(algorithm_col)[emissions_col].mean().sort_values()

    # Creazione del grafico a barre
    plt.figure(figsize=(12, 8))
    sns.barplot(x=emissions_summary.index, y=emissions_summary.values, palette='viridis')
    plt.title(title)
    plt.xlabel('Algorithm')
    plt.ylabel('Average Emissions')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("graphics/" + title)
    plt.show()


def plot_emissions_rate(df, algorithm_col='algorithm', dataset_col='dataset', phase_col='phase',
                        emissions_rate_col='emissions_rate'):
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df, x=algorithm_col, y=emissions_rate_col, hue=phase_col, palette='Set2')
    plt.title("Emissions Rate by Algorithm and Phase")
    plt.xlabel("Algorithm")
    plt.ylabel("Emissions Rate (kg CO2/sec)")
    plt.legend(title="Phase")
    plt.xticks(rotation=45)
    plt.savefig("graphics/Emissions_Rate_by_Algorithm_and_Phase")
    plt.show()


df = load_dataset('processedDatasets/meanEmissionsNew.csv')
df_raw = load_dataset('raw_merged_emissions.csv')
df_clean = remove_outliers(df)
"""
# u can exclude a language by choice
plot_execution_time_by_language_and_algorithm(df, 'language', 'algorithm', 'duration', 'phase', 'test',
                                              title="Execution_Time_Distribution_by_Language_and_Algorithm_(Test)",
                                              excluded_language="java")

plot_execution_time_by_language_and_algorithm(df, 'language', 'algorithm', 'duration', 'phase', 'train',
                                              title="Execution_Time_Distribution_by_Language_and_Algorithm_(Train)",
                                              excluded_language="java")

plot_co2_distribution_by_language(df, 'language', 'emissions')

plot_duration_vs_emissions(df, 'duration', 'emissions')

plot_co2_distribution_by_algorithm(df, 'algorithm', 'emissions')

plot_comparison_by_phase(df, 'duration', 'emissions', 'phase')

# need to refine also use the original dataset with the timestamp!!!
# plot_temporal_trends(df_raw, 'timestamp', 'duration', 'emissions')

plot_duration_by_algorithm_and_dataset(df, 'algorithm', 'duration', 'dataset')

plot_avg_co2_by_language(df, 'language', 'emissions')
plot_performance_heatmap(df, 'language', 'algorithm', 'duration')

plot_execution_time_by_dataset(df, 'dataset', 'duration', language_col='language')

plot_execution_time_by_phase(df, 'dataset', 'duration', 'phase', )

plot_correlation_cpu_energy_emissions(df, 'cpu_energy', 'emissions')

# removed the noob outlier from nicco dumb pc
columns_to_correlate = ['cpu_energy', 'emissions', 'energy_consumed', 'duration']
plot_multiple_correlations(df_clean, columns_to_correlate)

benchmark_by_language(df_clean,
                      language_col='language',
                      algorithm_col='algorithm',
                      dataset_col='dataset',
                      metrics=['duration', 'energy_consumed', 'emissions', 'cpu_energy', 'emissions_rate', 'cpu_power'])

plot_emissions_by_algorithm(df, 'algorithm', 'emissions')
"""

plot_emissions_rate(df)
