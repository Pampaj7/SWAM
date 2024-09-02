import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


def plot_co2_distribution_by_language(df, language_col, co2_col, title="CO₂ Emissions Distribution by Language"):
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=language_col, y=co2_col, data=df, palette='Set3')
    plt.title(title)
    plt.xlabel('language')
    plt.ylabel('emissions')
    plt.savefig("graphics/" + title)
    plt.show()


def plot_duration_vs_emissions(df, duration_col, co2_col, title="Durata vs Emissioni di CO₂"):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=duration_col, y=co2_col, data=df, hue='language', palette='Set1')
    plt.title(title)
    plt.xlabel('Durata')
    plt.ylabel('Emissioni di CO₂')
    plt.savefig("graphics/" + title)
    plt.show()


def plot_co2_distribution_by_algorithm(df, algorithm_col, co2_col,
                                       title="Distribuzione delle Emissioni di CO₂ per Algoritmo"):
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=algorithm_col, y=co2_col, data=df, palette='Set2')
    plt.title(title)
    plt.xlabel('Algoritmo')
    plt.ylabel('Emissioni di CO₂')
    plt.savefig("graphics/" + title)
    plt.show()


def plot_comparison_by_phase(df, duration_col, co2_col, phase_col, title="Confronto tra Fasi (Test vs Train)"):
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=phase_col, y=duration_col, data=df, palette='Set1')
    plt.title(f"Distribuzione della Durata per {title}")
    plt.xlabel('Fase')
    plt.ylabel('Durata')
    plt.show()
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=phase_col, y=co2_col, data=df, palette='Set2')
    plt.title(f"Distribuzione delle Emissioni di CO₂ per {title}")
    plt.xlabel('Fase')
    plt.ylabel('Emissioni di CO₂')
    plt.savefig("graphics/" + title)
    plt.show()


def plot_temporal_trends(df, time_col, duration_col, co2_col, title="Tendenze Temporali"):
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
                                           title="Durata per Algoritmo diviso per Dataset"):
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


def plot_avg_co2_by_language(df, language_col, co2_col, title="Emissioni di CO₂ Medie per Linguaggio"):
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
                             title="Heatmap delle Performance per Linguaggio e Algoritmo"):
    pivot_table = df.pivot_table(values=duration_col, index=algorithm_col, columns=language_col, aggfunc='mean')
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt=".2f")
    plt.title(title)
    plt.xlabel('Linguaggio')
    plt.ylabel('Algoritmo')
    plt.tight_layout()
    plt.savefig("graphics/" + title)
    plt.show()


df = load_dataset('processedDatasets/meanEmissionsNew.csv')
df_raw = load_dataset('raw_merged_emissions.csv')

#u can exclude a language by choice
plot_execution_time_by_language_and_algorithm(df, 'language', 'algorithm', 'duration', 'phase', 'test',
                                              title="Execution Time Distribution by Language and Algorithm (Test)",
                                              excluded_language="java")

plot_execution_time_by_language_and_algorithm(df, 'language', 'algorithm', 'duration', 'phase', 'train',
                                              title="Execution Time Distribution by Language and Algorithm (Train)",
                                              excluded_language="java")

plot_co2_distribution_by_language(df, 'language', 'emissions', title="CO₂ Emissions Distribution by Language")

plot_duration_vs_emissions(df, 'duration', 'emissions')

plot_co2_distribution_by_algorithm(df, 'algorithm', 'emissions')

plot_comparison_by_phase(df, 'duration', 'emissions', 'phase')

# need to refine
# plot_temporal_trends(df_raw, 'timestamp', 'duration', 'emissions')

plot_duration_by_algorithm_and_dataset(df, 'algorithm', 'duration', 'dataset')

plot_avg_co2_by_language(df, 'language', 'emissions')
plot_performance_heatmap(df, 'language', 'algorithm', 'duration')
