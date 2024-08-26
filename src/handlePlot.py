import pandas as pd
from utils import mean_unique_triplets, mean_group_by, saveCsv, plot

df = pd.read_csv('processedDatasets/meanEmissions.csv')

meanData = mean_unique_triplets(
    df,
    "duration",
    "energy_consumed",
    "emissions",
    "emissions_rate",
    "cpu_power",
    "cpu_energy"
)

output_csv_path = 'meanEmissions.csv'
saveCsv(meanData, output_csv_path)

print("Dataset con le medie delle triplette uniche:")
print(meanData)

group_by_algorithm = mean_group_by(meanData, "algorithm", "duration", "energy_consumed")
group_by_dataset = mean_group_by(meanData, "dataset", "duration", "energy_consumed")
group_by_language = mean_group_by(meanData, "language", "duration", "energy_consumed")

print("\nMedia raggruppata per 'dataset':")
print(group_by_dataset)

print("\nMedia raggruppata per 'language':")
print(group_by_language)

print("\nMedia raggruppata per 'algorithm':")
print(group_by_algorithm)

# Specifica i percorsi di salvataggio per i grafici
graphics_folder = 'graphics/'
graphics_paths = {
    "language_duration": graphics_folder + "language_duration.png",
    "dataset_energy_consumed": graphics_folder + "dataset_energy_consumed.png",
    "algorithm_duration": graphics_folder + "algorithm_duration.png"
}

# 1. Istogrammi Unificati
plot(df, plotType="histogram", hist_cols=['duration', 'energy_consumed', 'emissions'],
     save_path='graphics/histograms.png')

# 2. Boxplot Unificati
plot(df, plotType="boxPlot", hist_cols=['duration', 'energy_consumed', 'emissions'], save_path='graphics/boxplots.png')

# 3. Heatmap della Matrice di Correlazione
df_numeric = pd.read_csv("processedDatasets/meanEmissions.csv")
plot(df_numeric, plotType="heatmap", save_path='graphics/correlation_matrix.png')

# 4. Pair Plot
plot(df, pairplot_cols=['duration', 'energy_consumed', 'emissions'], plotType="pairplot",
     save_path='graphics/pairplot.png')

# 5. Bar Plot
plot(group_by_language, x_axis='language', y_axis='duration', plotType="barPlot",
     save_path=graphics_paths["language_duration"])
plot(group_by_dataset, x_axis='dataset', y_axis='energy_consumed', plotType="barPlot",
     save_path=graphics_paths["dataset_energy_consumed"])
plot(group_by_algorithm, x_axis='algorithm', y_axis='duration', plotType="barPlot",
     save_path=graphics_paths["algorithm_duration"])
