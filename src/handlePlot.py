import pandas as pd
from utils import *


csvFileName = "meanEmissionNew.csv"


df = pd.read_csv(f'processedDatasets/{csvFileName}')

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


