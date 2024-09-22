import pandas as pd
from utils import mean_unique_quadruplets, mean_group_by, saveCsv, plot, median_unique_quadruplets, retrive_data, \
    median_group_by, sum_metrics_per_language, sum_metrics_per_language_and_algorithm

output_csv_name = 'meanEmissionsNew.csv'
columns = ["duration", "energy_consumed", "emissions", "emissions_rate", "cpu_power", "cpu_energy"]

df = retrive_data(True)
meanData = mean_unique_quadruplets(df, columns)

saveCsv(meanData, output_csv_name)

print("Dataset con le medie dei quartetti unici:")
print(meanData)

group_by_algorithm = mean_group_by(meanData, "algorithm", columns)
group_by_dataset = mean_group_by(meanData, "dataset", columns)
group_by_language = mean_group_by(meanData, "language", columns)

saveCsv(group_by_language, "mean_group_by_language.csv")
saveCsv(group_by_algorithm, "mean_group_by_algorithm.csv")
saveCsv(group_by_dataset, "mean_group_by_dataset.csv")

print("\nMedia raggruppata per 'dataset':")
print(group_by_dataset)

print("\nMedia raggruppata per 'language':")
print(group_by_language)

print("\nMedia raggruppata per 'algorithm':")
print(group_by_algorithm)

medianData = median_unique_quadruplets(df, columns)

group_by_algorithm = median_group_by(medianData, "algorithm", columns)
group_by_dataset = median_group_by(medianData, "dataset", columns)
group_by_language = median_group_by(medianData, "language", columns)

saveCsv(group_by_language, "median_group_by_language.csv")
saveCsv(group_by_algorithm, "median_group_by_algorithm.csv")
saveCsv(group_by_dataset, "median_group_by_dataset.csv")

dfsum = pd.read_csv("processedDatasets/mean_group_by_algorithm.csv")
sumalgo = sum_metrics_per_language(dfsum)

sumalgolang = sum_metrics_per_language_and_algorithm(pd.read_csv("processedDatasets/meanEmissionsNew.csv"))

sumalgolang.to_csv("processedDatasets/sum_metrics_per_language_and_algo.csv")