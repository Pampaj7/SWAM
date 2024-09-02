import pandas as pd
from utils import mean_unique_quadruplets, mean_group_by, saveCsv, plot, median_unique_quadruplets, retrive_data, median_group_by


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
