from utils import *


df = pd.read_csv('merged_emissions.csv')
meanData = mean_unique_triplets(df, "duration", "energy_consumed")  # dataset con le medie di tutte le triplette uguali

# saveCsv(meanData, "meanEmissions.csv")  # salva nella cartella processedDataset

print(meanData)

# mean_group_by() rende un dataset con le medie calcolate raggruppando sulla prima stringa data in input

group_by_algorithm = mean_group_by(meanData, "algorithm", "duration", "energy_consumed")
group_by_dataset = mean_group_by(meanData, "dataset", "duration", "energy_consumed")
group_by_language = mean_group_by(meanData, "language", "duration", "energy_consumed")

print("Group by dataset")
print(group_by_dataset)
print("\nGroup by language")
print(group_by_language)
print("\nGroup by algorithm")
print(group_by_algorithm)

# per plottare scegli l'asse x e l'asse y

plot(group_by_language, "language", "duration")
plot(group_by_dataset, "dataset", "energy_consumed")
plot(group_by_algorithm, "algorithm", "duration")



# mediana

medianData = median_unique_triplets(df, "duration", "energy_consumed")
group_by_algorithm = mean_group_by(medianData, "algorithm", "duration", "energy_consumed")
group_by_dataset = mean_group_by(medianData, "dataset", "duration", "energy_consumed")
group_by_language = mean_group_by(medianData, "language", "duration", "energy_consumed")

plot(group_by_language, "language", "duration")
plot(group_by_dataset, "dataset", "energy_consumed")
plot(group_by_algorithm, "algorithm", "duration")