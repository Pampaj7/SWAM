import pandas as pd

# Verifica del file .data
data_path = '../../datasets/breastcancer/wdbc.data'
df = pd.read_csv(data_path, header=None)

# Aggiungi i nomi delle colonne (se conosci i nomi delle colonne)
column_names = ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
                'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
                'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
                'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']

df.columns = column_names

# Controllo di eventuali valori NaN
df = df.dropna()

# Conversione in CSV
df.to_csv('../../datasets/breastcancer/breastcancer.csv', index=False)

print("Conversione completata con successo!")
