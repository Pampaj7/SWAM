import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


def process_breast_cancer_data():
    # Carica i dati
    csv_file_path = "../../datasets/breastcancer/breastcancer.csv"
    data = pd.read_csv(csv_file_path)

    # Prepara i dati
    data["diagnosis"] = data["diagnosis"].apply(lambda x: 1 if x == "M" else 0)
    data = data.drop(columns=["id"])  # Drop the 'id' column

    # Standardizza i dati
    scaler = StandardScaler()
    data[data.columns.difference(["diagnosis"])] = scaler.fit_transform(
        data[data.columns.difference(["diagnosis"])]
    )

    # Salva i dati preprocessati
    save_processed_data(data, csv_file_path)


def process_iris_data():
    # Carica i dati Iris
    csv_file_path = "../../datasets/iris/iris.csv"
    data = pd.read_csv(csv_file_path, header=0)

    # Definisci i nomi delle colonne
    data.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
    ]

    # Converti le etichette in numeri
    data["species"] = data["species"].astype("category").cat.codes
    print(data)
    # Standardizza i dati
    scaler = StandardScaler()
    data[data.columns.difference(["species"])] = scaler.fit_transform(
        data[data.columns.difference(["species"])]
    )

    # Salva i dati preprocessati
    save_processed_data(data, csv_file_path)


def process_wine_quality_data():
    # Carica i dati
    csv_file_path = "../../datasets/winequality/wine_Data.csv"
    wine_data = pd.read_csv(csv_file_path)

    # Definisci i nomi delle colonne
    wine_data.columns = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
        "quality",
        "type",
    ]

    # Converti il tipo di vino in numeri
    wine_data["type"] = wine_data["type"].astype("category").cat.codes
    wine_data["quality"] = wine_data["quality"] - wine_data["quality"].min()

    # Standardizza i dati
    scaler = StandardScaler()
    wine_data[wine_data.columns.difference(["quality"])] = scaler.fit_transform(
        wine_data[wine_data.columns.difference(["quality"])]
    )

    # Salva i dati preprocessati
    save_processed_data(wine_data, csv_file_path)


def save_processed_data(data, original_csv_path):
    # Crea la cartella per i dati preprocessati se non esiste
    processed_folder = os.path.join(
        os.path.dirname(original_csv_path), "dataset_processed"
    )
    os.makedirs(processed_folder, exist_ok=True)

    # Salva l'intero dataset preprocessato come CSV
    dataset_name = os.path.splitext(os.path.basename(original_csv_path))[0]
    processed_file_path = os.path.join(
        processed_folder, f"{dataset_name}_processed.csv"
    )

    data.to_csv(processed_file_path, index=False)

    print(f"Processed data saved to {processed_file_path}")


# Process each dataset
process_breast_cancer_data()
process_iris_data()
process_wine_quality_data()
