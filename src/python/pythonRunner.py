import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from codecarbon import EmissionsTracker
import os

epochs = 1

# Percorso del file
file_path = 'python/emissions_detailed.csv'

# Verifica se il file esiste
if os.path.exists(file_path):
    # Cancella il file
    os.remove(file_path)
    print(f"{file_path} è stato cancellato.")
else:
    print(f"{file_path} non esiste.")

# Definizione degli algoritmi
algorithms = {
    LogisticRegression(
        random_state=42, max_iter=10000, class_weight="balanced"
    ): "logisticRegression",
    XGBClassifier(
        random_state=42,
        n_estimators=100,
        use_label_encoder=False,
        eval_metric="logloss",
    ): "XGBoost",
    DecisionTreeClassifier(random_state=42, class_weight="balanced"): "decisionTree",
    RandomForestClassifier(
        random_state=42, n_estimators=100, class_weight="balanced"
    ): "randomForest",
    KNeighborsClassifier(n_neighbors=5): "KNN",
    SVC(kernel="linear", random_state=42, class_weight="balanced"): "SVC",
    GaussianMixture(n_components=3, random_state=42): "GMM",
}


def run_algorithms(X, y, dataset_name):
    results = {name: {"accuracy": 0, "emissions": 0} for name in algorithms.values()}

    # Divisione dei dati in set di addestramento e di test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    for model, name in algorithms.items():
        print(f"Esecuzione dell'algoritmo: {name} sul dataset: {dataset_name}")

        file_name = "emissions_detailed.csv"

        # Itera su ciascuna epoca
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            tracker = EmissionsTracker(output_dir="python", output_file=file_name)
            tracker.start()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            emissions = tracker.stop()

            if emissions is None:
                emissions = 0

            accuracy = accuracy_score(y_test, y_pred)

            results[name]["accuracy"] += accuracy
            results[name]["emissions"] += emissions

            # Aggiunge i dati al file esistente

        results[name]["accuracy"] /= epochs
        results[name]["emissions"] /= epochs

        print(
            f"{name} Average Accuracy: {results[name]['accuracy']:.4f}, Average Emissions: {results[name]['emissions']:.4f} kg CO2"
        )

    return results


def breastCancerAlgos():
    # Carica i dati
    csv_file_path = "../datasets/breastcancer/breastcancer.csv"
    data = pd.read_csv(csv_file_path)

    # Prepara i dati
    data["diagnosis"] = data["diagnosis"].apply(lambda x: 1 if x == "M" else 0)
    X = data.drop(columns=["diagnosis", "id"])
    y = data["diagnosis"]

    # Standardizza i dati
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Esegui gli algoritmi e ottieni i risultati
    run_algorithms(X, y, dataset_name="breastCancer")


def irisAlgos():
    # Carica i dati Iris
    csv_file_path = (
        "../datasets/iris/iris.csv"  # Modifica con il percorso corretto del tuo file
    )
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

    X = data.drop("species", axis=1)
    y = data["species"]

    # Standardizza i dati
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Esegui gli algoritmi e ottieni i risultati
    run_algorithms(X, y, dataset_name="iris")


def wineQualityAlgos():
    wine_data = pd.read_csv("../datasets/winequality/wine_Data.csv")

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

    X = wine_data.drop("quality", axis=1)
    y = wine_data["quality"]

    # Standardizza i dati
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Esegui gli algoritmi e ottieni i risultati
    run_algorithms(X, y, dataset_name="wine")


def add_columns(file_path, language):
    df = pd.read_csv(file_path)

    # Creazione delle colonne vuote
    df["algorithm"] = ""
    df["dataset"] = ""
    df["language"] = ""

    # Lista degli algoritmi in ordine
    algorithms_order = [
        "logisticRegression",
        "XGBoost",
        "decisionTree",
        "randomForest",
        "KNN",
        "SVC",
        "GMM",
    ]

    # Lista dei dataset in ordine
    datasets_order = ["breastCancer", "iris", "wine"]

    num_algorithms = len(algorithms_order)
    dataset_size = (
            num_algorithms * epochs
    )  # Calcolo delle righe occupate da ciascun dataset

    # Assegna i valori alle righe
    for dataset_index, dataset_name in enumerate(datasets_order):
        start_dataset_row = dataset_index * dataset_size

        for i, algorithm in enumerate(algorithms_order):
            start_row = start_dataset_row + i * epochs
            end_row = start_row + epochs

            df.loc[start_row: end_row - 1, "algorithm"] = algorithm
            df.loc[start_row: end_row - 1, "dataset"] = dataset_name
            df.loc[start_row: end_row - 1, "language"] = language

    # Salva il file CSV con le nuove colonne
    df.to_csv(file_path, index=False)


breastCancerAlgos()

irisAlgos()

wineQualityAlgos()

#add_columns("python/emissions_detailed.csv", "python")
