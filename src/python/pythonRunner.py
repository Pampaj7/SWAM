import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from codecarbon import EmissionsTracker
from sklearn.naive_bayes import GaussianNB
import os

epochs = 1

# Percorso del file
file_path = ['python/train_emissions_detailed.csv', "python/test_emissions_detailed.csv",
             "python/emissions_detailed.csv"]

for file in file_path:
    if os.path.exists(file):
        # Cancella il file
        os.remove(file)
        print(f"{file} è stato cancellato.")
    else:
        print(f"{file} non esiste.")

# these hyperparameters are based on the default values of the algorithms
algorithms = {
    LogisticRegression(max_iter=10000): "logisticRegression",  # Solo max_iter
    AdaBoostClassifier(n_estimators=100): "AdaBoost",  # Solo n_estimators
    DecisionTreeClassifier(): "decisionTree",  # Default
    RandomForestClassifier(n_estimators=100): "randomForest",  # Solo n_estimators
    KNeighborsClassifier(n_neighbors=5): "KNN",  # Solo n_neighbors
    SVC(kernel="linear"): "SVC",  # Solo kernel
    GaussianNB(): "naiveBayes"  # Default
}


def run_algorithms(X, y, dataset_name):
    results = {name: {"accuracy": 0, "fit_emissions": 0, "predict_emissions": 0} for name in algorithms.values()}

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    for model, name in algorithms.items():
        print(f"Running algorithm: {name} on dataset: {dataset_name}")

        file_name = "emissions_detailed.csv"

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            fit_tracker = EmissionsTracker(output_dir="python", output_file=f"train_{file_name}")
            fit_tracker.start()
            model.fit(X_train, y_train)
            fit_emissions = fit_tracker.stop()

            if fit_emissions is None:
                fit_emissions = 0

            predict_tracker = EmissionsTracker(output_dir="python", output_file=f"test_{file_name}")
            predict_tracker.start()
            y_pred = model.predict(X_test)
            predict_emissions = predict_tracker.stop()

            if predict_emissions is None:
                predict_emissions = 0

            accuracy = accuracy_score(y_test, y_pred)

            results[name]["accuracy"] += accuracy
            results[name]["fit_emissions"] += fit_emissions
            results[name]["predict_emissions"] += predict_emissions

        # Average results over the epochs
        results[name]["accuracy"] /= epochs
        results[name]["fit_emissions"] /= epochs
        results[name]["predict_emissions"] /= epochs

        print(
            f"{name} Average Accuracy: {results[name]['accuracy']:.4f}, "
            f"Average Fit Emissions: {results[name]['fit_emissions']:.4f} kg CO2, "
            f"Average Predict Emissions: {results[name]['predict_emissions']:.4f} kg CO2"
        )

    return results


def breastCancerAlgos():
    # Carica i dati
    csv_file_path = "../datasets/breastcancer/breastCancer_processed.csv"
    data = pd.read_csv(csv_file_path)

    X = data.drop(columns=["target"])
    y = data["target"]

    run_algorithms(X, y, dataset_name="breastCancer")


def irisAlgos():
    # Carica i dati Iris
    csv_file_path = (
        "../datasets/iris/iris_processed.csv"  # Modifica con il percorso corretto del tuo file
    )
    data = pd.read_csv(csv_file_path, header=0)

    X = data.drop("target", axis=1)
    y = data["target"]

    run_algorithms(X, y, dataset_name="iris")


def wineQualityAlgos():
    wine_data = pd.read_csv("../datasets/winequality/wineQuality_processed.csv")

    X = wine_data.drop("target", axis=1)
    y = wine_data["target"]

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
        "adaBoost",
        "decisionTree",
        "randomForest",
        "KNN",
        "SVC",
        "naiveBayes",
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


def merge_and_add_source_column(file1_path, file2_path, output_file_path):
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    def extract_first_word(file_path):
        base_name = os.path.basename(file_path)
        first_word = base_name.split('_')[0]
        return first_word

    # Estrai la prima parola dai nomi dei file
    source1 = extract_first_word(file1_path)
    source2 = extract_first_word(file2_path)

    # Aggiungi la colonna ai DataFrame
    df1['phase'] = source1
    df2['phase'] = source2

    # Unisci i due DataFrame
    merged_df = pd.concat([df1, df2], ignore_index=True)

    # Salva il DataFrame unito in un nuovo file CSV
    merged_df.to_csv(output_file_path, index=False)

    print(f"File CSV fusi e salvati in {output_file_path}")


breastCancerAlgos()

irisAlgos()

wineQualityAlgos()

add_columns("python/test_emissions_detailed.csv", "python")
add_columns("python/train_emissions_detailed.csv", "python")
merge_and_add_source_column("python/test_emissions_detailed.csv", "python/train_emissions_detailed.csv",
                            "python/emissions_detailed.csv")
