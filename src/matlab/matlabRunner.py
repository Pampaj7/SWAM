import matlab.engine
from codecarbon import EmissionsTracker
import pandas as pd

epochs = 10
# Lista di combinazioni algoritmo-dataset
combinations = [
    ('logisticRegression', 'breastCancer'),
    ('XGBoost', 'breastCancer'),
    ('decisionTree', 'breastCancer'),
    ('randomForest', 'breastCancer'),
    ('KNN', 'breastCancer'),
    ('SVC', 'breastCancer'),
    ('GMM', 'breastCancer'),
    ('logisticRegression', 'iris'),
    ('XGBoost', 'iris'),
    ('decisionTree', 'iris'),
    ('randomForest', 'iris'),
    ('KNN', 'iris'),
    ('SVC', 'iris'),
    ('GMM', 'iris'),
    ('logisticRegression', 'wine'),
    ('XGBoost', 'wine'),
    ('decisionTree', 'wine'),
    ('randomForest', 'wine'),
    ('KNN', 'wine'),
    ('SVC', 'wine'),
    ('GMM', 'wine')
]


# Funzione per eseguire uno script MATLAB e tracciare il consumo energetico
def run_matlab_script(engine, algorithm, dataset):
    tracker = EmissionsTracker(output_dir='.', output_file=file_name)
    tracker.start()

    try:

        engine.runAlgorithm(algorithm, dataset, nargout=0)

    except Exception as e:
        print(f"Error executing {algorithm} on {dataset}: {e}")

    emissions = tracker.stop()
    print(f"Emissions for {algorithm} on {dataset}: {emissions} kg CO2")


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
        "GMM"
    ]

    # Lista dei dataset in ordine
    datasets_order = [
        "breastCancer",
        "iris",
        "wine"
    ]

    num_algorithms = len(algorithms_order)
    dataset_size = num_algorithms * epochs  # Calcolo delle righe occupate da ciascun dataset

    # Assegna i valori alle righe
    for dataset_index, dataset_name in enumerate(datasets_order):
        start_dataset_row = dataset_index * dataset_size

        for i, algorithm in enumerate(algorithms_order):
            start_row = start_dataset_row + i * epochs
            end_row = start_row + epochs

            df.loc[start_row:end_row - 1, "algorithm"] = algorithm
            df.loc[start_row:end_row - 1, "dataset"] = dataset_name
            df.loc[start_row:end_row - 1, "language"] = language

    # Salva il file CSV con le nuove colonne
    df.to_csv(file_path, index=False)


file_name = 'emissions_detailed.csv'

eng = matlab.engine.start_matlab()

for algorithm, dataset in combinations:
    for epoch in range(10):
        print(f"Running {algorithm} on {dataset}, epoch {epoch + 1}")
        run_matlab_script(eng, algorithm, dataset)

eng.quit()

add_columns(file_name, "matlab")
