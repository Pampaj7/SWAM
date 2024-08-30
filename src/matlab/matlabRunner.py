import matlab.engine
import pandas as pd
import os

# Percorso della cartella contenente i file da eliminare
directory_path = 'matlab/models'

# Elimina tutti i file nella cartella
if os.path.exists(directory_path):
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"{file_path} è stato cancellato.")
        else:
            print(f"{file_path} non è un file e non è stato cancellato.")
else:
    print(f"La cartella {directory_path} non esiste.")

# Percorso del file
file = 'matlab/emissions_detailed.csv'

# Verifica se il file esiste
if os.path.exists(file):
    # Cancella il file
    os.remove(file)
    print(f"{file} è stato cancellato.")
else:
    print(f"{file} non esiste.")

epochs = 1
# Lista di combinazioni algoritmo-dataset
combinations = [
    ("logisticRegression", "breastCancer"),
    ("XGBoost", "breastCancer"),
    ("decisionTree", "breastCancer"),
    ("randomForest", "breastCancer"),
    ("KNN", "breastCancer"),
    ("SVC", "breastCancer"),
    ("GMM", "breastCancer"),
    ("logisticRegression", "iris"),
    ("XGBoost", "iris"),
    ("decisionTree", "iris"),
    ("randomForest", "iris"),
    ("KNN", "iris"),
    ("SVC", "iris"),
    ("GMM", "iris"),
    ("logisticRegression", "wine"),
    ("XGBoost", "wine"),
    ("decisionTree", "wine"),
    ("randomForest", "wine"),
    ("KNN", "wine"),
    ("SVC", "wine"),
    ("GMM", "wine"),
]


def processCsv():
    # Percorso della cartella contenente i file CSV
    directory_path = 'matlab/models/'

    # Verifica se la cartella esiste
    if not os.path.exists(directory_path):
        print(f"La cartella {directory_path} non esiste.")
    else:
        # Elenco di tutti i file nella cartella
        for file_name in os.listdir(directory_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(directory_path, file_name)

                # Estrai informazioni dal nome del file
                base_name = os.path.splitext(file_name)[0]
                parts = base_name.split('_')

                if len(parts) == 4:
                    algorithm = parts[0]
                    dataset = parts[1]
                    phase = parts[2]
                    language = 'matlab'  # Come specificato

                    # Carica il file CSV in un DataFrame
                    df = pd.read_csv(file_path)

                    # Aggiungi le nuove colonne
                    df['algorithm'] = algorithm
                    df['dataset'] = dataset
                    df['language'] = language
                    df['phase'] = phase

                    # Salva il DataFrame aggiornato
                    updated_file_path = os.path.join(directory_path, file_name)
                    df.to_csv(updated_file_path, index=False)

                    print(f"Aggiornato: {updated_file_path}")
                else:
                    print(f"Nome del file non valido per: {file_name}")


def mergeCsvFiles(directory_path, merged_file_path):
    """
    Unisce tutti i file CSV presenti nella directory specificata e salva il risultato in un nuovo file CSV.

    Parameters:
    - directory_path: Percorso della directory contenente i file CSV da unire.
    - merged_file_path: Percorso del file CSV finale dove verranno salvati i dati uniti.
    """
    # Elenco di tutti i file nella cartella
    all_files = [os.path.join(directory_path, file_name) for file_name in os.listdir(directory_path) if
                 file_name.endswith('.csv')]

    # Lista per memorizzare i DataFrame
    dfs = []

    # Leggi e aggiungi ogni file CSV alla lista di DataFrame
    for file in all_files:
        df = pd.read_csv(file)
        dfs.append(df)

    # Unisci tutti i DataFrame in uno solo
    merged_df = pd.concat(dfs, ignore_index=True)

    # Salva il DataFrame unito in un nuovo file CSV
    merged_df.to_csv(merged_file_path, index=False)

    print(f"File CSV uniti e salvati in {merged_file_path}")


def runMatlabScript(engine, algorithm, dataset):
    try:
        engine.runAlgorithm(algorithm, dataset, nargout=0)

    except Exception as e:
        print(f"Error fitting {algorithm} on {dataset}: {e}")


eng = matlab.engine.start_matlab()

for algorithm, dataset in combinations:
    for epoch in range(epochs):
        print(f"Running {algorithm} on {dataset}, epoch {epoch + 1}")
        runMatlabScript(eng, algorithm, dataset)
eng.quit()

processCsv()
mergeCsvFiles("matlab/models", "matlab/emission_detailed.csv")
