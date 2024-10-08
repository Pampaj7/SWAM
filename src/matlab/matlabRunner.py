import matlab.engine
import pandas as pd
import os

epochs = 1

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

combinations = [
    ("logisticRegression", "breastCancer"),
    ("adaBoost", "breastCancer"),
    ("decisionTree", "breastCancer"),
    ("randomForest", "breastCancer"),
    ("KNN", "breastCancer"),
    ("SVC", "breastCancer"),
    ("naiveBayes", "breastCancer"),
    ("logisticRegression", "iris"),
    ("adaBoost", "iris"),
    ("decisionTree", "iris"),
    ("randomForest", "iris"),
    ("KNN", "iris"),
    ("SVC", "iris"),
    ("naiveBayes", "iris"),
    ("logisticRegression", "wine"),
    ("adaBoost", "wine"),
    ("decisionTree", "wine"),
    ("randomForest", "wine"),
    ("KNN", "wine"),
    ("SVC", "wine"),
    ("naiveBayes", "wine"),
]


def processCsv(language):
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
                    language = language  # Come specificato

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

processCsv('matlab')
mergeCsvFiles("matlab/models", "matlab/emissions_detailed.csv")
