import subprocess
from codecarbon import EmissionsTracker
import pandas as pd
import os


def processCsv():
    # Percorso della cartella contenente i file CSV
    directory_path = "/Users/pampaj/PycharmProjects/SWAM/src/cpp/output"

    # Verifica se la cartella esiste
    if not os.path.exists(directory_path):
        print(f"La cartella {directory_path} non esiste.")
    else:
        # Elenco di tutti i file nella cartella
        for file_name in os.listdir(directory_path):
            if file_name.endswith(".csv"):
                file_path = os.path.join(directory_path, file_name)

                # Estrai informazioni dal nome del file
                base_name = os.path.splitext(file_name)[0]
                parts = base_name.split("_")

                if len(parts) == 4:
                    algorithm = parts[0]
                    dataset = parts[1]
                    phase = parts[2]
                    language = "cpp"  # Come specificato

                    # Carica il file CSV in un DataFrame
                    df = pd.read_csv(file_path)

                    # Aggiungi le nuove colonne
                    df["algorithm"] = algorithm
                    df["dataset"] = dataset
                    df["language"] = language
                    df["phase"] = phase

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
    all_files = [
        os.path.join(directory_path, file_name)
        for file_name in os.listdir(directory_path)
        if file_name.endswith(".csv")
    ]

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


def compile_cpp():
    try:
        # Compila il codice C++
        subprocess.run(["make"], check=True)
        print("Compilazione completata con successo.")
    except subprocess.CalledProcessError as e:
        print("Errore durante la compilazione:")
        print(e)
        print("Output di errore:")
        print(e.stderr)


def run_cpp_program(dataset, algorithm, test):
    try:
        # Esegui l'eseguibile compilato
        result = subprocess.run(
            ["./src/cRunner", dataset, algorithm, test],
            capture_output=True,
            text=True,
            check=True,
        )
        print("Output del programma C++:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Errore durante l'esecuzione del programma C++:")
        print(e)
        print("Output di errore:")
        print(e.stderr)


def main():
    datasets = ["breastCancer", "wine", "iris"]
    algorithms = [
        "logisticRegression",
        # "XGBoost",
        "decisionTree",
        "randomForest",
        "KNN",
        "SVC",
        # "GMM",
        "adaBoost",
        "naiveBayes",
    ]
    repetition = 1
    new_data = []
    new_csv_filename = (
        "emissions_detailed.csv"  # Choose an appropriate name for the new file
    )

    for dataset in datasets:
        for algorithm in algorithms:
            for _ in range(repetition):
                # TODO: overwrite emission.csv or delete every time
                compile_cpp()

                print("Executing Cpp train:")
                print(f"with {dataset} , {algorithm}")
                # Run the model and capture the result
                run_cpp_program(dataset, algorithm, "false")

                os.rename(
                    "./output/emissions.csv",
                    f"./output/{algorithm}_{dataset}_train_emissions.csv",
                )

                print("Executing Cpp test:")
                print(f"with {dataset} , {algorithm}")
                # Run the model and capture the result
                run_cpp_program(dataset, algorithm, "true")

                # Print the result
                os.rename(
                    "./output/emissions.csv",
                    f"./output/{algorithm}_{dataset}_test_emissions.csv",
                )
    processCsv()
    mergeCsvFiles("output/", "./emissions_detailed.csv")


# main
if __name__ == "__main__":
    main()
