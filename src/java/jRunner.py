import subprocess
from codecarbon import EmissionsTracker
import pandas as pd
import os


def processCsv():
    # Percorso della cartella contenente i file CSV
    directory_path = "./output"

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
                    language = "java"  # Come specificato

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


# WATCHOUT!!!! first compile the java file with javac script.java and then run it with java script. now works with allRunner.py
def compile_java():
    try:
        # Compila il codice Java
        subprocess.run(["mvn", "clean", "compile"], check=True)
        print("Compilazione completata con successo.")
    except subprocess.CalledProcessError as e:
        print("Errore durante la compilazione:")
        print(e)
        print("Output di errore:")
        print(e.stderr)


def run_java_program(dataset, algorithm, train):
    # compile_java()
    try:
        # Esegui il programma Java
        result = subprocess.run(
            [
                "mvn",
                "exec:java",
                "-Dexec.mainClass=com.example.main",
                f"-Dexec.args={dataset} {algorithm} {train}",
            ],
            capture_output=True,
            # text=True,
            check=True,
        )
        print("Output del programma Java:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Errore durante l'esecuzione del programma Java:")
        print(e)
        print("Output di errore:")
        print(e.stderr)


def check_exist(emissions_file):
    # Check if the file already exists
    if os.path.exists(emissions_file):
        # Open both the original emissions.csv and the target file
        with open("./output/emissions.csv", "r") as src, open(
            emissions_file, "a"
        ) as dest:
            # Read the header from the source file
            src_lines = src.readlines()
            # If the target file is empty, include the header
            if os.path.getsize(emissions_file) == 0:
                dest.write(src_lines[0])  # Write the header
            # Write the data without the header
            dest.writelines(src_lines[1:])
        # Optionally, remove the source emissions.csv if no longer needed
        os.remove("./output/emissions.csv")
    else:
        # If the file does not exist, rename emissions.csv
        os.rename(
            "./output/emissions.csv",
            emissions_file,
        )


def main():
    datasets = ["breastCancer", "wine", "iris"]
    algorithms = [
        "randomForest",
        "logisticRegression",
        # "XGBoost",
        "decisionTree",
        "KNN",
        "SVC",
        # "GMM",
        "adaBoost",
        "naiveBayes",
    ]
    repetition = 30
    new_data = []
    new_csv_filename = (
        "emissions_detailed.csv"  # Choose an appropriate name for the new file
    )
    compile_java()
    for dataset in datasets:
        for algorithm in algorithms:
            for _ in range(repetition):
                # TODO: overwrite emission.csv or delete every time

                print("Executing java script:")
                print(f"with {dataset} , {algorithm}")

                # Run the model and capture the result
                run_java_program(dataset, algorithm, "true")
                check_exist(f"./output/{algorithm}_{dataset}_train_emissions.csv")

                run_java_program(dataset, algorithm, "false")
                check_exist(f"./output/{algorithm}_{dataset}_test_emissions.csv")

    processCsv()
    mergeCsvFiles("./output/", "emissions_detailed.csv")

    # Print the result


# main
if __name__ == "__main__":
    main()
