import subprocess
from codecarbon import EmissionsTracker
import pandas as pd


def compile_cpp():
    try:
        # Compila il codice C++
        subprocess.run(["make", "", "cRunner", "cRunner.cpp"], check=True)
        print("Compilazione completata con successo.")
    except subprocess.CalledProcessError as e:
        print("Errore durante la compilazione:")
        print(e)
        print("Output di errore:")
        print(e.stderr)


def run_cpp_program():
    compile_cpp()
    try:
        # Esegui l'eseguibile compilato
        result = subprocess.run(
            ["./src/cRunner"], capture_output=True, text=True, check=True
        )
        print("Output del programma C++:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Errore durante l'esecuzione del programma C++:")
        print(e)
        print("Output di errore:")
        print(e.stderr)


def mainR():
    datasets = ["breastCancer", "wine", "iris"]
    algorithms = [
        "logisticRegression",
        "XGBoost",
        "decisionTree",
        "randomForest",
        "KNN",
        "SVC",
        "GMM",
    ]
    repetition = 1
    new_data = []
    new_csv_filename = (
        "emissions_detailed.csv"  # Choose an appropriate name for the new file
    )

    for dataset in datasets:
        for algorithm in algorithms:
            for _ in range(repetition):

                tracker = EmissionsTracker(
                    output_dir="cpp", output_file="emissions.csv"
                )
                # TODO: overwrite emission.csv or delete every time

                print("Executing R script:")
                print(f"with {dataset} , {algorithm}")

                tracker.start()

                # Run the model and capture the result
                compile_cpp()
                result = run_cpp_program(dataset, algorithm)

                tracker.stop()

                # Print the result
                print("R function output:")
                print(result)

                new_data.append(
                    {"Algorithm": algorithm, "Dataset": dataset, "Language": "R"}
                )

    emissions_df = pd.read_csv("R/emissions.csv")
    new_data_df = pd.DataFrame(new_data)
    assert len(new_data_df) == len(
        emissions_df
    ), "Mismatch in row count between emissions data and new columns."
    emissions_df = pd.concat([emissions_df, new_data_df], axis=1)
    emissions_df.to_csv(f"R/{new_csv_filename}", index=False)

    print(f"{new_csv_filename} has been created with new columns.")
