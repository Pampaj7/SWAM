import os
import subprocess
from codecarbon import EmissionsTracker
import pandas as pd


def mainR():
    datasets = ["breastCancer", "wine", "iris"]
    algorithms = ["logisticRegression", "XGBoost", "decisionTree", "randomForest", "KNN", "SVC", "GMM"]
    repetition = 1
    new_data = []
    new_csv_filename = 'emissions_detailed.csv'  # Choose an appropriate name for the new file

    remove("emissions.csv")

    for dataset in datasets:
        for algorithm in algorithms:
            for _ in range(repetition):
                tracker = EmissionsTracker(output_dir='R')

                print("Executing R script:")
                print(f"with {dataset} , {algorithm}")

                tracker.start()

                # Run the model and capture the result
                result = run_r_script(dataset, algorithm)

                tracker.stop()

                # Print the result
                handle_subprocess_result(result)

                new_data.append({'algorithm': algorithm, 'dataset': dataset, 'language': 'R'})

    emissions_df = pd.read_csv('R/emissions.csv')
    new_data_df = pd.DataFrame(new_data)
    assert len(new_data_df) == len(emissions_df), "Mismatch in row count between emissions data and new columns."
    emissions_df = pd.concat([emissions_df, new_data_df], axis=1)
    emissions_df.to_csv(f'R/{new_csv_filename}', index=False)

    print(f"{new_csv_filename} has been created with new columns.")


def run_r_script(dataset, algorithm):
    subResult = subprocess.run(["Rscript", "R/rRunner.R", dataset, algorithm], capture_output=True, text=True)
    return subResult


def remove(filename):
    try:
        os.remove(f"R/{filename}")
    except FileNotFoundError:
        print("emission.csv doesn't exist yet")
    except Exception as e:
        print(f'error occurred: {e}')


def handle_subprocess_result(result: subprocess.CompletedProcess):
    print("Standard Output:")
    print(result.stdout)

    if result.stderr:
        print("Standard Error:")
        print(result.stderr)
    else:
        print("No errors occurred.")
