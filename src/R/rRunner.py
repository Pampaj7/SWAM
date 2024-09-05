import os
import subprocess
import time
from R.handleCsv import processCsv

from codecarbon import EmissionsTracker
import pandas as pd


def mainR():
    datasets = ["breastCancer", "wine", "iris"]
    algorithms = [
        "logisticRegression",
        "decisionTree",
        "randomForest",
        "KNN",
        "SVC",
        "adaBoost",
        "naiveBayes"
    ]
    repetition = 10
    savePath = "R/csv"
    language = "R"

    removeFile("emissions_detailed.csv")
    removeFolderFiles(savePath)

    for dataset in datasets:
        for algorithm in algorithms:
            for _ in range(repetition):
                print("Executing R script:")
                print(f"with {dataset} , {algorithm}")

                # Run the model and capture the result
                result = run_r_script(dataset, algorithm, savePath)

                # Print the result
                handle_subprocess_result(result)
                time.sleep(1)
    processCsv(language, savePath)


def run_r_script(dataset, algorithm, savePath):
    subResult = subprocess.run(
        ["Rscript", "R/rRunner.R", dataset, algorithm, savePath], capture_output=True, text=True
    )
    return subResult


def removeFile(filename):
    try:
        os.remove(f"R/{filename}")
    except FileNotFoundError:
        print("emission.csv doesn't exist yet")
    except Exception as e:
        print(f"error occurred: {e}")


def removeFolderFiles(folder):
    for filename in os.listdir(folder):
        if filename.endswith('.csv'):
            removeFile(f"csv/{filename}")



def handle_subprocess_result(result: subprocess.CompletedProcess):
    print("Standard Output:")
    print(result.stdout)

    if result.stderr:
        print("Standard Error:")
        print(result.stderr)
    else:
        print("No errors occurred.")


def maintest():
    processCsv("R", "R/csv")

