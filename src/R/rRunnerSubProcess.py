import subprocess
from codecarbon import EmissionsTracker


def run_r_script(dataset, algorithm):
    subResult = subprocess.run(["Rscript", "rRunner.R", dataset, algorithm], capture_output=True, text=True)
    print("R script output:")
    return subResult


if __name__ == "__main__":
    datasets = ["breastCancer", "wine", "iris"]
    algorithms = ["logisticRegression", "XGBoost", "decisionTree", "randomForest", "KNN", "SVC", "GMM"]

    for dataset in datasets:
        for algorithm in algorithms:
            tracker = EmissionsTracker()

            print("Executing R script:")
            print(f"with {dataset} , {algorithm}")

            tracker.start()

            # Run the model and capture the result
            result = run_r_script(dataset, algorithm)

            tracker.stop()

            # Print the result
            print("R function output:")
            print(result)
