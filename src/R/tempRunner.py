from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
from codecarbon import EmissionsTracker

# Enable the automatic conversion of pandas DataFrames to R data frames
pandas2ri.activate()

# Load the R script
r.source("rRunner.R")


def run_model_with_dataset(dataset_name, algorithm_name):
    # Call the R function with the given parameters
    result = r.run_model_with_dataset(dataset_name, algorithm_name)
    return result


if __name__ == "__main__":
    tracker = EmissionsTracker()

    # Example parameters
    dataset_name = "iris"  # Change this as needed
    algorithm_name = "randomForest"  # Change this as needed

    print("Executing R script:")
    tracker.start()

    # Run the model and capture the result
    result = run_model_with_dataset(dataset_name, algorithm_name)

    tracker.stop()

    # Print the result
    print("R function output:")
    print(result)
