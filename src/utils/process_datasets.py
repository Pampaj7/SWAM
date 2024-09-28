import pandas as pd
from sklearn.preprocessing import StandardScaler
import os


def save_dataset(X, y, dataset_name, original_path):
    """Save the processed dataset (features and labels) to a CSV file in the same directory as the original dataset."""
    processed_data = pd.DataFrame(X)
    processed_data["label"] = y

    # Get directory from original path and save the processed file there
    directory = os.path.dirname(original_path)
    processed_file_path = os.path.join(directory, f"{dataset_name}_processed.csv")

    processed_data.to_csv(processed_file_path, index=False)
    print(f"Dataset {dataset_name} saved to {processed_file_path}")


def breastCancerAlgos():
    """Process the Breast Cancer dataset."""
    csv_file_path = "../../datasets/breastcancer/breastcancer.csv"
    data = pd.read_csv(csv_file_path)

    # Prepare the data
    data["diagnosis"] = data["diagnosis"].apply(lambda x: 1 if x == "M" else 0)
    X = data.drop(columns=["diagnosis", "id"])
    y = data["diagnosis"]

    # Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Run algorithms and save the dataset
    save_dataset(X, y, dataset_name="breastCancer", original_path=csv_file_path)


def irisAlgos():
    """Process the Iris dataset."""
    csv_file_path = "../../datasets/iris/iris.csv"
    data = pd.read_csv(csv_file_path)

    # Define column names
    data.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
    ]

    # Convert labels to numeric
    data["species"] = data["species"].astype("category").cat.codes

    X = data.drop("species", axis=1)
    y = data["species"]

    # Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Run algorithms and save the dataset
    save_dataset(X, y, dataset_name="iris", original_path=csv_file_path)


def wineQualityAlgos():
    """Process the Wine Quality dataset with transformed 'quality' column."""
    csv_file_path = "../../datasets/winequality/wine_data.csv"
    wine_data = pd.read_csv(csv_file_path)

    # Define column names
    wine_data.columns = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
        "quality",
        "type",
    ]

    # Convert wine type to numeric
    wine_data["type"] = wine_data["type"].astype("category").cat.codes

    # Transform 'quality' column: 0 if < 5, 1 if >= 5
    wine_data["quality"] = wine_data["quality"].apply(lambda x: 1 if x > 5 else 0)

    X = wine_data.drop("quality", axis=1)
    y = wine_data["quality"]

    # Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Run algorithms and save the dataset
    save_dataset(X, y, dataset_name="wineQuality", original_path=csv_file_path)


if __name__ == "__main__":
    breastCancerAlgos()
    irisAlgos()
    wineQualityAlgos()
