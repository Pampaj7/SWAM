import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def logreg():
    csv_file_path = "../../../datasets/breastcancer/breastcancer.csv"
    data = pd.read_csv(csv_file_path)

    # Verifica dei dati di input
    print("\nData info:\n", data.info())
    print("\nData description:\n", data.describe())

    X = data.drop(columns=["diagnosis"])
    y = data["diagnosis"]

    y = y.map({"M": 1, "B": 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Verifica della divisione del dataset
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")


    model = LogisticRegression(max_iter=10000)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Verifica delle predizioni
    print(f"\nTrue labels: {y_test.values[:10]}")
    print(f"Predicted labels: {y_pred[:10]}")


    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(class_report)

