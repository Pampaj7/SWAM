import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

def XGBOOST():
    # Load the dataset
    data = pd.read_csv('/home/alessandro/PycharmProjects/SWAM/datasets/breastcancer/breastcancer.csv')

    # Data processing
    X = data.drop(columns=["diagnosis"])
    y = data["diagnosis"]
    y = y.map({"M": 1, "B": 0})

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create and train the XGBoost model
    model = xgb.XGBClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"\nAccuracy: {accuracy}")
    print("\nClassification Report:")
    print(class_report)

XGBOOST()
