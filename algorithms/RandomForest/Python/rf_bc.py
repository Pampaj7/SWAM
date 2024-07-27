import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def RandomForest():
    data = pd.read_csv('/home/alessandro/PycharmProjects/SWAM/datasets/breastcancer/breastcancer.csv')

    #Data processing
    X = data.drop(columns=["diagnosis"])
    y = data["diagnosis"]
    y = y.map({"M": 1, "B": 0})

    #Training and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    #Model Training
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    #Model validaton
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"\nAccuracy: {accuracy}")
    print("\nClassification Report:")
    print(class_report)