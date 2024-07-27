import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
def Decision_tree():
    data = pd.read_csv('/home/alessandro/PycharmProjects/SWAM/datasets/breastcancer/breastcancer.csv')
    print("Data head:\n", data.head())

    #Data processing
    X = data.drop(columns=['diagnosis'])
    y = data['diagnosis']
    y = y.map({'M': 1, 'B': 0})

    # Training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model= DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    #Prediction
    y_pred = model.predict(X_test)

    #Model validation
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"\nAccuracy: {accuracy}")
    print("\nClassification Report:")
    print(class_report)

Decision_tree()

