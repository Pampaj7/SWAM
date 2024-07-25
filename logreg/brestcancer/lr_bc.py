import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def logreg():
    csv_file_path = '../../datasets/breastcancer/breastcancer.csv'
    data = pd.read_csv(csv_file_path)

    X = data.drop(columns=['diagnosis'])
    y = data['diagnosis']

    y = y.map({'M': 1, 'B': 0})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=10000)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(class_report)
