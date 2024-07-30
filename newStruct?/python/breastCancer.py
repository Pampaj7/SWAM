from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from codecarbon import EmissionsTracker
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

algorithms = {
    LogisticRegression(random_state=42, max_iter=10000, class_weight='balanced'): "Logistic Regression",
    XGBClassifier(random_state=42, n_estimators=100, use_label_encoder=False, eval_metric='logloss'): "XGBoost",
    DecisionTreeClassifier(random_state=42, class_weight='balanced'): "Decision Tree",
    RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'): "Random Forest",
    KNeighborsClassifier(n_neighbors=5): "K-Nearest Neighbors",
    SVC(kernel='linear', random_state=42, class_weight='balanced'): "Support Vector Machine",
    GaussianMixture(n_components=2, random_state=42): "Gaussian Mixture Model",
    # gmm si fa uso da classificatori con soft-clustering
    # 2 componenti con class binaria = results scarsi
}


def breastCancerAlgos():
    # Carica i dati
    csv_file_path = "../../datasets/breastcancer/breastcancer.csv"
    data = pd.read_csv(csv_file_path)

    # Prepara i dati
    data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
    X = data.drop(columns=['diagnosis', 'id'])
    y = data['diagnosis']

    # Standardizza i dati
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Divisione dei dati in set di addestramento e di test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    tracker = EmissionsTracker()

    # Ciclo sugli algoritmi
    for model, name in algorithms.items():
        # Addestramento del modello
        tracker.start()
        if name == "Gaussian Mixture Model":
            model.fit(X_train)
            # Previsione sul set di test (soft clustering)
            y_pred = model.predict(X_test)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        tracker.stop()

        # Calcolare l'accuratezza
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {accuracy:.4f}")


def getGraphics():
    data = pd.read_csv("emissions.csv")

    # Convertire il timestamp in formato datetime
    # data['timestamp'] = pd.to_datetime(data['timestamp'])

    df = data[['timestamp', 'duration', 'emissions', 'emissions_rate', 'cpu_power', 'gpu_power', 'ram_power',
               'energy_consumed']]

    plt.figure(figsize=(12, 6))
    sns.lineplot(x='emissions', y='duration', data=df, marker='o')
    plt.title('Emissioni di CO2 nel Tempo')
    plt.xlabel('emission')
    plt.ylabel('duration')
    plt.grid(True)
    plt.show()


breastCancerAlgos()
getGraphics()
