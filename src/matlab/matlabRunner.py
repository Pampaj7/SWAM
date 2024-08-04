import matlab.engine
from codecarbon import EmissionsTracker

# Lista di combinazioni algoritmo-dataset
combinations = [
    ('Logistic Regression', 'breastcancer'),
    ('XGBoost', 'breastcancer'),
    ('Decision Tree', 'breastcancer'),
    ('Random Forest', 'breastcancer'),
    ('K-Nearest Neighbors', 'breastcancer'),
    ('Support Vector Machine', 'breastcancer'),
    ('Gaussian Mixture Model', 'breastcancer'),
    ('Logistic Regression', 'winequality'),
    ('XGBoost', 'winequality'),
    ('Decision Tree', 'winequality'),
    ('Random Forest', 'winequality'),
    ('K-Nearest Neighbors', 'winequality'),
    ('Support Vector Machine', 'winequality'),
    ('Gaussian Mixture Model', 'winequality'),
    ('Logistic Regression', 'iris'),
    ('XGBoost', 'iris'),
    ('Decision Tree', 'iris'),
    ('Random Forest', 'iris'),
    ('K-Nearest Neighbors', 'iris'),
    ('Support Vector Machine', 'iris'),
    ('Gaussian Mixture Model', 'iris')
]


# Funzione per eseguire uno script MATLAB e tracciare il consumo energetico
def run_matlab_script(algorithm, dataset):
    tracker = EmissionsTracker()
    tracker.start()

    try:
        # Avvia l'engine MATLAB
        eng = matlab.engine.start_matlab()

        # Cambia la directory di lavoro di MATLAB
        eng.cd('/Users/pampaj/PycharmProjects/SWAM/src/matlab', nargout=0)

        # Esegui la funzione MATLAB con i parametri
        eng.runAlgorithm(algorithm, dataset, nargout=0)

        # Termina l'engine MATLAB
        eng.quit()
    except Exception as e:
        print(f"Error executing {algorithm} on {dataset}: {e}")

    emissions = tracker.stop()
    print(f"Emissions for {algorithm} on {dataset}: {emissions} kg CO2")


# Esegui ogni combinazione e traccia il consumo energetico
for algorithm, dataset in combinations:
    run_matlab_script(algorithm, dataset)
