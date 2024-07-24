
% Leggi il dataset CSV
data = readtable('../../datasets/breastcancer/breastcancer.csv');

% Conversione della diagnosi in binario (M=1, B=0)
data.diagnosis = strcmp(data.diagnosis, 'M');

% Divisione dei dati in set di addestramento (80%) e di test (20%)
cv = cvpartition(size(data, 1), 'HoldOut', 0.2);
trainData = data(training(cv), :);
testData = data(test(cv), :);

% Estrazione delle variabili indipendenti (X) e della variabile dipendente (y)
X_train = trainData{:, 3:end}; % Presuppone che le prime due colonne siano ID e diagnosi
y_train = trainData.diagnosis;

X_test = testData{:, 3:end};
y_test = testData.diagnosis;

% Addestramento del modello di regressione logistica
mdl = fitglm(X_train, y_train, 'Distribution', 'binomial', 'Link', 'logit');

% Fare previsioni sul set di test
y_pred = round(predict(mdl, X_test));

% Calcolare l'accuratezza
accuracy = mean(y_pred == y_test);

% Stampare l'accuratezza
disp(['Accuracy: ', num2str(accuracy)]);
