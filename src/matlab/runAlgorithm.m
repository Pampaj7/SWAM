function runAlgorithm(algorithm, dataset)
    % Carica i dati
    switch dataset
        case 'breastCancer'
            data = readtable('../../datasets/breastcancer/breastcancer.csv');
            data.diagnosis = double(categorical(data.diagnosis)) - 1; % 1 per 'M', 0 per 'B'
            X = data{:, setdiff(data.Properties.VariableNames, {'diagnosis', 'id'})};
            y = data.diagnosis;
        case 'wine'
            data = readtable('../../datasets/winequality/wine_data.csv', 'ReadVariableNames', false);

            % Definisci i nomi delle colonne
            data.Properties.VariableNames = {'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', ...
                                             'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', ...
                                             'pH', 'sulphates', 'alcohol', 'quality', 'type'};

            X = data{:, 1:end-1};
            y = data.quality;
        case 'iris'
            data = readtable('../../datasets/iris/iris.csv', 'ReadVariableNames', false);

            % Definisci i nomi delle colonne
            data.Properties.VariableNames = {'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'};

            % Trasforma le specie in numeri
            y = double(categorical(data.species)); % Converti le specie in numeri
            X = data{:, setdiff(data.Properties.VariableNames, {'species'})}; % Usa tutte le colonne tranne 'species'
        otherwise
            error('Dataset unknown');
    end

    % Standardizza i dati
    X = normalize(X);

    % Divisione dei dati in set di addestramento e di test
    cv = cvpartition(y, 'HoldOut', 0.2);
    X_train = X(training(cv), :);
    X_test = X(test(cv), :);
    y_train = y(training(cv));
    y_test = y(test(cv));

    % Seleziona l'algoritmo e addestra il modello
    switch algorithm
        case 'logisticRegression'
            model = fitcecoc(X_train, y_train, 'Learners', 'linear');
            y_pred = round(predict(model, X_test));
        case 'XGBoost'
            % Nota: MATLAB non ha una funzione diretta per XGBoost, quindi usa un'alternativa o libreria se disponibile
            model = fitcensemble(X_train, y_train, 'Method', 'Bag', 'Learners', 'tree');
            y_pred = predict(model, X_test);
        case 'decisionTree'
            model = fitctree(X_train, y_train);
            y_pred = predict(model, X_test);
        case 'randomForest'
            model = fitcensemble(X_train, y_train, 'Method', 'Bag', 'Learners', 'tree');
            y_pred = predict(model, X_test);
        case 'KNN'
            model = fitcknn(X_train, y_train, 'NumNeighbors', 5);
            y_pred = predict(model, X_test);
        case 'SVC'
            model = fitcecoc(X_train, y_train);
            y_pred = predict(model, X_test);
        case 'GMM'
            options = statset('MaxIter', 1000, 'TolFun', 1e-5);
            model = fitgmdist(X_train, 3, 'Options', options, 'CovarianceType', 'diagonal', 'RegularizationValue', 0.1); %regiularizan value is added!
            y_pred = cluster(model, X_test);
        otherwise
            error('Algorithm unknown');
    end

    % Calcola l'accuratezza
    accuracy = sum(y_pred == y_test) / length(y_test);
    fprintf('Accuracy for %s on %s: %.2f%%\n', algorithm, dataset, accuracy * 100);
end
