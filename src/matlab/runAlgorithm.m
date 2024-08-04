function runAlgorithm(algorithm, dataset)
    % Carica i dati
    switch dataset
        case 'breastcancer'
            data = readtable('../../datasets/breastcancer/breastcancer.csv');
            data.diagnosis = double(categorical(data.diagnosis)) - 1; % 1 per 'M', 0 per 'B'
            X = data{:, setdiff(data.Properties.VariableNames, {'diagnosis', 'id'})};
            y = data.diagnosis;
        case 'winequality'
            data = readtable('../../datasets/winequality/wine_data.csv');
            X = data{:, 1:end-1};
            y = data.quality;
        case 'iris'
            data = readtable('../../datasets/iris/iris.csv');
            X = data{:, 1:end-1};
            y = double(categorical(data.Species));
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
        case 'Logistic Regression'
            model = fitglm(X_train, y_train, 'Distribution', 'binomial', 'Link', 'logit');
            y_pred = round(predict(model, X_test));
        case 'XGBoost'
            model = fitcensemble(X_train, y_train, 'Method', 'Bag', 'Learners', 'tree');
            y_pred = predict(model, X_test);
        case 'Decision Tree'
            model = fitctree(X_train, y_train);
            y_pred = predict(model, X_test);
        case 'Random Forest'
            model = fitcensemble(X_train, y_train, 'Method', 'Bag', 'Learners', 'tree');
            y_pred = predict(model, X_test);
        case 'K-Nearest Neighbors'
            model = fitcknn(X_train, y_train, 'NumNeighbors', 5);
            y_pred = predict(model, X_test);
        case 'Support Vector Machine'
            model = fitcsvm(X_train, y_train, 'KernelFunction', 'linear');
            y_pred = predict(model, X_test);
        case 'Gaussian Mixture Model'
            options = statset('MaxIter', 500, 'TolFun', 1e-5);
            model = fitgmdist(X_train, 3, 'Options', options, 'CovarianceType', 'full');
            y_pred = cluster(model, X_test);
        otherwise
            error('Algorithm unknown');
    end

    % Calcola l'accuratezza
    accuracy = sum(y_pred == y_test) / length(y_test);
    fprintf('Accuracy for %s on %s: %.2f%%\n', algorithm, dataset, accuracy * 100);
end
