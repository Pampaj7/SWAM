function runAlgo(algorithm, dataset)
    %needed to select path to own matlab functions
    pyenv('Version', '/Users/pampaj/anaconda3/envs/sw/bin/python');
    insert(py.sys.path, int32(0), '/Users/pampaj/PycharmProjects/SWAM/src/matlab/');

    switch dataset
        case 'breastCancer'
            data = readtable('../datasets/breastcancer/breastCancer_processed.csv');
            %disp(data.Properties.VariableNames);
            %displayed in Var..
            X = data{:, setdiff(data.Properties.VariableNames, {'Var31'})};  % Estrai tutte le colonne tranne 'target'
            y = data.Var31;
        case 'wine'
            data = readtable('../datasets/winequality/wineQuality_processed.csv', 'ReadVariableNames', false);
            X = data{:, setdiff(data.Properties.VariableNames, {'Var13'})};
            y = data.Var13;
        case 'iris'
            data = readtable('../datasets/iris/iris_processed.csv', 'ReadVariableNames', false);
            X = data{:, setdiff(data.Properties.VariableNames, {'Var5'})};
            y = data.Var5;
        otherwise
            error('Dataset unknown');
    end


    cv = cvpartition(y, 'HoldOut', 0.2);
    X_train = X(training(cv), :);
    X_test = X(test(cv), :);
    y_train = y(training(cv));
    y_test = y(test(cv));

    switch algorithm
        case 'logisticRegression'
            py.tracker_control.start_tracker('matlab/models', sprintf('%s_%s_train_emissions.csv', algorithm, dataset));
            model = fitcecoc(X_train, y_train, 'Learners', 'linear');
            py.tracker_control.stop_tracker();
        case 'adaBoost'
            py.tracker_control.start_tracker('matlab/models', sprintf('%s_%s_train_emissions.csv', algorithm, dataset));
            model = fitcensemble(X_train, y_train, 'Method', 'RUSBoost', 'NumLearningCycles', 100);
            py.tracker_control.stop_tracker();
        case 'decisionTree'
            py.tracker_control.start_tracker('matlab/models', sprintf('%s_%s_train_emissions.csv', algorithm, dataset));
            model = fitctree(X_train, y_train);
            py.tracker_control.stop_tracker();
        case 'randomForest'
            py.tracker_control.start_tracker('matlab/models', sprintf('%s_%s_train_emissions.csv', algorithm, dataset));
            model = fitcensemble(X_train, y_train, 'Method', 'Bag', 'NumLearningCycles', 100);
            py.tracker_control.stop_tracker();
        case 'KNN'
            py.tracker_control.start_tracker('matlab/models', sprintf('%s_%s_train_emissions.csv', algorithm, dataset));
            model = fitcknn(X_train, y_train, 'NumNeighbors', 5);
            py.tracker_control.stop_tracker();
        case 'SVC'
            py.tracker_control.start_tracker('matlab/models', sprintf('%s_%s_train_emissions.csv', algorithm, dataset));
            model = fitcecoc(X_train, y_train, 'Learners', templateSVM('KernelFunction', 'linear'));
            py.tracker_control.stop_tracker();
        case 'naiveBayes'
            py.tracker_control.start_tracker('matlab/models', sprintf('%s_%s_train_emissions.csv', algorithm, dataset));
            model = fitcnb(X_train, y_train, 'DistributionNames', 'kernel');
            py.tracker_control.stop_tracker();
        otherwise
            error('Algorithm unknown');
    end

    py.tracker_control.stop_tracker();

    py.tracker_control.start_tracker('matlab/models', sprintf('%s_%s_test_emissions.csv', algorithm, dataset));

    y_pred = predict(model, X_test);

    py.tracker_control.stop_tracker();

    % Calcola l'accuratezza
    accuracy = sum(y_pred == y_test) / length(y_test);
    fprintf('Accuracy for %s on %s: %.2f%%\n', algorithm, dataset, accuracy * 100);
end
