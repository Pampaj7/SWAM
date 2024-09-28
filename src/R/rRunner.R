library(randomForest)
library(caret)
library(rpart)
library(rpart.plot)
library(class)
library(glmnet)
library(caTools)
library(nnet)
library(xgboost)
library(e1071)
library(mclust)
library(reticulate)
library(adabag)


use_python('/Users/pampaj/anaconda3/envs/sw/bin/python', required = TRUE)
source_python("../matlab/tracker_control.py")

dataIris <- read.csv("../../datasets/iris/iris_processed.csv")
dataBreastCancer <- read.csv("../../datasets/breastcancer/breastCancer_processed.csv")
dataWine <- read.csv("../../datasets/winequality/wineQuality_processed.csv")



train_random_forest <- function(data, target, savePath, fileName, train_split = 0.8, ntree = 100) {
  # Convert the target column to a factor
  data[[target]] <- as.factor(data[[target]])

  # Split the data into training and test sets
  trainIndex <- createDataPartition(data[[target]], p = train_split, list = FALSE)
  trainData <- data[trainIndex, ]
  testData <- data[-trainIndex, ]

  # Train the Random Forest model

  formula <- as.formula(paste(target, "~ ."))

  start_tracker(savePath, paste(fileName, "train", "emissions.csv", sep = "_"))
  rfModel <- randomForest(formula, data = trainData, ntree = ntree, importance = TRUE)
  stop_tracker()

  # Plot the Random Forest model error rates
  # plot(rfModel)

  # Predict on the test set
  start_tracker(savePath, paste(fileName, "test", "emissions.csv", sep = "_"))
  predictions <- predict(rfModel, newdata = testData)
  stop_tracker()
  # Compute the confusion matrix
  confMatrix <- confusionMatrix(predictions, testData[[target]])

  # Return the trained model and the confusion matrix
  return(list(model = rfModel, confusion_matrix = confMatrix))
}
train_decision_tree <- function(data, target, savePath, fileName, train_split = 0.8) {

  # Split the data into training and test sets
  trainIndex <- createDataPartition(data[[target]], p = train_split, list = FALSE)
  trainData <- data[trainIndex, ]
  testData <- data[-trainIndex, ]

  # Train the Decision Tree model
  formula <- as.formula(paste(target, "~ ."))

  start_tracker(savePath, paste(fileName, "train", "emissions.csv", sep = "_"))
  dtModel <- rpart(formula, data = trainData, method = "class")
  stop_tracker()
  # Plot the Decision Tree
  # rpart.plot(dtModel, main = paste("Decision Tree for", target))

  # Predict on the test set
  start_tracker(savePath, paste(fileName, "test", "emissions.csv", sep = "_"))
  predictions <- predict(dtModel, newdata = testData, type = "class")
  stop_tracker()
  # Compute the confusion matrix or calculate RMSE for numeric targets
  if (is.factor(data[[target]])) {
    confMatrix <- confusionMatrix(predictions, testData[[target]])
  } else {
    # For regression models, calculate RMSE
    rmse <- sqrt(mean((predict(dtModel, newdata = testData) - testData[[target]])^2))
  }

  # Return the trained model and the performance metrics
  return(list(model = dtModel, performance = if (is.factor(data[[target]])) confMatrix else rmse))
}
train_knn <- function(data, target, savePath, fileName, train_split = 0.8, k = 5) {

  # Split the data into training and test sets
  trainIndex <- createDataPartition(data[[target]], p = train_split, list = FALSE)
  trainData <- data[trainIndex, ]
  testData <- data[-trainIndex, ]

  # Extract predictors and target
  # k-NN does not have a traditional training phase but requires storing the training data

  start_tracker(savePath, paste(fileName, "train", "emissions.csv", sep = "_"))
  trainX <- trainData[ , !names(trainData) %in% target]
  trainY <- trainData[[target]]
  stop_tracker()

  testX <- testData[ , !names(testData) %in% target]
  testY <- testData[[target]]

  # Train the k-NN model

  print("inizio test knn")
  # Note: k-NN training is done during prediction in the class package
  start_tracker(savePath, paste(fileName, "test", "emissions.csv", sep = "_"))
  predictions <- knn(train = trainX, test = testX, cl = trainY, k = k)
  stop_tracker()

  # Compute the confusion matrix or calculate RMSE for numeric targets
  if (is.factor(data[[target]])) {
    confMatrix <- confusionMatrix(predictions, testY)
  } else {
    rmse <- sqrt(mean((as.numeric(predictions) - as.numeric(testY))^2))
  }

  # Return the performance metrics
  return(if (is.factor(data[[target]])) confMatrix else rmse)
}
train_logistic_regression <- function(data, target, savePath, fileName, train_split = 0.8) {

    # Ensure the target column is a factor
    data[[target]] <- as.factor(data[[target]])

    # Determine the number of unique levels in the target variable
    num_levels <- length(levels(data[[target]]))

    # Split the data into training and testing sets
    split <- sample.split(data[[target]], SplitRatio = train_split)
    train_data <- subset(data, split == TRUE)
    test_data <- subset(data, split == FALSE)

    # Prepare predictors and check data types
    predictors <- names(data)[!names(data) %in% c(target, "ID")]
    train_data[predictors] <- lapply(train_data[predictors], function(x) {
        if (is.factor(x)) as.numeric(as.character(x)) else x
    })
    test_data[predictors] <- lapply(test_data[predictors], function(x) {
        if (is.factor(x)) as.numeric(as.character(x)) else x
    })

    # Remove non-numeric predictors if any
    predictors <- names(train_data)[sapply(train_data, is.numeric)]

    # Standardize predictors
    train_data[predictors] <- scale(train_data[predictors])
    test_data[predictors] <- scale(test_data[predictors])

    # Prepare data for glmnet
    x_train <- as.matrix(train_data[predictors])
    y_train <- as.factor(train_data[[target]])
    x_test <- as.matrix(test_data[predictors])
    y_test <- as.factor(test_data[[target]])

    if (num_levels == 2) {
        # Binary Logistic Regression with glmnet
        start_tracker(savePath, paste(fileName, "train", "emissions.csv", sep = "_"))
        model <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1)
        stop_tracker()

        # Make predictions
        start_tracker(savePath, paste(fileName, "test", "emissions.csv", sep = "_"))
        predictions <- predict(model, x_test, type = "response")
        stop_tracker()
        predicted_class <- ifelse(predictions > 0.5, levels(y_test)[2], levels(y_test)[1])

    } else if (num_levels > 2) {
        # Multi-Class Logistic Regression with glmnet
        start_tracker(savePath, paste(fileName, "train", "emissions.csv", sep = "_"))
        model <- cv.glmnet(x_train, y_train, family = "multinomial", alpha = 1)
        stop_tracker()
        # Make predictions
        start_tracker(savePath, paste(fileName, "test", "emissions.csv", sep = "_"))
        predictions <- predict(model, x_test, type = "class")
        stop_tracker()
        predicted_class <- predictions[, 1]

    } else {
        stop("The target variable must have at least one level.")
    }

    # Create confusion matrix
    confusion_matrix <- table(Predicted = predicted_class, Actual = y_test)

    # Return model and confusion matrix
    return(list(
        model = model,
        confusion_matrix = confusion_matrix
    ))
}
train_svc_classifier <- function(data, target, savePath, fileName, train_split = 0.8) {

    # Ensure the target column is a factor
    data[[target]] <- as.factor(data[[target]])

    # Determine the number of unique levels in the target variable
    num_levels <- length(levels(data[[target]]))

    # Split the data into training and testing sets
    split <- sample.split(data[[target]], SplitRatio = train_split)
    train_data <- subset(data, split == TRUE)
    test_data <- subset(data, split == FALSE)

    # Prepare predictors and check data types
    predictors <- names(data)[!names(data) %in% c(target, "ID")]
    train_data[predictors] <- lapply(train_data[predictors], function(x) {
        if (is.factor(x)) as.numeric(as.character(x)) else x
    })
    test_data[predictors] <- lapply(test_data[predictors], function(x) {
        if (is.factor(x)) as.numeric(as.character(x)) else x
    })

    # Remove non-numeric predictors if any
    predictors <- names(train_data)[sapply(train_data, is.numeric)]

    # Standardize predictors
    train_data[predictors] <- scale(train_data[predictors])
    test_data[predictors] <- scale(test_data[predictors])

    # Prepare data for svm
    x_train <- train_data[predictors]
    y_train <- train_data[[target]]
    x_test <- test_data[predictors]
    y_test <- test_data[[target]]

    if (num_levels == 2) {
        # Binary Classification
        start_tracker(savePath, paste(fileName, "train", "emissions.csv", sep = "_"))
        model <- svm(x_train, y_train, type = "C-classification", kernel = "radial")
        stop_tracker()

        # Make predictions
        start_tracker(savePath, paste(fileName, "test", "emissions.csv", sep = "_"))
        predictions <- predict(model, x_test)
        stop_tracker()

    } else if (num_levels > 2) {
        # Multi-Class Classification
        start_tracker(savePath, paste(fileName, "train", "emissions.csv", sep = "_"))
        model <- svm(x_train, y_train, type = "C-classification", kernel = "radial")
        stop_tracker()

        # Make predictions
        start_tracker(savePath, paste(fileName, "test", "emissions.csv", sep = "_"))
        predictions <- predict(model, x_test)
        stop_tracker()

    } else {
        stop("The target variable must have at least one level.")
    }

    # Create confusion matrix
    confusion_matrix <- table(Predicted = predictions, Actual = y_test)

    # Return model and confusion matrix
    return(list(
        model = model,
        confusion_matrix = confusion_matrix
    ))
}
train_naive_bayes <- function(data, target, savePath, fileName, train_split = 0.8) {
  # Convert the target column to a factor
  data[[target]] <- as.factor(data[[target]])

  # Split the data into training and test sets
  trainIndex <- createDataPartition(data[[target]], p = train_split, list = FALSE)
  trainData <- data[trainIndex, ]
  testData <- data[-trainIndex, ]

  # Train the Naive Bayes model
  formula <- as.formula(paste(target, "~ ."))

  start_tracker(savePath, paste(fileName, "train", "emissions.csv", sep = "_"))
  nbModel <- naiveBayes(formula, data = trainData)
  stop_tracker()

  # Predict on the test set
  start_tracker(savePath, paste(fileName, "test", "emissions.csv", sep = "_"))
  predictions <- predict(nbModel, newdata = testData)
  stop_tracker()

  # Compute the confusion matrix
  confMatrix <- confusionMatrix(predictions, testData[[target]])

  # Return the trained model and the confusion matrix
  return(list(model = nbModel, confusion_matrix = confMatrix))
}
train_adaboost <- function(data, target, savePath, fileName, train_split = 0.8, nIter = 100) {

  # Convert the target column to a factor
  data[[target]] <- as.factor(data[[target]])

  # Split the data into training and test sets
  trainIndex <- createDataPartition(data[[target]], p = train_split, list = FALSE)
  trainData <- data[trainIndex, ]
  testData <- data[-trainIndex, ]

  # Train the AdaBoost model
  formula <- as.formula(paste(target, "~ ."))

  start_tracker(savePath, paste(fileName, "train", "emissions.csv", sep = "_"))
  adaboostModel <- boosting(formula, data = trainData, boos = TRUE, mfinal = nIter)
  stop_tracker()

  # Predict on the test set
  start_tracker(savePath, paste(fileName, "test", "emissions.csv", sep = "_"))
  predictions <- predict.boosting(adaboostModel, newdata = testData)
  stop_tracker()

  # Compute the confusion matrix
  confMatrix <- confusionMatrix(predictions$class, testData[[target]])

  # Return the trained model and the confusion matrix
  return(list(model = adaboostModel, confusion_matrix = confMatrix))
}


run_model_with_dataset <- function(datasetName, algorithmName, savePath){
  dataset <- switch (datasetName,
    "iris" = dataIris,
    "breastCancer" = dataBreastCancer,
    "wine" = dataWine,
    stop("invalid dataset name")
  )
  dataMatrix <- dataset
  targetValue <- "target"
  name <- paste(algorithmName, datasetName, sep = "_")
  result <- switch (algorithmName,
    "randomForest" = train_random_forest(data = dataMatrix, target = targetValue, savePath = savePath, fileName = name),
    "decisionTree" = train_decision_tree(data = dataMatrix, target = targetValue, savePath = savePath, fileName = name),
    "KNN" = train_knn(data = dataMatrix, target = targetValue, savePath = savePath, fileName = name),
    "logisticRegression" = train_logistic_regression(data = dataMatrix, target = targetValue, savePath = savePath, fileName = name),
    "SVC" = train_svc_classifier(data = dataMatrix, target = targetValue, savePath = savePath, fileName = name),
    "naiveBayes" = train_naive_bayes(data = dataMatrix, target = targetValue, savePath = savePath, fileName = name),
    "adaBoost" = train_naive_bayes(data = dataMatrix, target = targetValue, savePath = savePath, fileName = name),
    stop("invalid algorithm name")
  )
  return(result)
}

args <- commandArgs(trailingOnly = TRUE)
dataset <- args[1]
algorithm <- args[2]
savePath <- args[3]

run_model_with_dataset(datasetName = dataset, algorithmName = algorithm, savePath = savePath)

