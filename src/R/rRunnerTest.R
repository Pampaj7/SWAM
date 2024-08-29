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

set.seed(42)


dataIris <- read.csv("../datasets/iris/iris.csv")
dataBreastCancer <- read.csv("../datasets/breastcancer/breastcancer.csv")
dataWine <- read.csv("../datasets/winequality/wine_data.csv")

load_split_and_predict_random_forest <- function(data, target, model_path, train_split = 0.8, seed = 42) {
  # Convert the target column to a factor
  data[[target]] <- as.factor(data[[target]])

  # Split the data into training and test sets
  set.seed(seed)
  trainIndex <- createDataPartition(data[[target]], p = train_split, list = FALSE)
  testData <- data[-trainIndex, ]  # Use the test set for predictions

  # Load the trained model from the file
  rfModel <- readRDS(model_path)

  # Predict on the test data set
  predictions <- predict(rfModel, newdata = testData)
}
load_split_and_predict_decision_tree <- function(data, target, model_path, train_split = 0.8, seed = 42) {
  # Convert the target column to a factor if it's not numeric
  if(!is.numeric(data[[target]])) {
    data[[target]] <- as.factor(data[[target]])
  }

  # Split the data into training and test sets
  set.seed(seed)  # For reproducibility
  trainIndex <- createDataPartition(data[[target]], p = train_split, list = FALSE)
  testData <- data[-trainIndex, ]  # Use the test set for predictions

  # Load the trained model from the file
  dtModel <- readRDS(model_path)

  # Predict on the test data set
  predictions <- predict(dtModel, newdata = testData, type = "class")
}
load_and_predict_knn <- function(data, target, model_path, train_split = 0.8, seed = 42) {
  # Load the trained model parameters from the file
  model_info <- readRDS(model_path)
  trainX <- model_info$trainX
  trainY <- model_info$trainY
  k <- model_info$k

  # Convert the target column to a factor if it's not numeric
  if (!is.numeric(data[[target]])) {
    data[[target]] <- as.factor(data[[target]])
  }

  # Split the data into training and test sets
  set.seed(seed)  # For reproducibility
  trainIndex <- createDataPartition(data[[target]], p = train_split, list = FALSE)
  testData <- data[-trainIndex, ]

  # Extract predictors for test data
  testX <- testData[ , !names(testData) %in% target]

  # Predict using k-NN
  predictions <- knn(train = trainX, test = testX, cl = trainY, k = k)
}
load_split_and_predict_logistic_regression <- function(data, target, model_path, train_split = 0.8, seed = 42) {
  # Ensure the target column is a factor
  data[[target]] <- as.factor(data[[target]])

  # Determine the number of unique levels in the target variable
  num_levels <- length(levels(data[[target]]))

  # Split the data into training and testing sets
  set.seed(seed)
  split <- sample.split(data[[target]], SplitRatio = train_split)
  test_data <- subset(data, split == FALSE)

  # Standardize predictors (excluding target and ID)
  predictors <- names(data)[!names(data) %in% c(target, "ID")]
  test_data[predictors] <- scale(test_data[predictors])

  # Prepare data for glmnet
  x_test <- as.matrix(test_data[predictors])
  y_test <- as.factor(test_data[[target]])

  # Load the trained model from the file
  model <- readRDS(model_path)

  if (num_levels == 2) {
    # Make predictions for binary classification
    predictions <- predict(model, x_test, type = "response")
    predicted_class <- ifelse(predictions > 0.5, levels(y_test)[2], levels(y_test)[1])
  } else if (num_levels > 2) {
    # Make predictions for multi-class classification
    predictions <- predict(model, x_test, type = "class")
    predicted_class <- predictions[, 1]
  } else {
    stop("The target variable must have at least one level.")
  }
}
load_split_and_predict_svc_classifier <- function(data, target, model_path, train_split = 0.8, seed = 42) {
  # Ensure the target column is a factor
  data[[target]] <- as.factor(data[[target]])

  # Determine the number of unique levels in the target variable
  num_levels <- length(levels(data[[target]]))

  # Split the data into training and testing sets
  set.seed(seed)
  split <- sample.split(data[[target]], SplitRatio = train_split)
  test_data <- subset(data, split == FALSE)

  # Standardize predictors (excluding target and ID)
  predictors <- names(data)[!names(data) %in% c(target, "ID")]
  test_data[predictors] <- scale(test_data[predictors])

  # Prepare data for svm
  x_test <- test_data[predictors]
  y_test <- test_data[[target]]

  # Load the trained model from the file
  model <- readRDS(model_path)

  if (num_levels == 2 || num_levels > 2) {
    # Make predictions for both binary and multi-class classification
    predictions <- predict(model, x_test)
  } else {
    stop("The target variable must have at least one level.")
  }
}


run_model_with_dataset <- function(datasetName, algorithmName){
  dataset <- switch (datasetName,
    "iris" = list(data = dataIris, target = "Species"),
    "breastCancer" = list(data = dataBreastCancer, target = "diagnosis"),
    "wine" = list(data = dataWine, target = "quality"),
    stop("invalid dataset name")
  )
  dataMatrix <- dataset$data
  targetValue <- dataset$target

  result <- switch (algorithmName,
    "randomForest" = train_random_forest(data = dataMatrix, target = targetValue),
    "decisionTree" = train_decision_tree(data = dataMatrix, target = targetValue),
    "KNN" = train_knn(data = dataMatrix, target = targetValue),
    "logisticRegression" = train_logistic_regression(data = dataMatrix, target = targetValue),
    "SVC" = train_svc_classifier(data = dataMatrix, target = targetValue),
    stop("invalid algorithm name")
  )
  return(result)
}

args <- commandArgs(trailingOnly = TRUE)
dataset <- args[1]
algorithm <- args[2]

run_model_with_dataset(datasetName = dataset, algorithmName = algorithm)

