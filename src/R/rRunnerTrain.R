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

fit_and_save_random_forest <- function(data, target, model_path, train_split = 0.8, ntree = 100, mtry = 3, seed = 42) {
  # Convert the target column to a factor
  data[[target]] <- as.factor(data[[target]])

  # Split the data into training and test sets
  set.seed(seed)
  trainIndex <- createDataPartition(data[[target]], p = train_split, list = FALSE)
  trainData <- data[trainIndex, ]

  # Train the Random Forest model
  formula <- as.formula(paste(target, "~ ."))
  rfModel <- randomForest(formula, data = trainData, ntree = ntree, mtry = mtry)

  # Save the trained model to a file
  saveRDS(rfModel, file = model_path)
}
fit_and_save_decision_tree <- function(data, target, model_path, train_split = 0.8, minsplit = 20, cp = 0.01, seed = 42) {
  # Convert the target column to a factor if it's not numeric
  if(!is.numeric(data[[target]])) {
    data[[target]] <- as.factor(data[[target]])
  }

  # Split the data into training and test sets
  set.seed(seed)  # For reproducibility
  trainIndex <- createDataPartition(data[[target]], p = train_split, list = FALSE)
  trainData <- data[trainIndex, ]

  # Train the Decision Tree model
  formula <- as.formula(paste(target, "~ ."))
  dtModel <- rpart(formula, data = trainData, method = "class", control = rpart.control(minsplit = minsplit, cp = cp))

  # Save the trained model to a file
  saveRDS(dtModel, file = model_path)
}
fit_and_save_knn <- function(data, target, model_path, train_split = 0.8, k = 5, seed = 42) {
  # Convert the target column to a factor if it's not numeric
  if (!is.numeric(data[[target]])) {
    data[[target]] <- as.factor(data[[target]])
  }

  # Split the data into training and test sets
  set.seed(seed)  # For reproducibility
  trainIndex <- createDataPartition(data[[target]], p = train_split, list = FALSE)
  trainData <- data[trainIndex, ]

  # Extract predictors and target
  trainX <- trainData[ , !names(trainData) %in% target]
  trainY <- trainData[[target]]

  # Save the model parameters to a file
  model_info <- list(trainX = trainX, trainY = trainY, k = k)
  saveRDS(model_info, file = model_path)
}
fit_and_save_logistic_regression <- function(data, target, model_path, train_split = 0.8, seed = 42) {
  # Ensure the target column is a factor
  data[[target]] <- as.factor(data[[target]])

  # Determine the number of unique levels in the target variable
  num_levels <- length(levels(data[[target]]))

  # Split the data into training and testing sets
  set.seed(seed)
  split <- sample.split(data[[target]], SplitRatio = train_split)
  train_data <- subset(data, split == TRUE)

  # Standardize predictors (excluding target and ID)
  predictors <- names(data)[!names(data) %in% c(target, "ID")]
  train_data[predictors] <- scale(train_data[predictors])

  # Prepare data for glmnet
  x_train <- as.matrix(train_data[predictors])
  y_train <- as.factor(train_data[[target]])

  if (num_levels == 2) {
    # Binary Logistic Regression with glmnet
    model <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1)
  } else if (num_levels > 2) {
    # Multi-Class Logistic Regression with glmnet
    model <- cv.glmnet(x_train, y_train, family = "multinomial", alpha = 1)
  } else {
    stop("The target variable must have at least one level.")
  }

  # Save the trained model to a file
  saveRDS(model, file = model_path)
}
fit_and_save_svc_classifier <- function(data, target, model_path, train_split = 0.8, seed = 42) {
  # Ensure the target column is a factor
  data[[target]] <- as.factor(data[[target]])

  # Determine the number of unique levels in the target variable
  num_levels <- length(levels(data[[target]]))

  # Split the data into training and testing sets
  set.seed(seed)
  split <- sample.split(data[[target]], SplitRatio = train_split)
  train_data <- subset(data, split == TRUE)

  # Standardize predictors (excluding target and ID)
  predictors <- names(data)[!names(data) %in% c(target, "ID")]
  train_data[predictors] <- scale(train_data[predictors])

  # Prepare data for svm
  x_train <- train_data[predictors]
  y_train <- train_data[[target]]

  if (num_levels == 2) {
    # Binary Classification
    model <- svm(x_train, y_train, type = "C-classification", kernel = "radial")
  } else if (num_levels > 2) {
    # Multi-Class Classification
    model <- svm(x_train, y_train, type = "C-classification", kernel = "radial", decision.values = TRUE)
  } else {
    stop("The target variable must have at least one level.")
  }

  # Save the trained model to a file
  saveRDS(model, file = model_path)
}


run_model_with_dataset <- function(datasetName, algorithmName, pathName){
  dataset <- switch (datasetName,
    "iris" = list(data = dataIris, target = "Species"),
    "breastCancer" = list(data = dataBreastCancer, target = "diagnosis"),
    "wine" = list(data = dataWine, target = "quality"),
    stop("invalid dataset name")
  )
  dataMatrix <- dataset$data
  targetValue <- dataset$target

  result <- switch (algorithmName,
    "randomForest" = fit_and_save_random_forest(data = dataMatrix, target = targetValue, model_path = pathName),
    "decisionTree" = fit_and_save_decision_tree(data = dataMatrix, target = targetValue, model_path = pathName),
    "KNN" = fit_and_save_knn(data = dataMatrix, target = targetValue, model_path = pathName),
    "logisticRegression" = fit_and_save_logistic_regression(data = dataMatrix, target = targetValue, model_path = pathName),
    "SVC" = fit_and_save_svc_classifier(data = dataMatrix, target = targetValue, model_path = pathName),
    stop("invalid algorithm name")
  )
  return(result)
}

args <- commandArgs(trailingOnly = TRUE)
dataset <- args[1]
algorithm <- args[2]
path <- args[3]

run_model_with_dataset(datasetName = dataset, algorithmName = algorithm, pathName = path)

