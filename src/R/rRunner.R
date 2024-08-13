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



dataIris = read.csv("/Users/pippodima/PycharmProjects/SWAM/datasets/iris/iris.csv")
dataBreastCancer = read.csv("/Users/pippodima/PycharmProjects/SWAM/datasets/breastcancer/breastcancer.csv")
dataWine = read.csv("/Users/pippodima/PycharmProjects/SWAM/datasets/winequality/wine_data.csv")
set.seed(42)


train_random_forest <- function(data, target, train_split = 0.8, ntree = 100, mtry = 3, seed=42) {
  # Convert the target column to a factor
  data[[target]] <- as.factor(data[[target]])

  # Split the data into training and test sets
  set.seed(seed)
  trainIndex <- createDataPartition(data[[target]], p = train_split, list = FALSE)
  trainData <- data[trainIndex, ]
  testData <- data[-trainIndex, ]

  # Train the Random Forest model
  formula <- as.formula(paste(target, "~ ."))
  rfModel <- randomForest(formula, data = trainData, ntree = ntree, mtry = mtry, importance = TRUE)

  # Plot the Random Forest model error rates
  plot(rfModel)

  # Predict on the test set
  predictions <- predict(rfModel, newdata = testData)

  # Compute the confusion matrix
  confMatrix <- confusionMatrix(predictions, testData[[target]])

  # Print the confusion matrix
  print(confMatrix)

  # Return the trained model and the confusion matrix
  return(list(model = rfModel, confusion_matrix = confMatrix))
}

rfIris = train_random_forest(data = dataIris, target = "Species")
rfCancer = train_random_forest(data = dataBreastCancer, target = "diagnosis")
rfWine = train_random_forest(data = dataWine, target = "type")


train_decision_tree <- function(data, target, train_split = 0.8, minsplit = 20, cp = 0.01, seed = 42) {
  # Convert the target column to a factor if it's not numeric
  if(!is.numeric(data[[target]])) {
    data[[target]] <- as.factor(data[[target]])
  }

  # Split the data into training and test sets
  set.seed(seed)  # For reproducibility
  trainIndex <- createDataPartition(data[[target]], p = train_split, list = FALSE)
  trainData <- data[trainIndex, ]
  testData <- data[-trainIndex, ]

  # Train the Decision Tree model
  formula <- as.formula(paste(target, "~ ."))
  dtModel <- rpart(formula, data = trainData, method = "class", control = rpart.control(minsplit = minsplit, cp = cp))

  # Plot the Decision Tree
  rpart.plot(dtModel, main = paste("Decision Tree for", target))

  # Predict on the test set
  predictions <- predict(dtModel, newdata = testData, type = "class")

  # Compute the confusion matrix or calculate RMSE for numeric targets
  if (is.factor(data[[target]])) {
    confMatrix <- confusionMatrix(predictions, testData[[target]])
    print(confMatrix)
  } else {
    # For regression models, calculate RMSE
    rmse <- sqrt(mean((predict(dtModel, newdata = testData) - testData[[target]])^2))
    print(paste("RMSE:", rmse))
  }

  # Return the trained model and the performance metrics
  return(list(model = dtModel, performance = if (is.factor(data[[target]])) confMatrix else rmse))
}

dtIris=train_decision_tree(data = dataIris, target = "Species")
dtCancer=train_decision_tree(data = dataBreastCancer, target = "diagnosis")
dtWine=train_decision_tree(data = dataWine, target = "type")


train_knn <- function(data, target, train_split = 0.8, k = 5, seed = 42) {
  # Convert the target column to a factor if it's not numeric
  if (!is.numeric(data[[target]])) {
    data[[target]] <- as.factor(data[[target]])
  }

  # Split the data into training and test sets
  set.seed(seed)  # For reproducibility
  trainIndex <- createDataPartition(data[[target]], p = train_split, list = FALSE)
  trainData <- data[trainIndex, ]
  testData <- data[-trainIndex, ]

  # Extract predictors and target
  trainX <- trainData[ , !names(trainData) %in% target]
  trainY <- trainData[[target]]
  testX <- testData[ , !names(testData) %in% target]
  testY <- testData[[target]]

  # Train the k-NN model
  # Note: k-NN training is done during prediction in the class package
  predictions <- knn(train = trainX, test = testX, cl = trainY, k = k)

  # Compute the confusion matrix or calculate RMSE for numeric targets
  if (is.factor(data[[target]])) {
    confMatrix <- confusionMatrix(predictions, testY)
    print(confMatrix)
  } else {
    # For regression models, calculate RMSE
    rmse <- sqrt(mean((as.numeric(predictions) - as.numeric(testY))^2))
    print(paste("RMSE:", rmse))
  }

  # Return the performance metrics
  return(if (is.factor(data[[target]])) confMatrix else rmse)
}

knnIris = train_knn(data = dataIris, target = "Species")
knnCancer = train_knn(data = dataBreastCancer, target = "diagnosis")
knnWine = train_knn(data = dataWine, target = "type")


train_logistic_regression <- function(data, target, train_split = 0.8, seed = 42) {
  # Ensure the target column is a factor
  data[[target]] <- as.factor(data[[target]])

  # Determine the number of unique levels in the target variable
  num_levels <- length(levels(data[[target]]))

  # Split the data into training and testing sets
  set.seed(seed)
  split <- sample.split(data[[target]], SplitRatio = train_split)
  train_data <- subset(data, split == TRUE)
  test_data <- subset(data, split == FALSE)

  # Standardize predictors (excluding target and ID)
  predictors <- names(data)[!names(data) %in% c(target, "ID")]
  train_data[predictors] <- scale(train_data[predictors])
  test_data[predictors] <- scale(test_data[predictors])

  # Prepare data for glmnet
  x_train <- as.matrix(train_data[predictors])
  y_train <- as.factor(train_data[[target]])
  x_test <- as.matrix(test_data[predictors])
  y_test <- as.factor(test_data[[target]])

  if (num_levels == 2) {
    # Binary Logistic Regression with glmnet
    model <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1)

    # Make predictions
    predictions <- predict(model, x_test, type = "response")
    predicted_class <- ifelse(predictions > 0.5, levels(y_test)[2], levels(y_test)[1])

  } else if (num_levels > 2) {
    # Multi-Class Logistic Regression with glmnet
    model <- cv.glmnet(x_train, y_train, family = "multinomial", alpha = 1)

    # Make predictions
    predictions <- predict(model, x_test, type = "class")
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

lrIris = train_logistic_regression(data = dataIris, target = "Species")
lrCancer = train_logistic_regression(data = dataBreastCancer, target = "diagnosis")
lrWine = train_logistic_regression(data = dataWine, target = "type")


train_xgboost_classifier <- function(data, target, train_split = 0.8, seed = 42) {
  # Ensure the target column is a factor
  data[[target]] <- as.factor(data[[target]])

  # Determine the number of unique levels in the target variable
  num_levels <- length(levels(data[[target]]))

  # Split the data into training and testing sets
  set.seed(seed)
  split <- sample.split(data[[target]], SplitRatio = train_split)
  train_data <- subset(data, split == TRUE)
  test_data <- subset(data, split == FALSE)

  # Standardize predictors (excluding target and ID)
  predictors <- names(data)[!names(data) %in% c(target, "ID")]
  train_data[predictors] <- scale(train_data[predictors])
  test_data[predictors] <- scale(test_data[predictors])

  # Prepare data for xgboost
  x_train <- as.matrix(train_data[predictors])
  y_train <- as.numeric(train_data[[target]]) - 1  # xgboost requires numeric labels (0-based)
  x_test <- as.matrix(test_data[predictors])
  y_test <- as.numeric(test_data[[target]]) - 1  # xgboost requires numeric labels (0-based)

  if (num_levels == 2) {
    # Binary Classification
    params <- list(
      objective = "binary:logistic",
      eval_metric = "logloss"
    )
    model <- xgboost(data = x_train, label = y_train, params = params, nrounds = 100, verbose = 0)

    # Make predictions
    predictions <- predict(model, x_test)
    predicted_class <- ifelse(predictions > 0.5, 1, 0)

  } else if (num_levels > 2) {
    # Multi-Class Classification
    params <- list(
      objective = "multi:softmax",
      num_class = num_levels,
      eval_metric = "mlogloss"
    )
    model <- xgboost(data = x_train, label = y_train, params = params, nrounds = 100, verbose = 0)

    # Make predictions
    predictions <- predict(model, x_test)
    predicted_class <- as.numeric(predictions)  # xgboost returns class labels directly

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

xgbIris = train_xgboost_classifier(data = dataIris, target = "Species")
xgbCancer = train_xgboost_classifier(data = dataBreastCancer, target = "diagnosis")
xgbWine = train_xgboost_classifier(data = dataWine, target = "type")

train_svc_classifier <- function(data, target, train_split = 0.8, seed = 42) {
  # Ensure the target column is a factor
  data[[target]] <- as.factor(data[[target]])

  # Determine the number of unique levels in the target variable
  num_levels <- length(levels(data[[target]]))

  # Split the data into training and testing sets
  set.seed(seed)
  split <- sample.split(data[[target]], SplitRatio = train_split)
  train_data <- subset(data, split == TRUE)
  test_data <- subset(data, split == FALSE)

  # Standardize predictors (excluding target and ID)
  predictors <- names(data)[!names(data) %in% c(target, "ID")]
  train_data[predictors] <- scale(train_data[predictors])
  test_data[predictors] <- scale(test_data[predictors])

  # Prepare data for svm

  x_train <- train_data[predictors]
  y_train <- train_data[[target]]
  x_test <- test_data[predictors]
  y_test <- test_data[[target]]

  if (num_levels == 2) {
    # Binary Classification
    model <- svm(x_train, y_train, type = "C-classification", kernel = "radial")

    # Make predictions
    predictions <- predict(model, x_test)

  } else if (num_levels > 2) {
    # Multi-Class Classification
    model <- svm(x_train, y_train, type = "C-classification", kernel = "radial", decision.values = TRUE)

    # Make predictions
    predictions <- predict(model, x_test)

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

svcIris = train_svc_classifier(data = dataIris, target = "Species")
svcCancer = train_svc_classifier(data = dataBreastCancer, target = "diagnosis")
svcWine = train_svc_classifier(data = dataWine, target = "type")

train_gaussian_mixture <- function(data, target, train_split = 0.8, seed = 42) {
  # Ensure the target column is a factor
  data[[target]] <- as.factor(data[[target]])

  # Determine the number of unique levels in the target variable
  num_levels <- length(levels(data[[target]]))

  # Split the data into training and testing sets
  set.seed(seed)
  split <- sample.split(data[[target]], SplitRatio = train_split)
  train_data <- subset(data, split == TRUE)
  test_data <- subset(data, split == FALSE)

  # Standardize predictors (excluding target and ID)
  predictors <- names(data)[!names(data) %in% c(target, "ID")]
  train_data[predictors] <- scale(train_data[predictors])
  test_data[predictors] <- scale(test_data[predictors])

  # Prepare data for Mclust
  x_train <- train_data[predictors]
  y_train <- train_data[[target]]
  x_test <- test_data[predictors]
  y_test <- test_data[[target]]

  # Fit Gaussian Mixture Model
  model <- Mclust(x_train, G = num_levels)

  # Make predictions
  predictions <- predict(model, x_test)$classification
  predicted_class <- factor(predictions, levels = 1:num_levels, labels = levels(y_test))

  # Create confusion matrix
  confusion_matrix <- table(Predicted = predicted_class, Actual = y_test)

  # Return model and confusion matrix
  return(list(
    model = model,
    confusion_matrix = confusion_matrix
  ))
}

gmIris = train_gaussian_mixture(data = dataIris, target = "Species")
gmCancer = train_gaussian_mixture(data = dataBreastCancer, target = "diagnosis")
gmWine = train_gaussian_mixture(data = dataWine, target = "type")



