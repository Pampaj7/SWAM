#include <armadillo>
#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/core/data/scaler_methods/standard_scaler.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/core/util/io.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>

using namespace arma;

// Function to train the logistic regression model
void TrainLogisticRegression(std::pair<arma::mat, arma::Row<size_t>> data) {
  // Load dataset
  //
  arma::mat x = data.first;
  arma::Row<size_t> y = data.second;
  // Check the dimensions of the dataset
  if (x.n_cols != y.n_elem) {
    throw std::runtime_error(
        "Mismatch between feature matrix and target vector size.");
  }

  // Split the data into training and testing sets
  arma::mat trainX, testX;
  arma::Row<size_t> trainY, testY;
  mlpack::data::Split(x, y, trainX, testX, trainY, testY, 0.2,
                      true); // 80% training, 20% testing

  // Standardize the training data
  mlpack::data::StandardScaler scaler;
  scaler.Fit(trainX);
  scaler.Transform(trainX, trainX);

  // Train the logistic regression model
  mlpack::LogisticRegression<> logreg(trainX, trainY);

  // Save the model
  mlpack::data::Save("./logistic_regression_model.bin", "logreg_model", logreg);
}

// Function to test the logistic regression model
void TestLogisticRegression(std::pair<arma::mat, arma::Row<size_t>> data) {
  try {
    arma::mat x = data.first;
    arma::Row<size_t> y = data.second;

    // Check the dimensions of the dataset
    if (x.n_cols != y.n_elem) {
      throw std::runtime_error(
          "Mismatch between feature matrix and target vector size.");
    }

    // Load the trained logistic regression model
    mlpack::LogisticRegression<> logreg;
    mlpack::data::Load("./logistic_regression_model.bin", "logreg_model",
                       logreg);

    // Split the data into training and testing sets
    arma::mat trainX, testX;
    arma::Row<size_t> trainY, testY;
    mlpack::data::Split(x, y, trainX, testX, trainY, testY, 0.2,
                        true); // 80% training, 20% testing

    // Standardize the test data using the same scaler
    mlpack::data::StandardScaler scaler;
    scaler.Fit(trainX);
    scaler.Transform(testX, testX);

    // Predict on the test set
    arma::Row<size_t> predictions;
    logreg.Classify(testX, predictions);

    // Calculate accuracy
    double accuracy = accu(predictions == testY) / (double)testY.n_elem;
    std::cout << "Logistic Regression Test Accuracy: " << accuracy * 100 << "%"
              << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
}
