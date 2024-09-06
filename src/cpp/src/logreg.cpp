#include "pythonLinker.h"
#import <Python.h>
#include <armadillo>
#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/core/data/scaler_methods/standard_scaler.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/core/util/io.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>

using namespace arma;
void SetSeedLR(int seed) {
  // Set the random seed for reproducibility
  arma::arma_rng::set_seed(seed);
  mlpack::RandomSeed(seed);
}

// Function to train the logistic regression model
void TrainLogisticRegression(std::pair<arma::mat, arma::Row<size_t>> data) {

  Py_Initialize(); // Initialize the Python Interpreter
  PyObject *pArgsStart = PyTuple_Pack(2, PyUnicode_FromString("output"),
                                      PyUnicode_FromString("emissions.csv"));
  PyObject *pArgsStop = PyTuple_New(0);

  SetSeedLR(42);
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

  // Call start_tracker

  scaler.Fit(trainX);

  scaler.Transform(trainX, trainX);

  // Train the logistic regression model
  CallPython("tracker_control", "Tracker", "start_tracker", pArgsStart);
  mlpack::LogisticRegression<> logreg(trainX, trainY);
  CallPython("tracker_control", "Tracker", "stop_tracker", pArgsStop);


  Py_DECREF(pArgsStart);

  // Save the model
  mlpack::data::Save("./logistic_regression_model.bin", "logreg_model", logreg);
}

// Function to test the logistic regression model
void TestLogisticRegression(std::pair<arma::mat, arma::Row<size_t>> data) {
  try {
    Py_Initialize(); // Initialize the Python Interpreter
    PyObject *pArgsStart = PyTuple_Pack(2, PyUnicode_FromString("output"),
                                        PyUnicode_FromString("emissions.csv"));
    PyObject *pArgsStop = PyTuple_New(0);

    SetSeedLR(42);
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
    CallPython("tracker_control", "Tracker", "start_tracker", pArgsStart);
    logreg.Classify(testX, predictions);
    CallPython("tracker_control", "Tracker", "stop_tracker", pArgsStop);

    // Calculate accuracy
    double accuracy = accu(predictions == testY) / (double)testY.n_elem;
    std::cout << "Logistic Regression Test Accuracy: " << accuracy * 100 << "%"
              << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
}
