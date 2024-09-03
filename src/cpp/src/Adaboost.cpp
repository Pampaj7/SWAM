#include "PythonLinker.h"
#import <Python.h>
#include <armadillo>
#include <iostream>
#include <mlpack/core.hpp>
#include <vector>
#include <xgboost/c_api.h>
// Error handling function for XGBoost API calls
inline void safe_xgboost(int err) {
  if (err != 0) {
    std::cerr << XGBGetLastError() << std::endl;
    exit(1);
  }
}

void SetSeedAda(unsigned long seed) { arma::arma_rng::set_seed(seed); }

// Function to convert arma::mat to float*
std::vector<float> convert_to_float(const arma::mat &mat) {
  std::vector<float> result(mat.n_elem);
  for (size_t i = 0; i < mat.n_elem; ++i) {
    result[i] = static_cast<float>(mat(i));
  }
  return result;
}

// Function to train the XGBoost AdaBoost model
void TrainAdaBoost(const std::pair<arma::mat, arma::Row<size_t>> &data) {
  try {
    Py_Initialize(); // Initialize the Python Interpreter
    PyObject *pArgsStart = PyTuple_Pack(2, PyUnicode_FromString("output"),
                                        PyUnicode_FromString("emissions.csv"));
    PyObject *pArgsStop = PyTuple_New(0);

    SetSeedAda(42);

    // Load dataset
    const arma::mat &X = data.first;
    const arma::Row<size_t> &y = data.second;

    // Check the dimensions of the dataset
    if (X.n_cols != y.n_elem) {
      throw std::runtime_error(
          "Mismatch between feature matrix and target vector size.");
    }

    // Split the data into training and testing sets (80% training, 20% testing)
    arma::mat trainX, testX;
    arma::Row<size_t> trainY, testY;
    mlpack::data::Split(X, y, trainX, testX, trainY, testY, 0.2, true);

    // Ensure trainX is in column-major format (each column is a sample)
    if (trainX.n_rows != X.n_rows) {
      trainX = trainX.t();
    }

    std::cout << "Training data dimensions: " << trainX.n_rows << " x "
              << trainX.n_cols << std::endl;

    // Convert data to float*
    std::vector<float> trainX_float = convert_to_float(trainX);

    // Convert data to DMatrix
    DMatrixHandle dtrain;
    safe_xgboost(XGDMatrixCreateFromMat(trainX_float.data(), trainX.n_cols,
                                        trainX.n_rows, 0, &dtrain));

    // Convert labels to float* (XGBoost prefers float labels)
    std::vector<float> trainY_float(trainY.n_elem);
    for (size_t i = 0; i < trainY.n_elem; ++i) {
      trainY_float[i] = static_cast<float>(trainY(i));
    }

    // Set labels
    safe_xgboost(XGDMatrixSetFloatInfo(dtrain, "label", trainY_float.data(),
                                       trainY_float.size()));

    // Set parameters
    const char *params = "objective=binary:logistic,booster=gbtree,eta=0.3,max_"
                         "depth=6,num_boost_round=100,tree_method=hist";

    // Create and train the model
    BoosterHandle booster;
    safe_xgboost(XGBoosterCreate(&dtrain, 1, &booster));
    safe_xgboost(XGBoosterSetParam(booster, "nthread",
                                   "0")); // Use all available threads

    // Train for multiple iterations
    // for (int i = 0; i < 10; ++i) {  // Adjust the number of iterations as
    // needed
    CallPython("tracker_control", "Tracker", "start_tracker", pArgsStart);
    safe_xgboost(XGBoosterUpdateOneIter(booster, 1, dtrain));
    CallPython("tracker_control", "Tracker", "stop_tracker", pArgsStop);
    // }

    // Save the model
    safe_xgboost(XGBoosterSaveModel(booster, "xgboost_adaboost_model.json"));

    // Clean up
    safe_xgboost(XGBoosterFree(booster));
    safe_xgboost(XGDMatrixFree(dtrain));

  } catch (const std::exception &e) {
    std::cerr << "Error in TrainAdaBoost: " << e.what() << std::endl;
  }
}

// Function to test the XGBoost AdaBoost model
void TestAdaBoost(const std::pair<arma::mat, arma::Row<size_t>> &data) {
  try {
    Py_Initialize(); // Initialize the Python Interpreter
    PyObject *pArgsStart = PyTuple_Pack(2, PyUnicode_FromString("output"),
                                        PyUnicode_FromString("emissions.csv"));
    PyObject *pArgsStop = PyTuple_New(0);
    SetSeedAda(42);

    // Load dataset
    const arma::mat &X = data.first;
    const arma::Row<size_t> &y = data.second;

    // Check the dimensions of the dataset
    if (X.n_cols != y.n_elem) {
      throw std::runtime_error(
          "Mismatch between feature matrix and target vector size.");
    }

    // Split the data into training and testing sets (80% training, 20% testing)
    arma::mat trainX, testX;
    arma::Row<size_t> trainY, testY;
    mlpack::data::Split(X, y, trainX, testX, trainY, testY, 0.2, true);

    // Ensure testX is in column-major format (each column is a sample)
    if (testX.n_rows != X.n_rows) {
      testX = testX.t();
    }

    std::cout << "Testing data dimensions: " << testX.n_rows << " x "
              << testX.n_cols << std::endl;

    // Load the trained model
    BoosterHandle booster;
    safe_xgboost(XGBoosterCreate(nullptr, 0, &booster));
    safe_xgboost(XGBoosterLoadModel(booster, "xgboost_adaboost_model.json"));

    // Convert test data to float*
    std::vector<float> testX_float = convert_to_float(testX);

    // Predict on the test set
    DMatrixHandle dtest;
    safe_xgboost(XGDMatrixCreateFromMat(testX_float.data(), testX.n_cols,
                                        testX.n_rows, 0, &dtest));

    bst_ulong out_len;
    const float *out_result;
    CallPython("tracker_control", "Tracker", "start_tracker", pArgsStart);

    safe_xgboost(
        XGBoosterPredict(booster, dtest, 0, 0, 0, &out_len, &out_result));
    CallPython("tracker_control", "Tracker", "stop_tracker", pArgsStop);

    // Convert predictions to binary classes
    std::vector<size_t> predictions(out_len);
    for (size_t i = 0; i < out_len; ++i) {
      predictions[i] = out_result[i] > 0.5 ? 1 : 0;
    }

    // Calculate accuracy
    size_t correct = 0;
    for (size_t i = 0; i < out_len; ++i) {
      if (predictions[i] == testY[i]) {
        ++correct;
      }
    }
    double accuracy = static_cast<double>(correct) / out_len;

    std::cout << "XGBoost AdaBoost Test Accuracy: " << accuracy * 100 << "%"
              << std::endl;

    // Clean up
    safe_xgboost(XGBoosterFree(booster));
    safe_xgboost(XGDMatrixFree(dtest));

  } catch (const std::exception &e) {
    std::cerr << "Error in TestAdaBoost: " << e.what() << std::endl;
  }
}
