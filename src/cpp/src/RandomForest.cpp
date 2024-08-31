#include "pythonLinker.h"
#include <Python.h>
#include <armadillo>
#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>

using namespace mlpack;
using namespace arma;

void SetSeedRF(int seed) {
  // Set the random seed for reproducibility
  arma::arma_rng::set_seed(seed);
  mlpack::RandomSeed(seed);
}

// Function to train the Random Forest model
void TrainRandomForest(std::pair<arma::mat, arma::Row<size_t>> data) {
  try {

    Py_Initialize(); // Initialize the Python Interpreter
    PyObject *pArgsStart = PyTuple_Pack(2, PyUnicode_FromString("output"),
                                        PyUnicode_FromString("emissions.csv"));
    PyObject *pArgsStop = PyTuple_New(0);

    SetSeedRF(42);
    // Load dataset
    arma::mat X = data.first;
    arma::Row<size_t> y = data.second;

    // Split the data into training and testing sets
    arma::mat trainX, testX;
    arma::Row<size_t> trainY, testY;
    data::Split(X, y, trainX, testX, trainY, testY, 0.2,
                true); // 80% training, 20% testing

    // Parameters for RandomForest
    const size_t numTrees = 100;          // Number of trees
    const size_t numClasses = 2;          // Number of classes
    const size_t minimumLeafSize = 1;     // Minimum leaf size
    const bool computeImportance = false; // Not computing feature importance
    const size_t maxDepth = 0;            // Unlimited depth

    // Set the seed for reproducibility
    arma::arma_rng::set_seed(42);

    CallPython("tracker_control", "Tracker", "start_tracker", pArgsStart);
    // Create and train the RandomForest model
    RandomForest<> rf(trainX, trainY, numTrees, numClasses, minimumLeafSize,
                      computeImportance, maxDepth);
    CallPython("tracker_control", "Tracker", "stop_tracker", pArgsStop);

    // Save the model to a file
    data::Save("./random_forest_model.bin", "rf_model", rf);
    std::cout << "Random Forest model saved to 'random_forest_model.bin'"
              << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
}

// Function to test the Random Forest model
void TestRandomForest(std::pair<arma::mat, arma::Row<size_t>> data) {
  try {
    Py_Initialize(); // Initialize the Python Interpreter
    PyObject *pArgsStart = PyTuple_Pack(2, PyUnicode_FromString("output"),
                                        PyUnicode_FromString("emissions.csv"));
    PyObject *pArgsStop = PyTuple_New(0);
    SetSeedRF(42);
    // Load dataset
    arma::mat X = data.first;
    arma::Row<size_t> y = data.second;

    // Split the data into training and testing sets
    arma::mat trainX, testX;
    arma::Row<size_t> trainY, testY;
    data::Split(X, y, trainX, testX, trainY, testY, 0.2,
                true); // 80% training, 20% testing

    // Load the trained RandomForest model
    RandomForest<> rf;
    data::Load("./random_forest_model.bin", "rf_model", rf);

    // Predict on the test set
    arma::Row<size_t> predictions;
    CallPython("tracker_control", "Tracker", "start_tracker", pArgsStart);
    rf.Classify(testX, predictions);
    CallPython("tracker_control", "Tracker", "stop_tracker", pArgsStop);

    // Calculate accuracy
    double accuracy = accu(predictions == testY) / (double)testY.n_elem;
    std::cout << "Random Forest Test Accuracy: " << accuracy * 100 << "%"
              << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
}
