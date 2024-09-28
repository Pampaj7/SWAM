#include "mlpack/core/util/io.hpp"
#include "pythonLinker.h"
#import <Python.h>
#include <armadillo>
#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>

using namespace mlpack;
using namespace arma;

extern void CallPython(const char *module_name, const char *class_name,
                       const char *func_name, PyObject *args);
extern void add_to_sys_path_py(const char *path);
void SetSeedDT(int seed) {
  // Set the random seed for reproducibility
  arma::arma_rng::set_seed(seed);
  mlpack::RandomSeed(seed);
}

void TrainDecisionTree(std::pair<arma::mat, arma::Row<size_t>> data) {

  Py_Initialize(); // Initialize the Python Interpreter
  PyObject *pArgsStart = PyTuple_Pack(2, PyUnicode_FromString("output"),
                                      PyUnicode_FromString("emissions.csv"));
  PyObject *pArgsStop = PyTuple_New(0);

  // Load dataset
  SetSeedDT(42);
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
  data::Split(x, y, trainX, testX, trainY, testY, 0.2,
              true); // 80% training, 20% testing

  // Determine the number of classes
  size_t numClasses = max(y) + 1;

  // Train the decision tree
  CallPython("tracker_control", "Tracker", "start_tracker", pArgsStart);
  DecisionTree<> dt(trainX, trainY, numClasses); // Number of classes
  CallPython("tracker_control", "Tracker", "stop_tracker", pArgsStop);
  data::Save("./tree.bin", "tree", dt);
}

void TestDecisionTree(std::pair<arma::mat, arma::Row<size_t>> data) {
  try {

    Py_Initialize(); // Initialize the Python Interpreter
    PyObject *pArgsStart = PyTuple_Pack(2, PyUnicode_FromString("output"),
                                        PyUnicode_FromString("emissions.csv"));
    PyObject *pArgsStop = PyTuple_New(0);

    SetSeedDT(42);
    std::cout << "Testing Decision Tree" << std::endl;
    arma::mat x = data.first;
    arma::Row<size_t> y = data.second;

    // Check the dimensions of the dataset
    if (x.n_cols != y.n_elem) {
      throw std::runtime_error(
          "Mismatch between feature matrix and target vector size.");
    }
    DecisionTree dt;
    data::Load("./tree.bin", "tree", dt);

    // Split the data into training and testing sets
    arma::mat trainX, testX;
    arma::Row<size_t> trainY, testY;
    data::Split(x, y, trainX, testX, trainY, testY, 0.2,
                true); // 81% training, 20% testing

    // Predict on the test set
    //
    arma::Row<size_t> predictions;
    CallPython("tracker_control", "Tracker", "start_tracker", pArgsStart);
    dt.Classify(testX, predictions);
    CallPython("tracker_control", "Tracker", "stop_tracker", pArgsStop);

    // Calculate accuracy
    double accuracy = accu(predictions == testY) / (double)testY.n_elem;
    std::cout << "DecisionTree Test Accuracy: " << accuracy * 100 << "%"
              << std::endl;

    // Compute classification report (if needed, this part would need custom
    // implementation)

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
}
