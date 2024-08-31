#include "pythonLinker.h"
#include <Python.h>
#include <armadillo>
#include <iostream>
#include <mlpack/core/data/split_data.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace cv::ml;
using namespace arma;

void SetSeedSVM(unsigned long seed) {
  arma::arma_rng::set_seed(seed); // Set seed for Armadillo
  mlpack::RandomSeed(seed);       // Set seed for mlpack
  cv::RNG rng(seed);              // Set seed for OpenCV
}
// Function to train the SVM model
void TrainSVM(std::pair<arma::mat, arma::Row<size_t>> data) {
  try {
    Py_Initialize(); // Initialize the Python Interpreter
    PyObject *pArgsStart = PyTuple_Pack(2, PyUnicode_FromString("output"),
                                        PyUnicode_FromString("emissions.csv"));
    PyObject *pArgsStop = PyTuple_New(0);
    SetSeedSVM(42);
    // Load dataset
    arma::mat X = data.first;
    arma::Row<size_t> y = data.second;

    // Check the dimensions of the dataset
    if (X.n_cols != y.n_elem) {
      throw std::runtime_error(
          "Mismatch between feature matrix and target vector size.");
    }

    // Split the data into training and testing sets
    arma::mat trainX, testX;
    arma::Row<size_t> trainY, testY;
    mlpack::data::Split(X, y, trainX, testX, trainY, testY, 0.2,
                        true); // 80% training, 20% testing

    // Convert Armadillo matrices to OpenCV matrices
    cv::Mat trainX_cv(trainX.n_cols, trainX.n_rows, CV_32F, trainX.memptr());
    cv::Mat trainY_cv(trainY.n_elem, 1, CV_32S, trainY.memptr());

    // Create and configure the SVM model
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::RBF); // RBF kernel
    svm->setC(1);
    svm->setGamma(0.5);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-6));

    // Train the SVM model
    CallPython("tracker_control", "Tracker", "start_tracker", pArgsStart);
    svm->train(trainX_cv, ROW_SAMPLE, trainY_cv);
    CallPython("tracker_control", "Tracker", "stop_tracker", pArgsStop);

    // Save the trained model to a file
    svm->save("./svm_model.xml");

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
}

// Function to test the SVM model
void TestSVM(std::pair<arma::mat, arma::Row<size_t>> data) {
  try {
    Py_Initialize(); // Initialize the Python Interpreter
    PyObject *pArgsStart = PyTuple_Pack(2, PyUnicode_FromString("output"),
                                        PyUnicode_FromString("emissions.csv"));
    PyObject *pArgsStop = PyTuple_New(0);
    SetSeedSVM(42);
    // Load dataset
    arma::mat X = data.first;
    arma::Row<size_t> y = data.second;

    // Check the dimensions of the dataset
    if (X.n_cols != y.n_elem) {
      throw std::runtime_error(
          "Mismatch between feature matrix and target vector size.");
    }

    // Split the data into training and testing sets
    arma::mat trainX, testX;
    arma::Row<size_t> trainY, testY;
    mlpack::data::Split(X, y, trainX, testX, trainY, testY, 0.2,
                        true); // 80% training, 20% testing

    // Convert Armadillo matrices to OpenCV matrices
    cv::Mat testX_cv(testX.n_cols, testX.n_rows, CV_32F, testX.memptr());
    cv::Mat testY_cv(testY.n_elem, 1, CV_32S, testY.memptr());

    // Load the trained SVM model from a file
    Ptr<SVM> svm = Algorithm::load<SVM>("./svm_model.xml");

    // Predict on the test set
    cv::Mat predictions;
    CallPython("tracker_control", "Tracker", "start_tracker", pArgsStart);
    svm->predict(testX_cv, predictions);
    CallPython("tracker_control", "Tracker", "stop_tracker", pArgsStop);

    // Ensure predictions are in the correct shape
    predictions = predictions.reshape(1, testY_cv.rows);

    // Convert predictions to the same type as testY_cv
    predictions.convertTo(predictions, CV_32S);

    // Calculate accuracy
    cv::Mat diff;
    cv::compare(predictions, testY_cv, diff, cv::CmpTypes::CMP_EQ);
    double accuracy = 100.0 * cv::countNonZero(diff) / testY_cv.rows;
    std::cout << "SVM Test Accuracy: " << accuracy << "%" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
}
