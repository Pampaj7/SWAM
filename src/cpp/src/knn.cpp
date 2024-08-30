#include <armadillo>
#include <iostream>
#include <mlpack/core/data/split_data.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace cv::ml;
using namespace arma;

void SetSeedKNN(unsigned long seed) {
  arma::arma_rng::set_seed(seed); // Set seed for Armadillo
  mlpack::RandomSeed(seed);       // Set seed for mlpack
  cv::RNG rng(seed);              // Set seed for OpenCV
}
// Function to train the KNN model
void TrainKnn(std::pair<arma::mat, arma::Row<size_t>> data) {
  try {
    SetSeedKNN(42);
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

    // Create and configure the KNN model
    Ptr<KNearest> knn = KNearest::create();
    knn->setDefaultK(3); // Set the number of neighbors to 3
    knn->setIsClassifier(true);

    // Train the KNN model
    knn->train(trainX_cv, ROW_SAMPLE, trainY_cv);

    // Save the trained model to a file
    knn->save("./knn_model.xml");

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
}

// Function to test the KNN model
void TestKnn(std::pair<arma::mat, arma::Row<size_t>> data) {
  try {
    SetSeedKNN(42);
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

    // Load the trained KNN model from a file
    Ptr<KNearest> knn = Algorithm::load<KNearest>("./knn_model.xml");

    // Predict on the test set
    cv::Mat predictions;
    knn->findNearest(testX_cv, knn->getDefaultK(), predictions);

    // Ensure predictions are in the correct shape
    predictions = predictions.reshape(1, testY_cv.rows);

    // Convert predictions to the same type as testY_cv
    predictions.convertTo(predictions, CV_32S);

    // Calculate accuracy
    cv::Mat diff;
    cv::compare(predictions, testY_cv, diff, cv::CmpTypes::CMP_EQ);
    double accuracy = 100.0 * cv::countNonZero(diff) / testY_cv.rows;
    std::cout << "KNN Test Accuracy: " << accuracy << "%" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
}
