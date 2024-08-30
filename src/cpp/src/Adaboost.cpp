
#include <armadillo>
#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/methods/adaboost/adaboost.hpp>

using namespace mlpack;
using namespace arma;

void SetSeedAda(unsigned long seed) {
  arma::arma_rng::set_seed(seed); // Set seed for Armadillo
  mlpack::RandomSeed(seed);       // Set seed for mlpack
}
// Function to train the AdaBoost model
void TrainAdaBoost(std::pair<arma::mat, arma::Row<size_t>> data) {
  try {
    SetSeedAda(42);
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

    // Parameters for AdaBoost
    const size_t numTrees = 100; // Number of trees (base learners)
    const size_t numClasses = 2; // Number of classes (adjust as needed)

    // Create and train the AdaBoost model
    AdaBoost<> adaboost(trainX, trainY, numTrees, numClasses);

    // Save the trained model to a file
    mlpack::data::Save("./adaboost_model.xml", "adaboost_model", adaboost);

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
}

// Function to test the AdaBoost model
void TestAdaBoost(std::pair<arma::mat, arma::Row<size_t>> data) {
  try {
    SetSeedAda(42);
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

    // Load the trained AdaBoost model from a file
    AdaBoost<> adaboost;
    mlpack::data::Load("./adaboost_model.xml", "adaboost_model", adaboost);

    // Predict on the test set
    arma::Row<size_t> predictions;
    adaboost.Classify(testX, predictions);

    // Calculate accuracy
    double accuracy = accu(predictions == testY) / (double)testY.n_elem;
    std::cout << "AdaBoost Test Accuracy: " << accuracy * 100 << "%"
              << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
}
