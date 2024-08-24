#include <armadillo>
#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>

using namespace mlpack;
using namespace arma;

// Function to load data from CSV file

int TrainRandomForest(std::pair<arma::mat, arma::Row<size_t>> data) {
  try {
    // Load dataset
    arma::mat X = data.first;
    arma::Row<size_t> y = data.second;

    // Split the data into training and testing sets
    arma::mat trainX, testX;
    arma::Row<size_t> trainY, testY;

    data::Split(X, y, trainX, testX, trainY, testY, 0.2,
                true); // 80% training, 20% testing

    // Parameters for RandomForest to match scikit-learn
    const size_t numTrees = 100; // Number of trees (equivalent to n_estimators)
    const size_t numClasses = 2; // Number of classes
    const size_t minimumLeafSize =
        1; // Minimum leaf size (equivalent to min_samples_leaf)
    const bool computeImportance = false; // Not computing feature importance
    const size_t maxDepth = 0; // Unlimited depth (equivalent to max_depth=None)

    // Set the seed for reproducibility (similar to random_state in
    // scikit-learn)
    arma::arma_rng::set_seed(42);

    // Create and train the RandomForest model
    RandomForest<> rf(trainX, trainY, numTrees);

    // Predict on the test set
    arma::Row<size_t> predictions;
    rf.Classify(testX, predictions);

    // Calculate accuracy
    double accuracy = accu(predictions == testY) / (double)testY.n_elem;
    std::cout << "Random Forest Test Accuracy: " << accuracy * 100 << "%"
              << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }

  return 0;
}
