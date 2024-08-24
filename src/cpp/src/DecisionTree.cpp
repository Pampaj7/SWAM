#include <armadillo>
#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>

using namespace mlpack;
using namespace arma;

int TrainDecisionTree(std::pair<arma::mat, arma::Row<size_t>> data) {
  try {
    // Load dataset
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
    DecisionTree<> dt(trainX, trainY, numClasses); // Number of classes

    // Predict on the test set
    arma::Row<size_t> predictions;
    dt.Classify(testX, predictions);

    // Calculate accuracy
    double accuracy = accu(predictions == testY) / (double)testY.n_elem;
    std::cout << "DecisionTree Test Accuracy: " << accuracy * 100 << "%"
              << std::endl;

    // Compute classification report (if needed, this part would need custom
    // implementation)

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1; // Return non-zero in case of error
  }

  return 0; // Return zero to indicate success
}
