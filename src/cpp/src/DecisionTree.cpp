// #include <armadillo>
#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>

using namespace mlpack;
using namespace arma;

int TrainDecisionTree(std::pair<arma::mat, arma::Row<size_t>> data) {
  try {
    // load dataset
    arma::mat x = data.first;
    arma::Row<size_t> y = data.second;
    std::cout << "data loaded successfully." << std::endl;
    std::cout << "feature matrix dimensions: " << x.n_rows << " x " << x.n_cols
              << std::endl;
    std::cout << "label vector size: " << y.n_elem << std::endl;

    // split the data into training and testing sets
    arma::mat trainX, testX;
    arma::Row<size_t> trainY, testY;

    data::Split(x, y, trainX, testX, trainY, testY, 0.2,
                true); // 80% training, 20% testing

    // Train the decision tree
    DecisionTree<> dt(trainX, trainY, 2); // 2 classes: benign and malignant

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
  }

  return 0;
}
