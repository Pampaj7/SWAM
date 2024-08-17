#include <armadillo>
#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>

using namespace mlpack;
using namespace arma;

// Function to load data from CSV file
std::pair<arma::mat, arma::Row<size_t>>
load_data_from_csv(const std::string &filename) {
  std::vector<std::vector<double>> X;
  std::vector<size_t> y;

  if (!std::filesystem::exists(filename)) {
    throw std::runtime_error("File does not exist: " + filename);
  }

  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + filename);
  }

  std::string line;
  // Skip the header if it exists
  std::getline(file, line);

  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string token;
    std::vector<double> row;
    int column = 0;

    while (std::getline(iss, token, ',')) {
      if (token.empty()) {
        continue; // Skip empty fields
      }

      try {
        if (column == 0) {
          // Skip the ID column
        } else if (column == 1) {
          // Convert 'M' to 1 and 'B' to 0
          y.push_back(token == "M" ? 1 : 0);
        } else {
          row.push_back(std::stod(token));
        }
      } catch (const std::exception &e) {
        throw std::runtime_error("Error parsing value: " + token +
                                 " in line: " + line);
      }
      column++;
    }

    if (!row.empty()) {
      X.push_back(row);
    }
  }

  if (X.empty() || y.empty()) {
    throw std::runtime_error("No data loaded from file");
  }

  // Convert std::vector to Armadillo matrix
  arma::mat arma_X(X[0].size(), X.size()); // Note the swapped dimensions
  for (size_t i = 0; i < X.size(); ++i) {
    for (size_t j = 0; j < X[i].size(); ++j) {
      arma_X(j, i) = X[i][j];
    }
  }

  arma::Row<size_t> arma_y(y.size());
  for (size_t i = 0; i < y.size(); ++i) {
    arma_y(i) = y[i];
  }

  return {arma_X, arma_y};
}

int main() {
  try {
    // Load dataset
    auto [X, y] =
        load_data_from_csv("../../../datasets/breastcancer/breastcancer.csv");
    std::cout << "Data loaded successfully." << std::endl;
    std::cout << "Feature matrix dimensions: " << X.n_rows << " x " << X.n_cols
              << std::endl;
    std::cout << "Label vector size: " << y.n_elem << std::endl;

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
    std::cout << "Test Accuracy: " << accuracy * 100 << "%" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }

  return 0;
}
