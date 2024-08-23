#include <algorithm> // For std::shuffle
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random> // For std::default_random_engine
#include <sstream>
#include <stdexcept>
#include <vector>
#include <xgboost/c_api.h>

using namespace std;

// Function to load data from CSV file
std::pair<std::vector<std::vector<float>>, std::vector<int>>
load_data_from_csv(const std::string &filename) {
  std::vector<std::vector<float>> X;
  std::vector<int> y;

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
    std::vector<float> row;
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
          row.push_back(std::stof(token));
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

  return {X, y};
}

void split_data(const std::vector<std::vector<float>> &X,
                const std::vector<int> &y,
                std::vector<std::vector<float>> &X_train,
                std::vector<int> &y_train,
                std::vector<std::vector<float>> &X_test,
                std::vector<int> &y_test, float test_size = 0.2,
                unsigned seed = 42) { // Add seed as a parameter
  size_t numRows = X.size();
  std::vector<size_t> indices(numRows);
  std::iota(indices.begin(), indices.end(), 0);

  // Shuffle indices
  auto rng = std::default_random_engine{seed}; // Use the seed here
  std::shuffle(indices.begin(), indices.end(), rng);

  size_t test_count = static_cast<size_t>(test_size * numRows);

  for (size_t i = 0; i < numRows; ++i) {
    if (i < test_count) {
      X_test.push_back(X[indices[i]]);
      y_test.push_back(y[indices[i]]);
    } else {
      X_train.push_back(X[indices[i]]);
      y_train.push_back(y[indices[i]]);
    }
  }
}

void train_and_evaluate_xgboost(const std::vector<std::vector<float>> &X,
                                const std::vector<int> &y,
                                unsigned seed = 42) { // Add seed as a parameter
  // Split the data
  std::vector<std::vector<float>> X_train, X_test;
  std::vector<int> y_train, y_test;
  split_data(X, y, X_train, y_train, X_test, y_test, 0.2,
             seed); // Pass the seed

  // Convert training data to DMatrix format
  size_t trainRows = X_train.size();
  size_t trainCols = X_train[0].size();

  std::vector<float> flattened_X_train(trainRows * trainCols);
  for (size_t i = 0; i < trainRows; ++i) {
    for (size_t j = 0; j < trainCols; ++j) {
      flattened_X_train[i * trainCols + j] = X_train[i][j];
    }
  }

  DMatrixHandle dtrain;
  XGDMatrixCreateFromMat(flattened_X_train.data(), trainRows, trainCols,
                         -999.0f, &dtrain);

  // Set labels
  std::vector<float> labels(y_train.begin(), y_train.end());
  XGDMatrixSetFloatInfo(dtrain, "label", labels.data(), trainRows);

  // Parameters for XGBoost
  BoosterHandle booster;
  const char *param_keys[] = {"booster", "objective", "eval_metric",
                              "max_depth", "eta"};
  const char *param_values[] = {"gbtree", "binary:logistic", "error", "6",
                                "0.3"};
  int num_params = sizeof(param_keys) / sizeof(param_keys[0]);

  XGBoosterCreate(&dtrain, 1, &booster);
  for (int i = 0; i < num_params; ++i) {
    XGBoosterSetParam(booster, param_keys[i], param_values[i]);
  }

  // Train the model
  int n_estimators = 100; // Number of boosting rounds
  for (int i = 0; i < n_estimators; ++i) {
    XGBoosterUpdateOneIter(booster, i, dtrain);
  }

  // Convert test data to DMatrix format
  size_t testRows = X_test.size();
  std::vector<float> flattened_X_test(testRows * trainCols);
  for (size_t i = 0; i < testRows; ++i) {
    for (size_t j = 0; j < trainCols; ++j) {
      flattened_X_test[i * trainCols + j] = X_test[i][j];
    }
  }

  DMatrixHandle dtest;
  XGDMatrixCreateFromMat(flattened_X_test.data(), testRows, trainCols, -999.0f,
                         &dtest);

  // Make predictions
  bst_ulong out_len;
  const float *preds;
  int ntree_limit = 0; // 0 means use all trees
  XGBoosterPredict(booster, dtest, 0, ntree_limit, 0, &out_len, &preds);

  // Evaluate accuracy
  int correct = 0;
  for (size_t i = 0; i < testRows; ++i) {
    int predicted_label = preds[i] > 0.5f ? 1 : 0;
    if (predicted_label == y_test[i]) {
      ++correct;
    }
  }
  double accuracy = static_cast<double>(correct) / testRows;
  std::cout << "Test Accuracy: " << accuracy * 100 << "%" << std::endl;

  // Cleanup
  XGDMatrixFree(dtrain);
  XGDMatrixFree(dtest);
  XGBoosterFree(booster);
}

int main() {
  try {
    // Load dataset
    auto [X, y] =
        load_data_from_csv("../../../datasets/breastcancer/breastcancer.csv");
    std::cout << "Data loaded successfully." << std::endl;
    std::cout << "Feature matrix dimensions: " << X.size() << " x "
              << X[0].size() << std::endl;
    std::cout << "Label vector size: " << y.size() << std::endl;

    // Train and evaluate the XGBoost model
    train_and_evaluate_xgboost(X, y, 42); // Pass the seed to the function

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }

  return 0;
}
