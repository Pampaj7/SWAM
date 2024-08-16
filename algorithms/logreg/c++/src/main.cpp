#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

class LogisticRegression {
private:
  std::vector<double> weights;
  double bias;
  double learning_rate;
  int num_iterations;

public:
  LogisticRegression(int n_features, double lr = 0.01, int iterations = 1000)
      : learning_rate(lr), num_iterations(iterations) {
    weights.resize(n_features, 0.0);
    bias = 0.0;
  }

  double sigmoid(double z) { return 1.0 / (1.0 + std::exp(-z)); }

  void fit(const std::vector<std::vector<double>> &X,
           const std::vector<int> &y) {
    int n_samples = X.size();
    int n_features = X[0].size();

    for (int i = 0; i < num_iterations; ++i) {
      std::vector<double> predictions(n_samples);

      // Forward pass
      for (int j = 0; j < n_samples; ++j) {
        double z = bias;
        for (int k = 0; k < n_features; ++k) {
          z += weights[k] * X[j][k];
        }
        predictions[j] = sigmoid(z);
      }

      // Compute gradients
      std::vector<double> dw(n_features, 0.0);
      double db = 0.0;

      for (int j = 0; j < n_samples; ++j) {
        double error = predictions[j] - y[j];
        for (int k = 0; k < n_features; ++k) {
          dw[k] += error * X[j][k];
        }
        db += error;
      }

      // Update parameters
      for (int k = 0; k < n_features; ++k) {
        weights[k] -= learning_rate * dw[k] / n_samples;
      }
      bias -= learning_rate * db / n_samples;
    }
  }

  std::vector<int> predict(const std::vector<std::vector<double>> &X) {
    std::vector<int> predictions;
    for (const auto &sample : X) {
      double z = bias;
      for (int i = 0; i < weights.size(); ++i) {
        z += weights[i] * sample[i];
      }
      predictions.push_back(sigmoid(z) >= 0.5 ? 1 : 0);
    }
    return predictions;
  }
};

// Assume you have a function to load data from CSV
// std::pair<std::vector<std::vector<double>>, std::vector<int>>
// load_data_from_csv(const std::string& filename);

// ... (LogisticRegression class implementation remains the same)

std::pair<std::vector<std::vector<double>>, std::vector<int>>
load_data_from_csv(const std::string &filename) {
  std::vector<std::vector<double>> X;
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

  return {X, y};
}

int main() {
  try {
    // Load data
    auto [X, y] = load_data_from_csv(
        "../../../../../../datasets/breastcancer/breastcancer.csv");

    // Print some information about the loaded data
    std::cout << "Loaded " << X.size() << " samples with " << X[0].size()
              << " features each.\n";

    // Split data into train and test sets (simple 80-20 split)
    int split_index = X.size() * 0.8;
    std::vector<std::vector<double>> X_train(X.begin(),
                                             X.begin() + split_index);
    std::vector<std::vector<double>> X_test(X.begin() + split_index, X.end());
    std::vector<int> y_train(y.begin(), y.begin() + split_index);
    std::vector<int> y_test(y.begin() + split_index, y.end());

    // Create and train the model
    LogisticRegression model(X[0].size());
    model.fit(X_train, y_train);

    // Make predictions
    std::vector<int> predictions = model.predict(X_test);

    // Calculate accuracy
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
      if (predictions[i] == y_test[i]) {
        correct++;
      }
    }
    double accuracy = static_cast<double>(correct) / predictions.size();

    std::cout << "Accuracy: " << accuracy << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
