#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <armadillo>

// Logistic Regression Class
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

  void fit(const arma::mat &X, const arma::Row<size_t> &y) {
    int n_samples = X.n_cols;
    int n_features = X.n_rows;

    // Convert Armadillo types to standard vectors for computation
    std::vector<std::vector<double>> X_vec(n_samples, std::vector<double>(n_features));
    std::vector<int> y_vec(n_samples);

    for (int i = 0; i < n_samples; ++i) {
      y_vec[i] = y[i];
      for (int j = 0; j < n_features; ++j) {
        X_vec[i][j] = X(j, i);
      }
    }

    for (int i = 0; i < num_iterations; ++i) {
      std::vector<double> predictions(n_samples);

      // Forward pass
      for (int j = 0; j < n_samples; ++j) {
        double z = bias;
        for (int k = 0; k < n_features; ++k) {
          z += weights[k] * X_vec[j][k];
        }
        predictions[j] = sigmoid(z);
      }

      // Compute gradients
      std::vector<double> dw(n_features, 0.0);
      double db = 0.0;

      for (int j = 0; j < n_samples; ++j) {
        double error = predictions[j] - y_vec[j];
        for (int k = 0; k < n_features; ++k) {
          dw[k] += error * X_vec[j][k];
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

  arma::Row<size_t> predict(const arma::mat &X) {
    int n_samples = X.n_cols;
    arma::Row<size_t> predictions(n_samples);

    // Convert Armadillo types to standard vector for prediction computation
    std::vector<std::vector<double>> X_vec(n_samples, std::vector<double>(X.n_rows));
    for (int i = 0; i < n_samples; ++i) {
      for (int j = 0; j < X.n_rows; ++j) {
        X_vec[i][j] = X(j, i);
      }
    }

    for (int j = 0; j < n_samples; ++j) {
      double z = bias;
      for (int i = 0; i < weights.size(); ++i) {
        z += weights[i] * X_vec[j][i];
      }
      predictions[j] = sigmoid(z) >= 0.5 ? 1 : 0;
    }
    return predictions;
  }
};

// Train Logistic Regression Model
int TrainLogreg(std::pair<arma::mat, arma::Row<size_t>> data) {
  try {
    // Load data
    arma::mat X = data.first;
    arma::Row<size_t> y = data.second;

    // Print some information about the loaded data
    std::cout << "Loaded " << X.n_cols << " samples with " << X.n_rows
              << " features each.\n";

    // Split data into train and test sets (simple 80-20 split)
    size_t split_index = static_cast<size_t>(X.n_cols * 0.8);
    arma::mat X_train = X.cols(0, split_index - 1);
    arma::mat X_test = X.cols(split_index, X.n_cols - 1);
    arma::Row<size_t> y_train = y.subvec(0, split_index - 1);
    arma::Row<size_t> y_test = y.subvec(split_index, y.n_elem - 1);

    // Create and train the model
    LogisticRegression model(X.n_rows);
    model.fit(X_train, y_train);

    // Make predictions
    arma::Row<size_t> predictions = model.predict(X_test);

    // Calculate accuracy
    size_t correct = 0;
    for (size_t i = 0; i < predictions.n_elem; ++i) {
      if (predictions[i] == y_test[i]) {
        correct++;
      }
    }
    double accuracy = static_cast<double>(correct) / predictions.n_elem;

    std::cout << "Logreg Accuracy: " << accuracy << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}

