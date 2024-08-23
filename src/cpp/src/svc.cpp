#include <armadillo>
#include <cmath>
#include <iostream>
#include <unordered_map>

class SVC {
public:
  SVC(double C = 1.0, const std::string &kernel = "linear", double gamma = 0.1)
      : C(C), kernel_type(kernel), gamma(gamma) {}

  void fit(const arma::mat &X, const arma::Row<size_t> &y) {
    size_t n_samples = X.n_cols;
    size_t n_features = X.n_rows;

    // Compute the kernel matrix
    arma::mat K = compute_kernel_matrix(X);

    // Initialize the alpha values (Lagrange multipliers)
    arma::vec alpha = arma::zeros(n_samples);

    // Training loop
    size_t max_iter = 1000;
    for (size_t iter = 0; iter < max_iter; ++iter) {
      for (size_t i = 0; i < n_samples; ++i) {
        double E_i = decision_function(K.col(i)) - static_cast<double>(y(i));
        if ((y(i) * E_i < -1e-3 && alpha(i) < C) ||
            (y(i) * E_i > 1e-3 && alpha(i) > 0)) {
          size_t j = (i + 1) % n_samples;
          double E_j = decision_function(K.col(j)) - static_cast<double>(y(j));

          double alpha_i_old = alpha(i);
          double alpha_j_old = alpha(j);

          // Compute L and H
          double L, H;
          if (y(i) != y(j)) {
            L = std::max(0.0, alpha(j) - alpha(i));
            H = std::min(C, C + alpha(j) - alpha(i));
          } else {
            L = std::max(0.0, alpha(i) + alpha(j) - C);
            H = std::min(C, alpha(i) + alpha(j));
          }

          if (L == H)
            continue;

          double eta = 2 * K(i, j) - K(i, i) - K(j, j);
          if (eta >= 0)
            continue;

          alpha(j) -= y(j) * (E_i - E_j) / eta;

          alpha(j) = std::clamp(alpha(j), L, H);

          if (std::abs(alpha(j) - alpha_j_old) < 1e-5)
            continue;

          alpha(i) += y(i) * y(j) * (alpha_j_old - alpha(j));

          // Update the bias term
          double b1 = b - E_i - y(i) * (alpha(i) - alpha_i_old) * K(i, i) -
                      y(j) * (alpha(j) - alpha_j_old) * K(i, j);
          double b2 = b - E_j - y(i) * (alpha(i) - alpha_i_old) * K(i, j) -
                      y(j) * (alpha(j) - alpha_j_old) * K(j, j);

          if (0 < alpha(i) && alpha(i) < C)
            b = b1;
          else if (0 < alpha(j) && alpha(j) < C)
            b = b2;
          else
            b = (b1 + b2) / 2;
        }
      }
    }

    // Store the support vectors
    for (size_t i = 0; i < n_samples; ++i) {
      if (alpha(i) > 1e-5) {
        support_vectors.push_back(X.col(i));
        support_vector_labels.push_back(y(i));
        support_vector_alphas.push_back(alpha(i));
      }
    }
  }

  arma::Row<size_t> predict(const arma::mat &X) const {
    size_t n_samples = X.n_cols;
    arma::Row<size_t> predictions(n_samples);

    for (size_t i = 0; i < n_samples; ++i) {
      double decision_value = decision_function(X.col(i));
      predictions(i) = decision_value >= 0 ? 1 : 0;
    }

    return predictions;
  }

private:
  double C;
  std::string kernel_type;
  double gamma;
  double b = 0.0;
  std::vector<arma::vec> support_vectors;
  std::vector<size_t> support_vector_labels;
  std::vector<double> support_vector_alphas;

  arma::mat compute_kernel_matrix(const arma::mat &X) const {
    size_t n_samples = X.n_cols;
    arma::mat K(n_samples, n_samples);

    for (size_t i = 0; i < n_samples; ++i) {
      for (size_t j = 0; j < n_samples; ++j) {
        K(i, j) = kernel(X.col(i), X.col(j));
      }
    }

    return K;
  }

  double kernel(const arma::vec &x1, const arma::vec &x2) const {
    if (kernel_type == "linear") {
      return arma::dot(x1, x2);
    } else if (kernel_type == "rbf") {
      return std::exp(-gamma * arma::dot(x1 - x2, x1 - x2));
    }
    throw std::runtime_error("Unsupported kernel type");
  }

  double decision_function(const arma::vec &x) const {
    double result = 0.0;
    for (size_t i = 0; i < support_vectors.size(); ++i) {
      result += support_vector_alphas[i] * support_vector_labels[i] *
                kernel(support_vectors[i], x);
    }
    return result + b;
  }
};
;

int TrainSVC(std::pair<arma::mat, arma::Row<size_t>> data) {
  arma::mat X = data.first;
  arma::Row<size_t> y = data.second;

  size_t train_size = static_cast<size_t>(0.8 * X.n_cols); // 80% for training
  arma::mat X_train = X.cols(0, train_size - 1);
  arma::Row<size_t> y_train = y.subvec(0, train_size - 1);

  arma::mat X_test = X.cols(train_size, X.n_cols - 1);
  arma::Row<size_t> y_test = y.subvec(train_size, y.n_elem - 1);

  SVC svc(1.0, "rbf", 5);
  svc.fit(X_train, y_train);

  arma::Row<size_t> predictions = svc.predict(X_test);

  // Calculate accuracy
  size_t correct_predictions = 0;
  for (size_t i = 0; i < y_test.n_elem; ++i) {
    if (predictions(i) == y_test(i)) {
      ++correct_predictions;
    }
  }

  double accuracy =
      (static_cast<double>(correct_predictions) / y_test.n_elem) * 100.0;
  std::cout << "SVC Accuracy: " << accuracy << "%" << std::endl;

  return 0;
}
