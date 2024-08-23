// #include "loader.cpp"
#include <algorithm>
#include <armadillo>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <vector>

// Your load_data_from_csv function goes here

class KNNClassifier {
private:
  arma::mat X_train;
  arma::Row<size_t> y_train;
  size_t k;

  struct DistanceIndex {
    double distance;
    size_t index;

    bool operator<(const DistanceIndex &other) const {
      return distance < other.distance;
    }
  };

public:
  KNNClassifier(size_t k = 3) : k(k) {}

  void fit(const arma::mat &X, const arma::Row<size_t> &y) {
    X_train = X;
    y_train = y;
  }

  arma::Row<size_t> predict(const arma::mat &X_test) {
    arma::Row<size_t> predictions(X_test.n_cols);

    for (size_t i = 0; i < X_test.n_cols; ++i) {
      predictions(i) = predict_single(X_test.col(i));
    }

    return predictions;
  }

private:
  size_t predict_single(const arma::vec &x) {
    std::vector<DistanceIndex> distances;

    for (size_t i = 0; i < X_train.n_cols; ++i) {
      double dist = arma::norm(X_train.col(i) - x);
      distances.push_back({dist, i});
    }

    std::partial_sort(distances.begin(), distances.begin() + k,
                      distances.end());

    std::unordered_map<size_t, size_t> class_counts;
    for (size_t i = 0; i < k; ++i) {
      size_t label = y_train(distances[i].index);
      class_counts[label]++;
    }

    size_t max_count = 0;
    size_t predicted_class = 0;
    for (const auto &pair : class_counts) {
      if (pair.second > max_count) {
        max_count = pair.second;
        predicted_class = pair.first;
      }
    }

    return predicted_class;
  }
};

int TrainKnn(std::pair<arma::mat, arma::Row<size_t>> data) {

  try {
    arma::mat X = data.first;
    arma::Row<size_t> y = data.second;
    // Split the data into training and testing sets (80-20 split)
    size_t trainSize = static_cast<size_t>(X.n_cols * 0.8);
    arma::mat X_train = X.cols(0, trainSize - 1);
    arma::mat X_test = X.cols(trainSize, X.n_cols - 1);
    arma::Row<size_t> y_train = y.subvec(0, trainSize - 1);
    arma::Row<size_t> y_test = y.subvec(trainSize, y.n_elem - 1);

    // Create and train the KNN classifier
    KNNClassifier knn_classifier(5); // Using k=5
    knn_classifier.fit(X_train, y_train);

    // Make predictions
    arma::Row<size_t> predictions = knn_classifier.predict(X_test);

    // Calculate accuracy
    size_t correct = arma::accu(predictions == y_test);
    double accuracy = static_cast<double>(correct) / y_test.n_elem;

    std::cout << "KNN Accuracy: " << accuracy * 100 << "%" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
