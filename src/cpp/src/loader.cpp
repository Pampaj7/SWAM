#include <armadillo>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

std::pair<arma::mat, arma::Row<size_t>>
load_data_from_csv(const std::string &filename) {
  std::vector<std::vector<double>> X;
  std::vector<size_t> y;
  size_t targetColumnIndex = std::string::npos;

  if (!std::filesystem::exists(filename)) {
    throw std::runtime_error("File does not exist: " + filename);
  }

  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + filename);
  }

  std::string line;
  std::getline(file, line); // Read the header line

  // Parse the header to get the number of columns
  std::istringstream headerStream(line);
  size_t numColumns = 0;
  std::string headerToken;
  while (std::getline(headerStream, headerToken, ',')) {
    numColumns++;
  }

  // Set the target column index as the last column
  targetColumnIndex = numColumns - 1;

  // Process the data
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string token;
    std::vector<double> row;
    size_t column = 0;

    while (std::getline(iss, token, ',')) {
      if (token.empty()) {
        continue; // Skip empty fields
      }

      try {
        double value = std::stod(token);
        if (column == targetColumnIndex) {
          y.push_back(static_cast<size_t>(value)); // Store the target value
        } else {
          row.push_back(value); // Store the feature value
        }
        column++;
      } catch (const std::exception &e) {
        throw std::runtime_error("Error parsing value: " + token +
                                 " in line: " + line);
      }
    }

    if (row.size() == targetColumnIndex) { // Ensure the row is valid
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
