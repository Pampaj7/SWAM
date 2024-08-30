#include <armadillo>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

std::pair<arma::mat, arma::Row<size_t>>
load_data_from_csv(const std::string &filename,
                   const std::string &targetColumnName) {
  std::vector<std::vector<double>> X;
  std::vector<size_t> y;
  std::vector<std::string> headers;
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

  // Parse the header to get the column names
  std::istringstream headerStream(line);
  std::string headerToken;
  while (std::getline(headerStream, headerToken, ',')) {
    headers.push_back(headerToken);
  }

  // Find the index of the target column by name
  for (size_t i = 0; i < headers.size(); ++i) {
    if (headers[i] == targetColumnName) {
      targetColumnIndex = i;
      break;
    }
  }

  if (targetColumnIndex == std::string::npos) {
    throw std::runtime_error("Target column name not found in the header: " +
                             targetColumnName);
  }

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
