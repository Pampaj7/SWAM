#include <armadillo>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <stdexcept>

std::pair<arma::mat, arma::Row<size_t>>
load_data_from_csv(const std::string &filename,
                      const std::string &targetColumnName,
                      const std::unordered_map<std::string, size_t>& targetMapping = {}) {
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
  std::getline(file, line); // Read header line

  std::istringstream headerStream(line);
  std::string headerToken;
  std::vector<std::string> headers;
  size_t targetColumnIndex = std::string::npos;

  // Parse the header to get the column names
  while (std::getline(headerStream, headerToken, ',')) {
    headers.push_back(headerToken);
  }

  // Find the index of the target column
  for (size_t i = 0; i < headers.size(); ++i) {
    std::cout << headers[i] << std::endl;
    if (headers[i] == targetColumnName) {
      targetColumnIndex = i;
      break;
    }
  }

  if (targetColumnIndex == std::string::npos) {
    throw std::runtime_error("Target column name not found in the header: " + targetColumnName);
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
        if (column == 0) {
          // Skip the ID column
        } else if (column == targetColumnIndex) {
          if (targetMapping.empty()) {
            // If no mapping provided, assume the target is numeric
            y.push_back(static_cast<size_t>(std::stoi(token)));
          } else {
            // Use the mapping to encode the target value
            auto it = targetMapping.find(token);
            if (it != targetMapping.end()) {
              y.push_back(it->second);
            } else {
              throw std::runtime_error("Unrecognized target value: " + token);
            }
          }
        } else {
          row.push_back(std::stod(token));
        }
        column++;
      } catch (const std::exception &e) {
        throw std::runtime_error("Error parsing value: " + token +
                                 " in line: " + line);
      }
    }

    if (!row.empty() && row.size() == headers.size() - 2) { // Ensure row size matches the number of features
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
