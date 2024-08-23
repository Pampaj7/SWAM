// Your First C++ Program

#include "DecisionTree.cpp"
#include "RandomForest.cpp"
#include "XGBOOST.cpp"
#include "knn.cpp"
#include "loader.cpp"
#include "logreg.cpp"
#include "svc.cpp"
#include <iostream>

// Define dataset information structure
struct DatasetInfo {
  std::string filePath;
  std::string targetColumn;
  std::unordered_map<std::string, size_t> targetMapping;
};

// Define a mapping for datasets
const std::unordered_map<std::string, DatasetInfo> datasetMap = {
    {"breast_cancer",
     {"../../datasets/breastcancer/breastcancer.csv",
      "diagnosis",
      {{"M", 1}, {"B", 0}}}},
    {"iris",
     {"../../datasets/iris/iris.csv",
      "Species",
      {{"setosa", 0}, {"versicolor", 1}, {"virginica", 2}}}},
    {"wine_quality", {"../../datasets/winequality/wine_data.csv", "quality"}}};

// Function to load dataset based on string identifier
std::pair<arma::mat, arma::Row<size_t>>
loadDataset(const std::string &datasetName) {
  auto it = datasetMap.find(datasetName);
  if (it == datasetMap.end()) {
    throw std::runtime_error("Unknown dataset type: " + datasetName);
  }

  const DatasetInfo &info = it->second;
  return load_data_from_csv(info.filePath, info.targetColumn,
                            info.targetMapping);
}
int main(int argc, char *argv[]) {

  string datasetName = argv[1];
  auto dataset = loadDataset(datasetName);
  TrainLogreg(dataset);
  TrainDecisionTree(dataset);
  TrainRandomForest(dataset);
  TrainXGBOOST(dataset);
  TrainKnn(dataset);
  TrainSVC(dataset);
}
