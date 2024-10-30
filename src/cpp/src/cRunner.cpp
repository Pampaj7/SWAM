#include "Adaboost.cpp"
#include "DecisionTree.cpp"
#include "RandomForest.cpp"
#include "XGBOOST.cpp"
#include "knn.cpp"
#include "loader.cpp"
#include "logreg.cpp"
#include "naiveBayes.cpp"
#include "pythonLinker.cpp"
#include "svc.cpp"
#include <iostream>

enum Algs {
  LOGREG,
  DECISION_TREE,
  RANDOM_FOREST,
  XGBOOST,
  KNN,
  SVC,
  ADABOOST,
  NAIVEBAYES,
  UNKNOWN
};
// Define dataset information structure
struct DatasetInfo {
  std::string filePath;
};

// Define a mapping for datasets
const std::unordered_map<std::string, DatasetInfo> datasetMap = {
    {"breastCancer",
     {"../datasets/breastcancer/breastCancer_processed.csv"}},
    {"iris",
     {
         "../datasets/iris/iris_processed.csv",
     }},
    {"wine",
     {
         "../datasets/winequality/wineQuality_processed.csv",
     }}};

// Function to load dataset based on string identifier
std::pair<arma::mat, arma::Row<size_t>>
loadDataset(const std::string &datasetName) {
  auto it = datasetMap.find(datasetName);
  if (it == datasetMap.end()) {
    throw std::runtime_error("Unknown dataset type: " + datasetName);
  }

  const DatasetInfo &info = it->second;
  return load_data_from_csv(it->second.filePath);
  // it->second.targetMapping);
}

Algs getAlgorithmFromString(const std::string &algo) {
  if (algo == "logisticRegression")
    return LOGREG;
  if (algo == "decisionTree")
    return DECISION_TREE;
  if (algo == "randomForest")
    return RANDOM_FOREST;
  if (algo == "XGBoost")
    return XGBOOST;
  if (algo == "KNN")
    return KNN;
  if (algo == "SVC")
    return SVC;
  if (algo == "adaBoost")
    return ADABOOST;
  if (algo == "naiveBayes")
    return NAIVEBAYES;
  return UNKNOWN;
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    cerr << "Usage: " << argv[0] << " <dataset_name> <algorithm> <train-test>"
         << endl;
    return 1;
  }

  string datasetName = argv[1];
  string algorithm = argv[2];
  string train = argv[3];
  bool Test;
  if (!strcmp(argv[3], "true")) {
    Test = true;
  } else {
    Test = false;
  }

  auto dataset = loadDataset(datasetName);

  // Determine which algorithm to run
  Algs algo = getAlgorithmFromString(algorithm);

  switch (algo) {
  case LOGREG:
    if (Test) {
      TestLogisticRegression(dataset);
    } else {
      TrainLogisticRegression(dataset);
    }
    break;
  case DECISION_TREE:
    if (Test) {
      TestDecisionTree(dataset);
    } else
      TrainDecisionTree(dataset);
    break;
  case RANDOM_FOREST:
    if (Test) {
      TestRandomForest(dataset);
    } else
      TrainRandomForest(dataset);
    break;
  case XGBOOST:
    TrainXGBOOST(dataset);
    break;
  case KNN:
    if (Test) {
      TestKnn(dataset);
    } else
      TrainKnn(dataset);
    break;
  case SVC:
    if (Test) {
      TestSVM(dataset);
    } else
      TrainSVM(dataset);
    break;
  case ADABOOST:
    if (Test) {
      TestAdaBoost(dataset);
    } else {
      TrainAdaBoost(dataset);
    }
    break;
  case NAIVEBAYES:
    if (Test) {
      TestNaiveBayes(dataset);
    } else {
      TrainNaiveBayes(dataset);
    }
    break;
  default:
    cerr << "Unknown algorithm: " << algorithm << endl;
    return 1;
  }
  return 0;
}
