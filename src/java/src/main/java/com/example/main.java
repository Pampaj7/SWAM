package com.example;

import weka.core.Instances;

public class main {

  public static Instances loadDataset(String datasetName) {
    // Assuming you have a DatasetLoader class with a loadDataset method
    String dataset_path = "";
    switch (datasetName) {
      case "breastCancer":
        dataset_path = "breast_cancer";
        break;
      case "iris":
        dataset_path = "iris";
        break;
      case "wine":
        dataset_path = "winequality";
        break;
    }
    Instances data = loader.loadDataset(dataset_path);
    // switch case to select the target label based on the dataset name

    if (data != null) {
      System.out.println("Dataset loaded successfully: " + data.relationName());
      System.out.println("Number of instances: " + data.numInstances());
    } else {
      System.out.println("Failed to load the dataset.");
    }

    return data;
  }

  public static void main(String[] args) {
    if (args.length < 3) {
      System.out.println("Usage: java Main <dataset_name> <algorithm>");
      System.out.println("Available algorithms: logisticregression, decisiontree, randomforest, xgboost, svc, knn");
      return;
    }
    String datasetName = args[0];
    String algorithm = args[1];
    String train = args[2];
    Instances data = loadDataset(datasetName);
    String targetLabel = "";
    switch (datasetName) {
      case "breastCancer":
        targetLabel = "diagnosis";
        break;
      case "iris":
        targetLabel = "species";
        break;
      case "wine":
        targetLabel = "quality";
        break;
    }

    if (data == null) {
      System.out.println("No data loaded. Exiting.");
      return;
    }

    try {
      switch (algorithm.toLowerCase()) {
        case "logisticregression":
          if (train.equals("true")) {
            logreg.train(data, targetLabel);
          } else {
            logreg.test(data, targetLabel);
          }
          break;
        case "decisiontree":
          if (train.equals("true")) {
            decisiontree.train(data, targetLabel);
          } else {
            decisiontree.test(data, targetLabel);
          }
          break;
        case "randomforest":
          if (train.equals("true")) {
            randomforest.train(data, targetLabel);
          } else {
            randomforest.test(data, targetLabel);
          }
          break;
        case "xgboost":
          xgboost.train(data, targetLabel);
          break;
        case "svc":
          if (train.equals("true")) {
            svc.train(data, targetLabel);
          } else {
            svc.test(data, targetLabel);
          }
          break;
        case "knn":
          if (train.equals("true")) {
            knn.train(data, targetLabel);
          } else {
            knn.test(data, targetLabel);
          }
          break;
        case "naivebayes":
          if (train.equals("true")) {
            naivebayes.train(data, targetLabel);
          } else {
            naivebayes.test(data, targetLabel);
          }
          break;
        case "adaboost":
          if (train.equals("true")) {
            adaboost.train(data, targetLabel);
          } else {
            adaboost.test(data, targetLabel);
          }
          break;
        default:
          System.out.println("Unknown algorithm: " + algorithm);
          System.out.println("Available algorithms: logreg, decisiontree, randomforest, xgboost, svc, knn");
          break;
      }
    } catch (Exception e) {
      e.printStackTrace();
    }
  }
}
