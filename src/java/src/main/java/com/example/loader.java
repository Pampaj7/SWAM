package com.example;

import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.io.File;
import java.io.IOException;

public class loader {

  public static Instances loadDataset(String datasetName) {
    String filePath = getFilePath(datasetName);
    if (filePath == null) {
      System.out.println("Dataset name not recognized.");
      return null;
    }

    CSVLoader loader = new CSVLoader();
    try {
      loader.setSource(new File(filePath));
      Instances data = loader.getDataSet();
      return data;
    } catch (IOException e) {
      System.out.println("Error loading dataset: " + e.getMessage());
      return null;
    }
  }

  private static String getFilePath(String datasetName) {
    switch (datasetName.toLowerCase()) {
      case "breast_cancer":
        return "../../datasets/breastcancer/dataset_processed/breastcancer_processed.csv";
      case "iris":
        return "../../datasets/iris/dataset_processed/iris_processed.csv";
      case "winequality":
        return "../../datasets/winequality/dataset_processed/wine_Data_processed.csv";
      default:
        return null;
    }
  }
}
