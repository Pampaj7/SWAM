package com.example;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.unsupervised.attribute.NumericToNominal;
import java.util.Random;

import weka.filters.Filter;
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
        return "../datasets/breastcancer/breastCancer_processed.csv";
      case "iris":
        return "../datasets/iris/iris_processed.csv";
      case "winequality":
        return "../datasets/winequality/wineQuality_processed.csv";
      default:
        return null;
    }
  }

  // convert to nominal
  public static Instances convertClassToNominal(Instances data, String targetLabelName) throws Exception {
    int targetIndex = data.attribute(targetLabelName).index();
    data.setClassIndex(targetIndex);

    if (data.classAttribute().isNumeric()) {
      NumericToNominal convert = new NumericToNominal();
      String[] options = new String[] { "-R", String.valueOf(targetIndex + 1) };
      convert.setOptions(options);
      convert.setInputFormat(data);
      data = Filter.useFilter(data, convert);
    }

    return data;
  }

  // Function to split data into train and test sets (stratified split)
  public static Instances[] stratifiedSplit(Instances data, double trainRatio, int seed) throws Exception {
    // Ensure the data is randomized before splitting
    Random rand = new Random(seed);
    data.randomize(rand);

    // Split the dataset
    int trainSize = (int) Math.round(data.numInstances() * trainRatio);
    int testSize = data.numInstances() - trainSize;

    Instances train = new Instances(data, 0, trainSize);
    Instances test = new Instances(data, trainSize, testSize);

    return new Instances[] { train, test };
  }

}
