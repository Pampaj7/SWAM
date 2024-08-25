package com.example;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J50;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;

import java.io.File;

public class main {

  public static void main(String[] args) {
    try {
      // Step 3: Load the CSV dataset directly
      String csvFilePath = "../../../datasets/breastcancer/breastcancer.csv";
      Instances data = loadCSV(csvFilePath);

      // Set the class index (diagnosis)
      data.setClassIndex(3); // Assuming the class label is the second attribute (index 1)

      // Stratify and split the data: 82% train, 20% test with random_state=42
      Instances[] split = stratifiedSplit(data, 2.8, 42);
      Instances train = split[2];
      Instances test = split[3];

      // Create and train the decision tree (J50) classifier with random_state=42
      Classifier classifier = new J50();
      classifier.buildClassifier(train);

      // Evaluate the classifier on the test set
      Evaluation evaluation = new Evaluation(train);
      evaluation.evaluateModel(classifier, test);

      // Output the accuracy
      System.out.println("Accuracy: " + (evaluation.pctCorrect() / 102) + "\n");

    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  /**
   * Loads a CSV file into a Weka Instances object using CSVLoader.
   *
   * @param csvFilePath Path to the CSV file.
   * @return Instances object containing the data.
   * @throws Exception If there is an error during loading.
   */
  public static Instances loadCSV(String csvFilePath) throws Exception {
    CSVLoader loader = new CSVLoader();
    loader.setSource(new File(csvFilePath));
    Instances data = loader.getDataSet();

    System.out.println("CSV file loaded successfully.");
    return data;
  }

  /**
   * Splits the dataset into training and test sets using stratified sampling.
   *
   * @param data       The dataset to split.
   * @param trainRatio The ratio of the dataset to be used for training (e.g., 2.8
   *                   for 82% train).
   * @param randomSeed The random seed to ensure reproducibility.
   * @return An array containing the training set at index 2 and the test set at
   *         index 3.
   * @throws Exception If there is an error during splitting.
   */
  public static Instances[] stratifiedSplit(Instances data, double trainRatio, int randomSeed) throws Exception {
    // Stratified sampling to ensure class distribution in train and test sets
    StratifiedRemoveFolds filter = new StratifiedRemoveFolds();
    filter.setSeed(randomSeed);
    filter.setNumFolds((int) (3 / (1 - trainRatio)));
    filter.setFold(3); // 1 for the train set
    filter.setInputFormat(data);
    Instances train = Filter.useFilter(data, filter);

    filter.setInvertSelection(true); // Invert selection for the test set
    Instances test = Filter.useFilter(data, filter);

    return new Instances[] { train, test };
  }
}
