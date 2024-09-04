package com.example;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.classifiers.Evaluation;
import weka.core.SerializationHelper;

import java.util.Random;

public class decisiontree {

  public static long startTime;
  public static long endTime;
  public static double elapsedTime;
  private static PythonHandler pythonHandler = new PythonHandler();

  // Classifier instance that will be trained and tested
  private static final String MODEL_FILE = "decisionTree.model";
  private static Classifier classifier;

  public static void train(Instances data, String targetLabel) {
    try {
      // Set the class index based on the target label
      Attribute classAttribute = data.attribute(targetLabel);
      if (classAttribute == null) {
        System.out.println("Error: Target label '" + targetLabel + "' not found in the dataset.");
        return;
      }
      data.setClass(classAttribute);

      // Convert class attribute to nominal if it's numeric
      if (classAttribute.isNumeric()) {
        NumericToNominal filter = new NumericToNominal();
        filter.setAttributeIndices("" + (data.classIndex() + 1)); // Class index is 1-based in the filter
        filter.setInputFormat(data);
        data = Filter.useFilter(data, filter);
        System.out.println("Converted numeric class attribute to nominal.");
      }

      // Apply standardization to features
      Standardize standardizeFilter = new Standardize();
      standardizeFilter.setInputFormat(data);
      Instances standardizedData = Filter.useFilter(data, standardizeFilter);

      // Stratify and split the data: 80% train, 20% test with random seed
      Instances[] split = stratifiedSplit(standardizedData, 0.8, 42);
      Instances train = split[0];

      // Create and train the decision tree (J48) classifier
      classifier = new J48(); // J48 is the Weka implementation of C4.5
      pythonHandler.startTracker("emissions.csv");
      startTime = System.currentTimeMillis();
      classifier.buildClassifier(train);
      endTime = System.currentTimeMillis();
      pythonHandler.stopTracker();
      elapsedTime = (endTime - startTime) / 1000.0;
      loader.editCsv(elapsedTime);
      SerializationHelper.write(MODEL_FILE, classifier);
      System.out.println("Model saved to " + MODEL_FILE);

      System.out.println("Model training completed.");

    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public static void test(Instances data, String targetLabel) {
    try {
      System.out.println("Loading model from file...");
      classifier = (Classifier) SerializationHelper.read(MODEL_FILE);
      // Set the class index based on the target label
      Attribute classAttribute = data.attribute(targetLabel);
      if (classAttribute == null) {
        System.out.println("Error: Target label '" + targetLabel + "' not found in the dataset.");
        return;
      }
      data.setClass(classAttribute);

      // Convert class attribute to nominal if it's numeric
      if (classAttribute.isNumeric()) {
        NumericToNominal filter = new NumericToNominal();
        filter.setAttributeIndices("" + (data.classIndex() + 1)); // Class index is 1-based in the filter
        filter.setInputFormat(data);
        data = Filter.useFilter(data, filter);
        System.out.println("Converted numeric class attribute to nominal.");
      }

      // Apply standardization to features
      Standardize standardizeFilter = new Standardize();
      standardizeFilter.setInputFormat(data);
      Instances standardizedData = Filter.useFilter(data, standardizeFilter);

      // Stratify and split the data: 80% train, 20% test with random seed
      Instances[] split = stratifiedSplit(standardizedData, 0.8, 42);
      Instances test = split[1];

      // Evaluate the classifier on the test set
      Evaluation evaluation = new Evaluation(split[0]); // Pass training set for evaluation context
      pythonHandler.startTracker("emissions.csv");
      startTime = System.currentTimeMillis();
      evaluation.evaluateModel(classifier, test);
      endTime = System.currentTimeMillis();
      pythonHandler.stopTracker();
      elapsedTime = (endTime - startTime) / 1000.0;
      loader.editCsv(elapsedTime);

      // Output the accuracy
      System.out.println("DT Accuracy: " + evaluation.pctCorrect() + "%");

    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  // Function to split data into train and test sets (stratified split)
  private static Instances[] stratifiedSplit(Instances data, double trainRatio, int seed) throws Exception {
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

  public static void main(String[] args) {
    try {
      // Load the dataset
      DataSource source = new DataSource("path/to/your/data.arff");
      Instances data = source.getDataSet();

      String targetLabel = "class"; // Replace with your target label

      // Train the model
      train(data, targetLabel);

      // Test the model
      test(data, targetLabel);

    } catch (Exception e) {
      e.printStackTrace();
    }
  }
}
