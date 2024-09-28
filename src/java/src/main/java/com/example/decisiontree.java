package com.example;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.classifiers.Evaluation;
import weka.core.SerializationHelper;

public class decisiontree {

  private static PythonHandler pythonHandler = new PythonHandler();

  // Classifier instance that will be trained and tested
  private static final String MODEL_FILE = "decisionTree.model";
  private static Classifier classifier;

  public static void train(Instances data, String targetLabel) {
    try {

      // convert class attribute to nominal since that's what WEKA wants
      data = loader.convertClassToNominal(data, targetLabel);

      Instances[] split = loader.stratifiedSplit(data, 0.8, 42);
      Instances train = split[0];

      // Create and train the decision tree (J48) classifier
      classifier = new J48(); // J48 is the Weka implementation of C4.5
      pythonHandler.startTracker("emissions.csv");
      classifier.buildClassifier(train);
      pythonHandler.stopTracker();
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

      // Convert class attribute to nominal since that's what WEKA wants
      data = loader.convertClassToNominal(data, "target");

      // Stratify and split the data: 80% train, 20% test with random seed
      Instances[] split = loader.stratifiedSplit(data, 0.8, 42);
      Instances test = split[1];

      // Evaluate the classifier on the test set
      Evaluation evaluation = new Evaluation(split[0]); // Pass training set for evaluation context
      pythonHandler.startTracker("emissions.csv");
      evaluation.evaluateModel(classifier, test);
      pythonHandler.stopTracker();

      // Output the accuracy
      System.out.println("DT Accuracy: " + evaluation.pctCorrect() + "%");

    } catch (Exception e) {
      e.printStackTrace();
    }
  }
}
