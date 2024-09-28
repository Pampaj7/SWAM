package com.example;

import weka.classifiers.meta.AdaBoostM1;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.classifiers.Evaluation;

public class adaboost {
  private static final String MODEL_FILE = "adaboostModel.model";
  private static AdaBoostM1 adaboost;
  private static PythonHandler pythonHandler = new PythonHandler();

  public static void train(Instances data, String targetLabelName) throws Exception {
    // Set the class index based on the target label name
    data = loader.convertClassToNominal(data, targetLabelName);
    Instances[] split = loader.stratifiedSplit(data, 0.8, 42);
    Instances train = split[0];

    // Create and configure the AdaBoost model
    adaboost = new AdaBoostM1();
    adaboost.setNumIterations(100); // Number of iterations
    adaboost.setClassifier(new weka.classifiers.functions.SMO()); // Use SMO as the base classifier
    pythonHandler.startTracker("emissions.csv");
    adaboost.buildClassifier(train);
    pythonHandler.stopTracker();

    // Save the model to a file
    SerializationHelper.write(MODEL_FILE, adaboost);
    System.out.println("Model saved to " + MODEL_FILE);
  }

  public static void test(Instances data, String targetLabelName) throws Exception {
    // Load the model from the file
    if (adaboost == null) {
      adaboost = (AdaBoostM1) SerializationHelper.read(MODEL_FILE);
    }

    if (adaboost == null) {
      System.out.println("Error: Model could not be loaded.");
      return;
    }

    // Set the class index based on the target label name
    data = loader.convertClassToNominal(data, targetLabelName);
    Instances[] split = loader.stratifiedSplit(data, 0.8, 42);
    Instances test = split[1];

    Evaluation evaluation = new Evaluation(split[0]); // Pass training set for evaluation context
    pythonHandler.startTracker("emissions.csv");
    evaluation.evaluateModel(adaboost, test);
    pythonHandler.stopTracker();
    System.out.println("AdaBoost Accuracy: " + evaluation.pctCorrect() + "%");
  }
}
