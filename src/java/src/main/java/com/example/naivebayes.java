package com.example;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.classifiers.Evaluation;

public class naivebayes {
  private static final String MODEL_FILE = "naiveBayesModel.model";
  private static NaiveBayes naiveBayes;
  private static PythonHandler pythonHandler = new PythonHandler();

  public static void train(Instances data, String targetLabelName) throws Exception {
    // Set the class index based on the target label name
    data = loader.convertClassToNominal(data, targetLabelName);
    Instances[] split = loader.stratifiedSplit(data, 0.8, 42);
    Instances train = split[0];

    // Create and configure the NaiveBayes model
    naiveBayes = new NaiveBayes();
    pythonHandler.startTracker("emissions.csv");
    naiveBayes.buildClassifier(train);
    pythonHandler.stopTracker();
    // Save the model to a file
    SerializationHelper.write(MODEL_FILE, naiveBayes);
    System.out.println("Model saved to " + MODEL_FILE);
  }

  public static void test(Instances data, String targetLabelName) throws Exception {
    // Load the model from the file
    if (naiveBayes == null) {
      naiveBayes = (NaiveBayes) SerializationHelper.read(MODEL_FILE);
    }

    if (naiveBayes == null) {
      System.out.println("Error: Model could not be loaded.");
      return;
    }

    // Set the class index based on the target label name
    data = loader.convertClassToNominal(data, targetLabelName);
    Instances[] split = loader.stratifiedSplit(data, 0.8, 42);
    Instances test = split[1];

    // Evaluate the model using Weka's Evaluation class
    Evaluation evaluation = new Evaluation(split[0]);
    pythonHandler.startTracker("emissions.csv");
    evaluation.evaluateModel(naiveBayes, test);
    pythonHandler.stopTracker();
    System.out.println("Naive Bayes Test Accuracy: " + evaluation.pctCorrect() + "%");
  }

}
