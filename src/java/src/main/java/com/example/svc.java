package com.example;

import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.core.SerializationHelper;
import weka.classifiers.Evaluation;

public class svc {

  private static final String MODEL_FILE = "svcModel.model";
  private static SMO smo;
  private static PythonHandler pythonHandler = new PythonHandler();

  public static void train(Instances data, String targetLabelName) throws Exception {
    // Set the class index based on the target label name
    data = loader.convertClassToNominal(data, targetLabelName);
    Instances[] split = loader.stratifiedSplit(data, 0.8, 42);
    Instances train = split[0];

    // Create and configure the SVC (SMO) model
    smo = new SMO();
    pythonHandler.startTracker("emissions.csv");
    smo.buildClassifier(train);
    pythonHandler.stopTracker();
    // Save the model to a file
    SerializationHelper.write(MODEL_FILE, smo);
    System.out.println("Model saved to " + MODEL_FILE);
  }

  public static void test(Instances data, String targetLabelName) throws Exception {
    // Load the model from the file
    if (smo == null) {
      smo = (SMO) SerializationHelper.read(MODEL_FILE);
    }

    if (smo == null) {
      System.out.println("Error: Model could not be loaded.");
      return;
    }

    // Set the class index based on the target label name
    data = loader.convertClassToNominal(data, targetLabelName);
    Instances[] split = loader.stratifiedSplit(data, 0.8, 42);
    Instances test = split[1];

    data.setClassIndex(data.attribute(targetLabelName).index());

    // Evaluate the model using Weka's Evaluation class
    Evaluation evaluation = new Evaluation(split[0]); // Pass training set for evaluation context
    pythonHandler.startTracker("emissions.csv");
    evaluation.evaluateModel(smo, test);
    pythonHandler.stopTracker();
    System.out.println("SVC Test Accuracy: " + evaluation.pctCorrect());
  }
}
