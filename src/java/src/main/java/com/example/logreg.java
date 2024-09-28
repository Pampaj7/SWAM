package com.example;

import weka.classifiers.functions.Logistic;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.classifiers.Evaluation;

public class logreg {
  public static long startTime;
  public static long endTime;
  public static double elapsedTime;
  private static final String MODEL_FILE = "logisticModel.model";
  private static Logistic logistic;
  private static PythonHandler pythonHandler = new PythonHandler();

  public static void train(Instances data, String targetLabelName) {
    try {
      // Perform data processing and training
      data = loader.convertClassToNominal(data, targetLabelName);
      Instances[] split = loader.stratifiedSplit(data, 0.8, 42);
      Instances train = split[0];

      logistic = new Logistic();
      String[] options = new String[2];
      options[0] = "-M";
      options[1] = "10000";
      logistic.setOptions(options);

      pythonHandler.startTracker("emissions.csv");
      logistic.buildClassifier(train);
      pythonHandler.stopTracker();

      // Save the model to a file
      SerializationHelper.write(MODEL_FILE, logistic);
      System.out.println("Model saved to " + MODEL_FILE);
    } catch (Exception e) {
      e.printStackTrace();
    } finally {
      pythonHandler.cleanup();
    }
  }

  public static void test(Instances data, String targetLabelName) throws Exception {
    if (logistic == null) {
      logistic = (Logistic) SerializationHelper.read(MODEL_FILE);
    }

    if (logistic == null) {
      System.out.println("Error: Model could not be loaded.");
      return;
    }

    data = loader.convertClassToNominal(data, targetLabelName);
    Instances[] split = loader.stratifiedSplit(data, 0.8, 42);
    Instances test = split[1];

    Evaluation evaluation = new Evaluation(split[0]);
    pythonHandler.startTracker("emissions.csv");
    evaluation.evaluateModel(logistic, test);
    pythonHandler.stopTracker();
    System.out.println("Logistic Regression Accuracy: " + evaluation.pctCorrect() + "%");

  }
}
