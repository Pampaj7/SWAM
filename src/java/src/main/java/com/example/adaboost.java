package com.example;

import scala.language;
import weka.classifiers.Classifier;
import weka.classifiers.meta.AdaBoostM1;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.classifiers.Evaluation;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class adaboost {
  public static long startTime;
  public static long endTime;
  public static double elapsedTime;
  private static final String MODEL_FILE = "adaboostModel.model";
  private static AdaBoostM1 adaboost;
  private static PythonHandler pythonHandler = new PythonHandler();

  public static Instances convertClassToNominal(Instances data, String targetLabelName) throws Exception {
    int targetIndex = data.attribute(targetLabelName).index();
    data.setClassIndex(targetIndex);

    // Apply NumericToNominal filter if the class attribute is numeric
    if (data.classAttribute().isNumeric()) {
      NumericToNominal convert = new NumericToNominal();
      String[] options = new String[] { "-R", String.valueOf(targetIndex + 1) };
      convert.setOptions(options);
      convert.setInputFormat(data);
      data = Filter.useFilter(data, convert);
    }

    return data;
  }

  public static void train(Instances data, String targetLabelName) throws Exception {
    // Set the class index based on the target label name
    data = convertClassToNominal(data, targetLabelName);
    data.setClassIndex(data.attribute(targetLabelName).index());

    // Create and configure the AdaBoost model
    adaboost = new AdaBoostM1();
    adaboost.setClassifier(new weka.classifiers.functions.SMO()); // Use SMO as the base classifier
    pythonHandler.startTracker("emissions.csv");
    startTime = System.currentTimeMillis();
    adaboost.buildClassifier(data);
    endTime = System.currentTimeMillis();
    pythonHandler.stopTracker();
    elapsedTime = (endTime - startTime) / 1000.0;
    loader.editCsv(elapsedTime);

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
    data = convertClassToNominal(data, targetLabelName);
    data.setClassIndex(data.attribute(targetLabelName).index());

    // Evaluate the model using Weka's Evaluation class
    Evaluation evaluation = evaluateModel(adaboost, data);
    System.out.println("AdaBoost Test Accuracy: " + evaluation.pctCorrect());
  }

  private static Evaluation evaluateModel(AdaBoostM1 model, Instances data) throws Exception {
    // Perform a train-test split
    int trainSize = (int) (data.numInstances() * 0.8);
    int testSize = data.numInstances() - trainSize;
    Instances trainData = new Instances(data, 0, trainSize);
    Instances testData = new Instances(data, trainSize, testSize);

    // Initialize Evaluation object
    Evaluation evaluation = new Evaluation(trainData);
    pythonHandler.startTracker("emissions.csv");
    startTime = System.currentTimeMillis();
    evaluation.evaluateModel(model, testData);
    endTime = System.currentTimeMillis();
    pythonHandler.stopTracker();
    elapsedTime = (endTime - startTime) / 1000.0;
    loader.editCsv(elapsedTime);

    return evaluation;
  }
}
