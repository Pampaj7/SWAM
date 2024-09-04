package com.example;

import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.classifiers.Evaluation;

public class knn {

  public static long startTime;
  public static long endTime;
  public static double elapsedTime;
  private static final String MODEL_FILE = "knnModel.model";
  private static IBk knn;
  private static PythonHandler pythonHandler = new PythonHandler();

  public static void train(Instances data, String targetLabelName) throws Exception {
    // Set the class index based on the target label name
    int targetIndex = data.attribute(targetLabelName).index();
    data.setClassIndex(targetIndex);

    // Create and configure the k-NN model
    knn = new IBk();
    knn.setKNN(3); // Set k to 3 for example
    pythonHandler.startTracker("emissions.csv");
    startTime = System.currentTimeMillis();
    knn.buildClassifier(data);
    endTime = System.currentTimeMillis();
    elapsedTime = (endTime - startTime) / 1000.0;
    System.out.println("k-NN Training Time: " + elapsedTime + " seconds");
    pythonHandler.stopTracker();
    loader.editCsv(elapsedTime);

    // Save the model to a file
    SerializationHelper.write(MODEL_FILE, knn);
    System.out.println("Model saved to " + MODEL_FILE);
  }

  public static void test(Instances data, String targetLabelName) throws Exception {
    // Load the model from the file
    if (knn == null) {
      knn = (IBk) SerializationHelper.read(MODEL_FILE);
    }

    if (knn == null) {
      System.out.println("Error: Model could not be loaded.");
      return;
    }

    // Set the class index based on the target label name
    int targetIndex = data.attribute(targetLabelName).index();
    data.setClassIndex(targetIndex);

    // Evaluate the model using Weka's Evaluation class
    pythonHandler.startTracker("emissions.csv");
    startTime = System.currentTimeMillis();
    Evaluation evaluation = evaluateModel(knn, data);
    endTime = System.currentTimeMillis();
    pythonHandler.stopTracker();
    elapsedTime = (endTime - startTime) / 1000.0;
    loader.editCsv(elapsedTime);
    System.out.println("k-NN Test Accuracy: " + evaluation.pctCorrect());
  }

  private static Evaluation evaluateModel(IBk model, Instances data) throws Exception {
    // Perform a train-test split
    int trainSize = (int) (data.numInstances() * 0.8);
    int testSize = data.numInstances() - trainSize;
    Instances trainData = new Instances(data, 0, trainSize);
    Instances testData = new Instances(data, trainSize, testSize);

    // Initialize Evaluation object
    Evaluation evaluation = new Evaluation(trainData);
    // pythonHandler.startTracker("emissions.csv");
    evaluation.evaluateModel(model, testData);
    // pythonHandler.stopTracker();

    return evaluation;
  }
}
