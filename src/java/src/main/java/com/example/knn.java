package com.example;

import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.classifiers.Evaluation;

public class knn {

  private static final String MODEL_FILE = "knnModel.model";
  private static IBk knn;
  private static PythonHandler pythonHandler = new PythonHandler();

  public static void train(Instances data, String targetLabelName) throws Exception {
    // Set the class index based on the target label name
    int targetIndex = data.attribute(targetLabelName).index();
    data.setClassIndex(targetIndex);
    Instances[] split = loader.stratifiedSplit(data, 0.8, 42);
    Instances train = split[0];

    // Create and configure the k-NN model
    knn = new IBk();
    knn.setKNN(5); // Set k to 3 for example
    pythonHandler.startTracker("emissions.csv");
    knn.buildClassifier(train);
    pythonHandler.stopTracker();

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
    Instances[] split = loader.stratifiedSplit(data, 0.8, 42);
    Instances test = split[1];

    Evaluation evaluation = new Evaluation(split[0]); // Pass training set for evaluation context
    pythonHandler.startTracker("emissions.csv");
    evaluation.evaluateModel(knn, test);
    pythonHandler.stopTracker();
    // System.out.println("k-NN Accuracy: " + evaluation.toSummaryString() + "%");
  }
}
