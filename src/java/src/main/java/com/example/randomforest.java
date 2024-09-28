package com.example;

import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.classifiers.Evaluation;
import weka.core.SerializationHelper;

public class randomforest {

  private static RandomForest forest;
  private static final String MODEL_FILE = "randomForestModel.model";
  private static PythonHandler pythonHandler = new PythonHandler();

  public static void train(Instances data, String targetLabel) {
    try {

      data = loader.convertClassToNominal(data, targetLabel);

      // Stratify and split the data: 80% train, 20% test with random seed
      Instances[] split = loader.stratifiedSplit(data, 0.8, 42);
      Instances trainData = split[0];

      // Verify if the training data is correctly set
      if (trainData.numInstances() == 0) {
        System.out.println("Error: Training data is empty.");
        return;
      }

      // Build and train the Random Forest model
      forest = new RandomForest();
      forest.setNumIterations(100); // Number of trees in the forest
      forest.setSeed(42); // Random seed
      pythonHandler.startTracker("emissions.csv");
      forest.buildClassifier(trainData);
      pythonHandler.stopTracker();
      // Save the model to a file
      SerializationHelper.write(MODEL_FILE, forest);
      System.out.println("Model saved to " + MODEL_FILE);

    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public static void test(Instances data, String targetLabel) {
    try {
      // Load the model from the file
      System.out.println("Loading model from file...");
      forest = (RandomForest) SerializationHelper.read(MODEL_FILE);

      if (forest == null) {
        System.out.println("Error: Model could not be loaded.");
        return;
      }

      data = loader.convertClassToNominal(data, targetLabel);
      // Stratify and split the data: 80% train, 20% test with random seed
      Instances[] split = loader.stratifiedSplit(data, 0.8, 42);
      Instances testData = split[1];

      // Evaluate the model on the testing data
      Evaluation evaluation = new Evaluation(split[0]);
      pythonHandler.startTracker("emissions.csv");
      evaluation.evaluateModel(forest, testData);
      pythonHandler.stopTracker();

      // Print accuracy and classification report
      System.out.println("RF Accuracy: " + evaluation.pctCorrect() + "%");

    } catch (Exception e) {
      e.printStackTrace();
    }
  }
}
