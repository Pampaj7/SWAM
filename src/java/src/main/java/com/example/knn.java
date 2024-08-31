package com.example;

import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;

public class knn {

  private static final String MODEL_FILE = "knnModel.model";
  private static IBk knn;

  public static void train(Instances data, String targetLabelName) throws Exception {
    int targetIndex = data.attribute(targetLabelName).index();
    data.setClassIndex(targetIndex);

    // Set the class index based on the target label name

    // Create and configure the k-NN model
    knn = new IBk();
    knn.setKNN(3); // Set k to 3 for example
    knn.buildClassifier(data);

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
    data.setClassIndex(data.attribute(targetLabelName).index());

    double accuracy = evaluateModel(knn, data);
    System.out.println("k-NN Test Accuracy: " + accuracy * 100 + "%");
  }

  private static double evaluateModel(Classifier model, Instances data) throws Exception {
    // Perform a train-test split
    int trainSize = (int) (data.numInstances() * 0.8);
    int testSize = data.numInstances() - trainSize;
    Instances trainData = new Instances(data, 0, trainSize);
    Instances testData = new Instances(data, trainSize, testSize);

    // Evaluate the model
    int correct = 0;
    for (int i = 0; i < testData.numInstances(); i++) {
      double actualClass = testData.instance(i).classValue();
      double predictedClass = model.classifyInstance(testData.instance(i));

      if (actualClass == predictedClass) {
        correct++;
      }
    }
    return (double) correct / testData.numInstances();
  }

}
