package com.example;

import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Attribute;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;

public class knn {

  public static Instances convertClassToNominal(Instances data, String targetLabelName) {
    // Get the class index based on the target label name
    int targetIndex = data.attribute(targetLabelName).index();
    data.setClassIndex(targetIndex);

    // Check if the class attribute is numeric and convert it to nominal if
    // necessary
    if (data.classAttribute().isNumeric()) {
      List<String> uniqueValues = new ArrayList<>();
      HashMap<Double, String> valueMap = new HashMap<>();
      Enumeration<?> enu = data.enumerateInstances();

      // Gather unique numeric values to create nominal values
      while (enu.hasMoreElements()) {
        double value = ((weka.core.Instance) enu.nextElement()).classValue();
        String valueString = String.valueOf(value);
        if (!uniqueValues.contains(valueString)) {
          uniqueValues.add(valueString);
          valueMap.put(value, valueString);
        }
      }

      // Create a nominal attribute
      ArrayList<String> nominalValues = new ArrayList<>(uniqueValues);
      Attribute nominalClassAttribute = new Attribute(targetLabelName, nominalValues);

      // Replace the numeric class attribute with the new nominal attribute
      Instances newData = new Instances(data);
      newData.deleteAttributeAt(targetIndex);
      newData.insertAttributeAt(nominalClassAttribute, targetIndex);
      newData.setClassIndex(targetIndex);

      // Update the class values to nominal
      for (int i = 0; i < newData.numInstances(); i++) {
        double numericValue = data.instance(i).classValue();
        newData.instance(i).setClassValue(valueMap.get(numericValue));
      }
      return newData;
    }

    return data;
  }

  public static Classifier train(Instances data, String targetLabelName) throws Exception {
    // Set the class index based on the target label name
    int targetIndex = data.attribute(targetLabelName).index();
    data.setClassIndex(targetIndex);

    // Create and configure the k-NN model
    IBk knn = new IBk(); // Default k is 1
    knn.setKNN(3); // Set k to 3 for example
    knn.buildClassifier(data);
    double accuracy = evaluateModel(knn, data);
    System.out.println("knn Accuracy: " + accuracy * 100 + "%");

    return knn;
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
