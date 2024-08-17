package com.example;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Attribute;
import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class main {
  public static void main(String[] args) throws Exception {
    // Load the dataset
    CSVLoader loader = new CSVLoader();
    loader.setSource(new File("../../../datasets/breastcancer/breastcancer.csv"));
    Instances data = loader.getDataSet();

    // Assuming "diagnosis" is the last attribute and needs to be mapped to 0 and 1
    // Make sure to set the class index accordingly
    data.setClassIndex(1);

    // Convert class attribute to binary values (0 and 1)
    Attribute classAttribute = data.classAttribute();
    Map<String, Double> classMapping = new HashMap<>();
    classMapping.put("M", 1.0);
    classMapping.put("B", 0.0);

    for (int i = 0; i < data.numInstances(); i++) {
      double classValue = classMapping.get(data.instance(i).stringValue(classAttribute));
      data.instance(i).setClassValue(classValue);
    }

    // Split the data into training and testing sets
    int trainSize = (int) (data.numInstances() * 0.8);
    int testSize = data.numInstances() - trainSize;
    Instances trainData = new Instances(data, 0, trainSize);
    Instances testData = new Instances(data, trainSize, testSize);

    // Build and train the Random Forest model
    RandomForest forest = new RandomForest();
    forest.setNumIterations(100); // n_estimators in Python
    forest.setSeed(42); // random_state in Python
    forest.buildClassifier(trainData);

    // Evaluate the model on the testing data
    Evaluation evaluation = new Evaluation(trainData);
    evaluation.evaluateModel(forest, testData);

    // Print accuracy and classification report
    System.out.println("Accuracy: " + evaluation.pctCorrect() + "%");
  }
}
