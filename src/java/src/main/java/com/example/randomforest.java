package com.example;

import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.classifiers.Evaluation;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.io.File;

public class randomforest {

  public static void train(Instances data, String targetLabel) {
    try {
      // Set the class index based on the target label
      Attribute classAttribute = data.attribute(targetLabel);
      if (classAttribute == null) {
        System.out.println("Error: Target label '" + targetLabel + "' not found in the dataset.");
        return;
      }
      data.setClass(classAttribute);

      // Convert class attribute to nominal if it's numeric
      if (classAttribute.isNumeric()) {
        NumericToNominal filter = new NumericToNominal();
        filter.setAttributeIndices("" + (data.classIndex() + 1)); // Class index is 1-based in the filter
        filter.setInputFormat(data);
        data = Filter.useFilter(data, filter);
        System.out.println("Converted numeric class attribute to nominal.");
      }

      // If the class attribute is nominal but not binary, convert it to binary
      if (classAttribute.numValues() > 2) {
        // Convert nominal classes to binary values (0 and 1)
        Attribute classAttributeAfterConversion = data.classAttribute();
        Map<String, Double> classMapping = new HashMap<>();
        for (int i = 0; i < classAttributeAfterConversion.numValues(); i++) {
          classMapping.put(classAttributeAfterConversion.value(i), (double) i);
        }

        for (int i = 0; i < data.numInstances(); i++) {
          double classValue = classMapping.get(data.instance(i).stringValue(classAttributeAfterConversion));
          data.instance(i).setClassValue(classValue);
        }
      }

      // Apply standardization to features
      Standardize standardizeFilter = new Standardize();
      standardizeFilter.setInputFormat(data);
      Instances standardizedData = Filter.useFilter(data, standardizeFilter);

      // Split the data into training and testing sets
      Instances[] split = stratifiedSplit(standardizedData, 0.8, 42);
      Instances trainData = split[0];
      Instances testData = split[1];

      // Build and train the Random Forest model
      RandomForest forest = new RandomForest();
      forest.setNumIterations(100); // Number of trees in the forest
      forest.setSeed(42); // Random seed
      forest.buildClassifier(trainData);

      // Evaluate the model on the testing data
      Evaluation evaluation = new Evaluation(trainData);
      evaluation.evaluateModel(forest, testData);

      // Print accuracy and classification report
      System.out.println("RF Accuracy: " + evaluation.pctCorrect() + "%");

    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  // Function to split data into train and test sets (stratified split)
  private static Instances[] stratifiedSplit(Instances data, double trainRatio, int seed) throws Exception {
    // Ensure the data is randomized before splitting
    Random rand = new Random(seed);
    data.randomize(rand);

    // Split the dataset
    int trainSize = (int) Math.round(data.numInstances() * trainRatio);
    int testSize = data.numInstances() - trainSize;

    Instances train = new Instances(data, 0, trainSize);
    Instances test = new Instances(data, trainSize, testSize);

    return new Instances[] { train, test };
  }
}
