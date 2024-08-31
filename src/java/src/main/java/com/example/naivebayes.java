package com.example;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.classifiers.Evaluation;

import java.util.Random;

public class naivebayes {

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

      // Apply standardization to features
      Standardize standardizeFilter = new Standardize();
      standardizeFilter.setInputFormat(data);
      Instances standardizedData = Filter.useFilter(data, standardizeFilter);

      // Stratify and split the data: 80% train, 20% test with random seed
      Instances[] split = stratifiedSplit(standardizedData, 0.8, 42);
      Instances train = split[0];
      Instances test = split[1];

      // Create and train the Naive Bayes classifier
      Classifier classifier = new NaiveBayes();
      classifier.buildClassifier(train);

      // Evaluate the classifier on the test set
      Evaluation evaluation = new Evaluation(train);
      evaluation.evaluateModel(classifier, test);

      // Output the accuracy
      System.out.println("Naive Bayes Accuracy: " + evaluation.pctCorrect() + "%");

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

  public static void main(String[] args) {
    try {
      // Load the dataset
      DataSource source = new DataSource("path/to/your/dataset.arff");
      Instances data = source.getDataSet();

      // Call the train method with the target label
      train(data, "targetLabel"); // Replace "targetLabel" with the actual target attribute name

    } catch (Exception e) {
      e.printStackTrace();
    }
  }
}
