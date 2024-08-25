package com.example;

import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class svc {

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

  public static Classifier train(Instances data, String targetLabelName) throws Exception {
    // Set the class index based on the target label name
    data = convertClassToNominal(data, targetLabelName);
    data.setClassIndex(data.attribute(targetLabelName).index());

    // Create and configure the SVC (SMO) model
    SMO smo = new SMO();
    smo.buildClassifier(data);
    double accuracy = evaluateModel(smo, data);
    System.out.println("SVC Accuracy: " + accuracy);

    return smo;
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
