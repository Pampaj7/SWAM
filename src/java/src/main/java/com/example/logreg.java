package com.example;

import scala.tools.nsc.doc.html.HtmlTags.P;
import weka.classifiers.functions.Logistic;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.core.SerializationHelper;

public class logreg {

  private static final String MODEL_FILE = "logisticModel.model";
  private static Logistic logistic;
  private static PythonHandler pythonHandler = new PythonHandler();

  public static Instances convertClassToNominal(Instances data, String targetLabelName) throws Exception {
    int targetIndex = data.attribute(targetLabelName).index();
    data.setClassIndex(targetIndex);

    if (data.classAttribute().isNumeric()) {
      NumericToNominal convert = new NumericToNominal();
      String[] options = new String[] { "-R", String.valueOf(targetIndex + 1) };
      convert.setOptions(options);
      convert.setInputFormat(data);
      data = Filter.useFilter(data, convert);
    }

    return data;
  }

  public static void train(Instances data, String targetLabelName) {
    try {
      // Start the tracker

      // Perform data processing and training
      data = convertClassToNominal(data, targetLabelName);

      logistic = new Logistic();

      pythonHandler.startTracker("emissions.csv");
      logistic.buildClassifier(data);

      // Stop the tracker
      pythonHandler.stopTracker();

      // Save the model to a file
      SerializationHelper.write(MODEL_FILE, logistic);
      System.out.println("Model saved to " + MODEL_FILE);
    } catch (Exception e) {
      e.printStackTrace();
    } finally {
      pythonHandler.cleanup();
    }
  }

  public static void test(Instances data, String targetLabelName) throws Exception {
    if (logistic == null) {
      logistic = (Logistic) SerializationHelper.read(MODEL_FILE);
    }

    if (logistic == null) {
      System.out.println("Error: Model could not be loaded.");
      return;
    }

    data.setClassIndex(data.attribute(targetLabelName).index());
    data = convertClassToNominal(data, targetLabelName);

    pythonHandler.startTracker("emissions.csv");
    double accuracy = evaluateModel(logistic, data);
    pythonHandler.stopTracker();
    System.out.println("Logistic Regression Test Accuracy: " + accuracy);
  }

  private static double evaluateModel(Logistic model, Instances data) throws Exception {
    int trainSize = (int) (data.numInstances() * 0.8);
    int testSize = data.numInstances() - trainSize;
    Instances trainData = new Instances(data, 0, trainSize);
    Instances testData = new Instances(data, trainSize, testSize);

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
