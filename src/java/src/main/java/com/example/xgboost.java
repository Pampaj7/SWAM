package com.example;

import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;
import ml.dmlc.xgboost4j.java.Booster;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics; // For feature scaling
import weka.core.Attribute;
import weka.core.Instances;
import java.util.*;
import java.util.stream.IntStream;

public class xgboost {

  public static void train(Instances data, String targetLabel) throws XGBoostError {
    try {
      // Set the class index based on the target label
      Attribute classAttribute = data.attribute(targetLabel);
      if (classAttribute == null) {
        System.out.println("Error: Target label '" + targetLabel + "' not found in the dataset.");
        return;
      }
      data.setClass(classAttribute);

      // Extract features and labels
      List<double[]> featuresList = new ArrayList<>();
      List<Float> labelsList = new ArrayList<>();
      for (int i = 0; i < data.numInstances(); i++) {
        double[] instanceValues = data.instance(i).toDoubleArray();
        double[] features = new double[instanceValues.length - 1]; // Exclude class attribute
        System.arraycopy(instanceValues, 0, features, 0, features.length);
        featuresList.add(features);

        // Convert class attribute to numeric label (0 or 1)
        double label = data.instance(i).classValue();
        if (label != 0 && label != 1) {
          // System.out.println("Warning: Label '" + label + "' is not binary. Mapping it
          // to 0.");
          label = 0;
        }
        labelsList.add((float) label);
      }

      // Convert lists to arrays
      int numRows = featuresList.size();
      int numCols = featuresList.get(0).length;
      float[] flattenedFeatures = new float[numRows * numCols];
      float[] labelsArray = new float[labelsList.size()];

      for (int i = 0; i < numRows; i++) {
        double[] row = featuresList.get(i);
        for (int j = 0; j < numCols; j++) {
          flattenedFeatures[i * numCols + j] = (float) row[j];
        }
        labelsArray[i] = labelsList.get(i);
      }

      // Shuffle and split the data
      int splitIndex = (int) (0.8 * numRows); // 80% train, 20% test
      int[] indices = IntStream.range(0, numRows).toArray();
      shuffleArray(indices);

      // Feature scaling
      double[][] scaledFeatures = new double[numRows][numCols];
      for (int i = 0; i < numCols; i++) {
        double[] column = new double[numRows];
        for (int j = 0; j < numRows; j++) {
          column[j] = featuresList.get(j)[i];
        }
        DescriptiveStatistics stats = new DescriptiveStatistics(column);
        double mean = stats.getMean();
        double std = stats.getStandardDeviation();
        for (int j = 0; j < numRows; j++) {
          scaledFeatures[j][i] = (column[j] - mean) / std;
        }
      }

      // Convert scaledFeatures to float[] for DMatrix
      float[] flattenedScaledFeatures = new float[numRows * numCols];
      for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
          flattenedScaledFeatures[i * numCols + j] = (float) scaledFeatures[i][j];
        }
      }

      // Create DMatrix with scaled features
      DMatrix trainData = createDMatrix(flattenedScaledFeatures, labelsArray, indices, 0, splitIndex, numCols);
      DMatrix testData = createDMatrix(flattenedScaledFeatures, labelsArray, indices, splitIndex, numRows, numCols);

      // Set parameters
      Map<String, Object> params = new HashMap<>();
      params.put("objective", "binary:logistic");
      params.put("max_depth", 3);
      params.put("eta", 0.1);
      params.put("silent", 1);

      // Train the model
      Booster booster = XGBoost.train(trainData, params, 100, new HashMap<>(), null, null);

      // Make predictions
      float[][] predictions = booster.predict(testData);

      // Evaluate the model
      float[] testLabels = Arrays.copyOfRange(labelsArray, splitIndex, numRows);
      evaluate(testLabels, predictions);
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  private static DMatrix createDMatrix(float[] features, float[] labels, int[] indices, int start, int end, int numCols)
      throws XGBoostError {
    int numRows = end - start;
    float[] subFeatures = new float[numRows * numCols];
    float[] subLabels = new float[numRows];

    for (int i = start; i < end; i++) {
      int idx = indices[i];
      System.arraycopy(features, idx * numCols, subFeatures, (i - start) * numCols, numCols);
      subLabels[i - start] = labels[idx];
    }

    DMatrix dMatrix = new DMatrix(subFeatures, numRows, numCols);
    dMatrix.setLabel(subLabels);
    return dMatrix;
  }

  private static void shuffleArray(int[] array) {
    Random rnd = new Random(42); // Set random seed for reproducibility
    for (int i = array.length - 1; i > 0; i--) {
      int index = rnd.nextInt(i + 1);
      // Swap elements
      int temp = array[index];
      array[index] = array[i];
      array[i] = temp;
    }
  }

  private static void evaluate(float[] trueLabels, float[][] predictions) {
    int correct = 0;
    int numPredictions = predictions.length;

    // Ensure predictions and trueLabels lengths match
    if (trueLabels.length != numPredictions) {
      throw new IllegalArgumentException("Mismatch between trueLabels and predictions lengths.");
    }

    for (int i = 0; i < numPredictions; i++) {
      float predictedLabel = (predictions[i][0] > 0.5f) ? 1f : 0f;
      if (predictedLabel == trueLabels[i]) {
        correct++;
      }
    }
    float accuracy = (float) correct / trueLabels.length;
    System.out.println("xgboost Accuracy: " + accuracy);
  }
}
