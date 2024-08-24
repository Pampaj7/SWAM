package com.example;

import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;
import ml.dmlc.xgboost4j.java.Booster;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics; // For feature scaling
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.stream.IntStream;

public class main {

  public static void main(String[] args) throws XGBoostError, IOException {
    // Load the dataset
    String filePath = "../../../datasets/breastcancer/breastcancer.csv";
    List<float[]> featuresList = new ArrayList<>();
    List<Float> labelsList = new ArrayList<>();

    try (CSVParser parser = new CSVParser(new FileReader(filePath), CSVFormat.DEFAULT.withFirstRecordAsHeader())) {
      for (CSVRecord record : parser) {
        // Skip the ID column and process the rest
        float[] features = new float[record.size() - 2]; // Skip ID and Diagnosis
        for (int i = 2; i < record.size(); i++) { // Start from index 2 to skip ID and Diagnosis
          features[i - 2] = Float.parseFloat(record.get(i));
        }
        featuresList.add(features);

        // Convert Diagnosis to numeric label (M = 1, B = 0)
        String diagnosis = record.get(1);
        labelsList.add(diagnosis.equals("M") ? 1f : 0f);
      }
    }

    // Convert lists to arrays
    int numRows = featuresList.size();
    int numCols = featuresList.get(0).length;

    float[] flattenedFeatures = new float[numRows * numCols];
    float[] labelsArray = new float[labelsList.size()];

    for (int i = 0; i < numRows; i++) {
      float[] row = featuresList.get(i);
      System.arraycopy(row, 0, flattenedFeatures, i * numCols, numCols);
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
    // ... (flatten scaledFeatures)

    // Create DMatrix with scaled features
    DMatrix trainData = createDMatrix(flattenedScaledFeatures, labelsArray, indices, 0, splitIndex, numCols);
    DMatrix testData = createDMatrix(flattenedScaledFeatures, labelsArray, indices, splitIndex, numRows, numCols);
    // Create training and testing data
    // DMatrix trainData = createDMatrix(flattenedFeatures, labelsArray, indices, 0,
    // splitIndex, numCols);
    // DMatrix testData = createDMatrix(flattenedFeatures, labelsArray, indices,
    // splitIndex, numRows, numCols);

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
    System.out.println("Accuracy: " + accuracy);
  }
}
