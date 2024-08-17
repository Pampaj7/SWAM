package com.example;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

class LogisticRegression {
  private double[] weights;
  private double bias;
  private double learningRate;
  private int numIterations;

  public LogisticRegression(int nFeatures, double lr, int iterations) {
    this.weights = new double[nFeatures];
    this.bias = 0.0;
    this.learningRate = lr;
    this.numIterations = iterations;
  }

  private double sigmoid(double z) {
    return 1.0 / (1.0 + Math.exp(-z));
  }

  public void fit(List<double[]> X, List<Integer> y) {
    int nSamples = X.size();
    int nFeatures = X.get(0).length;

    for (int i = 0; i < numIterations; ++i) {
      double[] predictions = new double[nSamples];

      // Forward pass
      for (int j = 0; j < nSamples; ++j) {
        double z = bias;
        for (int k = 0; k < nFeatures; ++k) {
          z += weights[k] * X.get(j)[k];
        }
        predictions[j] = sigmoid(z);
      }

      // Compute gradients
      double[] dw = new double[nFeatures];
      double db = 0.0;

      for (int j = 0; j < nSamples; ++j) {
        double error = predictions[j] - y.get(j);
        for (int k = 0; k < nFeatures; ++k) {
          dw[k] += error * X.get(j)[k];
        }
        db += error;
      }

      // Update parameters
      for (int k = 0; k < nFeatures; ++k) {
        weights[k] -= learningRate * dw[k] / nSamples;
      }
      bias -= learningRate * db / nSamples;
    }
  }

  public List<Integer> predict(List<double[]> X) {
    List<Integer> predictions = new ArrayList<>();
    for (double[] sample : X) {
      double z = bias;
      for (int i = 0; i < weights.length; ++i) {
        z += weights[i] * sample[i];
      }
      predictions.add(sigmoid(z) >= 0.5 ? 1 : 0);
    }
    return predictions;
  }
}

public class main {
  public static List<double[]> X = new ArrayList<>();
  public static List<Integer> y = new ArrayList<>();

  public static void main(String[] args) {
    String datasetPath = "../../../datasets/breastcancer/breastcancer.csv";
    loadData(datasetPath);

    // Split data into train and test sets (simple 80-20 split)
    int splitIndex = (int) (X.size() * 0.8);
    List<double[]> XTrain = X.subList(0, splitIndex);
    List<double[]> XTest = X.subList(splitIndex, X.size());
    List<Integer> yTrain = y.subList(0, splitIndex);
    List<Integer> yTest = y.subList(splitIndex, y.size());

    // Create and train the model
    LogisticRegression model = new LogisticRegression(X.get(0).length, 0.001, 10000);
    model.fit(XTrain, yTrain);

    // Make predictions
    List<Integer> predictions = model.predict(XTest);

    // Calculate accuracy
    int correct = 0;
    for (int i = 0; i < predictions.size(); ++i) {
      if (predictions.get(i).equals(yTest.get(i))) {
        correct++;
      }
    }
    double accuracy = (double) correct / predictions.size();
    System.out.println("Accuracy: " + accuracy * 100 + "%");
  }

  public static void loadData(String filePath) {
    try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
      String line;
      boolean header = true;
      while ((line = br.readLine()) != null) {
        if (header) {
          header = false;
          continue;
        }
        String[] values = line.split(",");
        double[] features = new double[values.length - 2];
        for (int i = 2; i < values.length; i++) {
          features[i - 2] = Double.parseDouble(values[i]);
        }
        X.add(features);
        y.add(values[1].equals("M") ? 1 : 0);
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}
