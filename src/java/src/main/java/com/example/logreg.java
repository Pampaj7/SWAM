package com.example;

import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.core.SerializationHelper;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.IOException;
import java.io.*;
import java.util.concurrent.*;

public class logreg {

  private static final String MODEL_FILE = "logisticModel.model";
  private static Logistic logistic;

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

  public static void train(Instances data, String targetLabelName) throws Exception {
    ProcessBuilder pb = new ProcessBuilder(
        "/Users/niccolomarini/Documents/GitHub/SWAM/.venv/bin/python3",
        "/Users/niccolomarini/Documents/GitHub/SWAM/src/java/src/main/python/tracker_control.py");
    Process process = pb.start();

    ExecutorService executor = Executors.newFixedThreadPool(2);

    try (
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(process.getOutputStream()));
        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()))) {

      // Start reading output and error in separate threads
      Future<?> outputFuture = executor.submit(() -> readStream(reader, "Python output"));
      Future<?> errorFuture = executor.submit(() -> readStream(errorReader, "Python error"));

      // Start tracker
      writer.write("start emissions.csv\n");
      writer.flush();
      Thread.sleep(10000);

      // Perform data processing
      data.setClassIndex(data.attribute(targetLabelName).index());
      data = convertClassToNominal(data, targetLabelName);

      logistic = new Logistic();
      logistic.buildClassifier(data);

      // Stop tracker
      writer.write("stop\n");
      writer.flush();

      // Exit Python script
      writer.write("exit\n");
      writer.flush();

      // Wait for the process to complete with a timeout
      if (!process.waitFor(30, TimeUnit.SECONDS)) {
        System.err.println("Python process did not finish in time. Forcibly terminating.");
        process.destroyForcibly();
      }

      // Shutdown the executor and wait for stream reading to complete
      executor.shutdown();
      if (!executor.awaitTermination(5, TimeUnit.SECONDS)) {
        System.err.println("Stream reading did not complete in time.");
        executor.shutdownNow();
      }

      int exitCode = process.exitValue();
      System.out.println("Python process exited with code: " + exitCode);

      // Save the model to a file
      SerializationHelper.write(MODEL_FILE, logistic);
      System.out.println("Model saved to " + MODEL_FILE);
    } catch (Exception e) {
      e.printStackTrace();
    } finally {
      executor.shutdownNow(); // Ensure executor is shut down even if an exception occurs
      process.destroyForcibly(); // Ensure the process is terminated
    }
  }

  private static void readStream(BufferedReader reader, String streamName) {
    try {
      String line;
      while ((line = reader.readLine()) != null) {
        System.out.println(streamName + ": " + line);
      }
    } catch (IOException e) {
      System.err.println("Error reading " + streamName + ": " + e.getMessage());
    }
  }

  public static void test(Instances data, String targetLabelName) throws Exception {
    // Load the model from the file
    if (logistic == null) {
      logistic = (Logistic) SerializationHelper.read(MODEL_FILE);
    }

    if (logistic == null) {
      System.out.println("Error: Model could not be loaded.");
      return;
    }

    // Set the class index based on the target label name
    data.setClassIndex(data.attribute(targetLabelName).index());
    data = convertClassToNominal(data, targetLabelName);

    double accuracy = evaluateModel(logistic, data);
    System.out.println("Logistic Regression Test Accuracy: " + accuracy);
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
