package com.example;

import java.io.*;
import java.util.concurrent.*;

public class PythonHandler {

  private Process process;
  private ExecutorService executor;

  public void startTracker(String outputFile) throws IOException, InterruptedException {
    ProcessBuilder pb = new ProcessBuilder(
        "/Users/pampaj/anaconda3/envs/sw/bin/python",
        "/Users/pampaj/PycharmProjects/SWAM/src/java/src/main/python/tracker_control.py");
    // "/Users/niccolomarini/Documents/GitHub/SWAM/.venv/bin/python",
    // "/Users/niccolomarini/Documents/GitHub/SWAM/src/java/src/main/python/tracker_control.py");
    this.process = pb.start();
    this.executor = Executors.newFixedThreadPool(2);

    BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(process.getOutputStream()));
    BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
    BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));

    // Start reading output and error in separate threads
    executor.submit(() -> readStream(reader, "Python output"));
    executor.submit(() -> readStream(errorReader, "Python error"));

    // Send the start command to the Python script
    writer.write("start " + outputFile + "\n");
    writer.flush();
    // Thread.sleep(10000); // Sleep to ensure the tracker starts
  }

  public void stopTracker() throws IOException, InterruptedException {
    if (process != null) {
      BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(process.getOutputStream()));

      // Send the stop command to the Python script
      writer.write("stop\n");
      writer.flush();

      // Send the exit command to the Python script
      writer.write("exit\n");
      writer.flush();

      stopPythonScript();
    }
  }

  private void readStream(BufferedReader reader, String streamName) {
    try {
      String line;
      while ((line = reader.readLine()) != null) {
        System.out.println(streamName + ": " + line);
      }
    } catch (IOException e) {
      System.err.println("Error reading " + streamName + ": " + e.getMessage());
    }
  }

  private void stopPythonScript() throws InterruptedException, IOException {
    if (process != null) {
      if (!process.waitFor(30, TimeUnit.SECONDS)) {
        System.err.println("Python process did not finish in time. Forcibly terminating.");
        process.destroyForcibly();
      }

      executor.shutdown();
      if (!executor.awaitTermination(5, TimeUnit.SECONDS)) {
        System.err.println("Stream reading did not complete in time.");
        executor.shutdownNow();
      }

      int exitCode = process.exitValue();
      System.out.println("Python process exited with code: " + exitCode);
    }
  }

  public void cleanup() {
    if (executor != null) {
      executor.shutdownNow(); // Ensure executor is shut down even if an exception occurs
    }
    if (process != null) {
      process.destroyForcibly(); // Ensure the process is terminated
    }
  }
}
