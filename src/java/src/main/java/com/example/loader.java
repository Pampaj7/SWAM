package com.example;

import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.FileWriter;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
// import org.apache.commons.csv.Header;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.csv.QuoteMode;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.util.List;
import java.util.Map;
import java.util.ArrayList;

public class loader {
  private static void saveToCsv(Instances data, String filePath) {
    try (PrintWriter writer = new PrintWriter(new FileWriter(filePath))) {
      // Write the header
      for (int i = 0; i < data.numAttributes(); i++) {
        writer.print(data.attribute(i).name());
        if (i < data.numAttributes() - 1) {
          writer.print(",");
        }
      }
      writer.println();

      // Write the data
      for (int i = 0; i < data.numInstances(); i++) {
        for (int j = 0; j < data.numAttributes(); j++) {
          writer.print(data.instance(i).value(j));
          if (j < data.numAttributes() - 1) {
            writer.print(",");
          }
        }
        writer.println();
      }
    } catch (IOException e) {
      System.out.println("Error saving the dataset: " + e.getMessage());
    }
  }

  public static void editCsv(double time) {
    try {
      Reader reader = new FileReader("./output/emissions.csv");
      CSVParser csvParser = new CSVParser(reader, CSVFormat.DEFAULT.withFirstRecordAsHeader());
      Writer writer = new FileWriter("./output/emissions.csv");
      CSVPrinter csvPrinter = new CSVPrinter(writer,
          CSVFormat.DEFAULT.withHeader(csvParser.getHeaderMap().keySet().toArray(new String[0])));
      // Get header and records
      Map<String, Integer> headerMap = csvParser.getHeaderMap();
      List<org.apache.commons.csv.CSVRecord> records = csvParser.getRecords();

      // Write updated records
      for (org.apache.commons.csv.CSVRecord record : records) {
        // Convert record to a mutable list
        List<String> recordList = new ArrayList<>();
        for (int i = 0; i < record.size(); i++) {
          if (i == 3) {
            recordList.add(Double.toString(time));
          } else {
            recordList.add(record.get(i));
          }
        }
        csvPrinter.printRecord(recordList);
      }
      csvPrinter.flush();
    } catch (IOException e) {
      System.out.println("Error loading dataset: " + e.getMessage());
    }

  }

  public static Instances loadDataset(String datasetName) {
    String filePath = getFilePath(datasetName);
    if (filePath == null) {
      System.out.println("Dataset name not recognized.");
      return null;
    }

    CSVLoader loader = new CSVLoader();
    try {
      loader.setSource(new File(filePath));
      Instances data = loader.getDataSet();
      return data;
    } catch (IOException e) {
      System.out.println("Error loading dataset: " + e.getMessage());
      return null;
    }
  }

  private static String getFilePath(String datasetName) {
    switch (datasetName.toLowerCase()) {
      case "breast_cancer":
        return "../../datasets/breastcancer/dataset_processed/breastcancer_processed.csv";
      case "iris":
        return "../../datasets/iris/dataset_processed/iris_processed.csv";
      case "winequality":
        return "../../datasets/winequality/dataset_processed/wine_Data_processed.csv";
      default:
        return null;
    }
  }
}
