package com.example;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class main {
  public static void main(String[] args) {
    // Create a SparkSession
    SparkSession spark = SparkSession.builder()
        .appName("BreastCancerRandomForest")
        .master("local[*]")
        .getOrCreate();
    spark.sparkContext().setLogLevel("ERROR");

    // Load the CSV data
    Dataset<Row> data = spark.read()
        .option("header", "true")
        .option("inferSchema", "true")
        .csv("../../../datasets/breastcancer/breastcancer.csv");

    // Define feature columns
    String[] featureColumns = {
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean",
        "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
        "smoothness_se", "compactness_se", "concavity_se", "concave_points_se",
        "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
        "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
        "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
    };

    // Split the data into training and test sets
    Dataset<Row>[] splits = data.randomSplit(new double[] { 0.7, 0.3 });
    Dataset<Row> trainingData = splits[0];
    Dataset<Row> testData = splits[1];

    // Index labels, adding metadata to the label column
    StringIndexer labelIndexer = new StringIndexer()
        .setInputCol("diagnosis")
        .setOutputCol("indexedLabel");

    // Vector Assembler
    VectorAssembler assembler = new VectorAssembler()
        .setInputCols(featureColumns)
        .setOutputCol("features");

    // Train a RandomForest model
    RandomForestClassifier rf = new RandomForestClassifier()
        .setLabelCol("indexedLabel")
        .setFeaturesCol("features")
        .setNumTrees(100);

    // Convert indexed labels back to original labels
    IndexToString labelConverter = new IndexToString()
        .setInputCol("prediction")
        .setOutputCol("predictedLabel")
        .setLabels(labelIndexer.fit(data).labels());

    // Chain indexers and forest in a Pipeline
    Pipeline pipeline = new Pipeline()
        .setStages(new PipelineStage[] { labelIndexer, assembler, rf, labelConverter });

    // Train model
    PipelineModel model = pipeline.fit(trainingData);

    // Make predictions
    Dataset<Row> predictions = model.transform(testData);

    // Evaluate model
    MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("indexedLabel")
        .setPredictionCol("prediction")
        .setMetricName("accuracy");
    double accuracy = evaluator.evaluate(predictions);

    // Print only the accuracy
    System.out.println("Test Accuracy = " + accuracy);

    // Stop the SparkSession
    spark.stop();
  }
}
