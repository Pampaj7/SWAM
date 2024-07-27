package com.example.LogisticRegressionJava;

import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.SimpleBounds;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;

public class LogisticRegression {
    private RealVector coefficients;

    public LogisticRegression(RealMatrix X, RealVector y) {
        int n = X.getColumnDimension();
        BOBYQAOptimizer optimizer = new BOBYQAOptimizer(2 * n + 1);
        ObjectiveFunction objectiveFunction = new ObjectiveFunction(point -> logLikelihood(point, X, y));
        InitialGuess initialGuess = new InitialGuess(new double[n]);
        SimpleBounds bounds = new SimpleBounds(new double[n], new double[n]);
        PointValuePair optimum = optimizer.optimize(objectiveFunction, GoalType.MAXIMIZE, initialGuess, bounds);
        coefficients = new ArrayRealVector(optimum.getPoint());
    }

    private double logLikelihood(double[] betas, RealMatrix X, RealVector y) {
        RealVector betaVector = new ArrayRealVector(betas);
        RealVector XBeta = X.operate(betaVector);
        double logLikelihood = 0.0;
        for (int i = 0; i < X.getRowDimension(); i++) {
            double p = 1 / (1 + Math.exp(-XBeta.getEntry(i)));
            logLikelihood += y.getEntry(i) * Math.log(p) + (1 - y.getEntry(i)) * Math.log(1 - p);
        }
        return logLikelihood;
    }

    public double predict(RealVector x) {
        double linearCombination = coefficients.dotProduct(x);
        return 1 / (1 + Math.exp(-linearCombination));
    }

    public static void main(String[] args) {
        // Example data
        double[][] data = {
                { 1.0, 2.0 },
                { 2.0, 3.0 },
                { 3.0, 4.0 },
                { 4.0, 5.0 }
        };

        double[] target = { 0, 0, 1, 1 };

        RealMatrix X = MatrixUtils.createRealMatrix(data);
        RealVector y = new ArrayRealVector(target);

        LogisticRegression logisticRegression = new LogisticRegression(X, y);

        RealVector newPoint = new ArrayRealVector(new double[] { 1.5, 2.5 });
        double prediction = logisticRegression.predict(newPoint);

        System.out.println("Prediction: " + prediction);
    }
}
