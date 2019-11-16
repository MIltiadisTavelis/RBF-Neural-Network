package io.github.ghadj.rbfneuralnetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Centre {
    private List<Double> coordinates;
    private List<Double> weights = new ArrayList<>();
    private double output; // gaussian function
    private double sigma; // standard deviation

    public Centre(List<Double> coordinates, double sigma, int numOutputNeurons) {
        this.coordinates = coordinates;
        this.sigma = sigma;
        for (int i = 0; i < numOutputNeurons; i++)
            weights.add((new Random()).nextDouble() * 2 - 1); // [-1, 1]
    }

    public double gaussianFunction(List<Double> pattern) {
        output = Math.exp((-1) * euclideanDistance(pattern, coordinates) / (2 * Math.pow(sigma, 2)));
        return output;
    }

    private double euclideanDistance(List<Double> x, List<Double> y) {
        double sum = 0.0;
        for (int i = 0; i < x.size() && i < y.size(); i++)
            sum += Math.pow(x.get(i) - y.get(i), 2);
        return sum;
    }

    public void updateCoordinates(List<Double> x, List<Double> errors, double learningRate) {
        double sum = 0.0;
        for (int i = 0; i < errors.size() && i < weights.size(); i++)
            sum += errors.get(i) * weights.get(i) * output;
        for (Double c : this.coordinates) {
            c = c + learningRate * sum * (x.get(coordinates.indexOf(c)) - c) / Math.pow(sigma, 2);
        }
    }

    public void updateSigma(List<Double> x, List<Double> errors, double learningRate) {
        double sum = 0.0;
        for (int i = 0; i < errors.size() && i < weights.size(); i++)
            sum += errors.get(i) * weights.get(i) * output;
        this.sigma = this.sigma
                + learningRate * sum * (euclideanDistance(x, this.coordinates) / Math.pow(this.sigma, 3));
    }

    public void updateWeights(List<Double> errors, double learningRate) {
        for (Double w : this.weights)
            w = w + learningRate * errors.get(this.weights.indexOf(w)) * output;
    }

    public List<Double> getCoordinates() {
        return coordinates;
    }

    public double getSigma() {
        return sigma;
    }

    public List<Double> getWeights() {
        return weights;
    }
}