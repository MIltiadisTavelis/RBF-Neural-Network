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
        output = 0.0;
        return 0.0;
    }

    public double euclideanDistance() {
        return 0.0;
    }

    public void updateCoordinates(List<Double> errors, double learningRate) {

    }

    public void updateSigma(List<Double> errors, double learningRate) {

    }

    public void updateWeights(List<Double> errors, double learningRate) {

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