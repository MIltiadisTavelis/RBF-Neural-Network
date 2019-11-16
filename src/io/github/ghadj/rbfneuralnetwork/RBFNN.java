package io.github.ghadj.rbfneuralnetwork;

import java.util.*;

public class RBFNN {
    private double biasWeight;
    private List<Centre> centres;
    private List<Double> centresWeights;
    private List<Double> trainErrorList = new ArrayList<Double>();
    private List<Double> testErrorList = new ArrayList<Double>();

    public RBFNN(int numHiddenLayerNeurons, int numInputNeurons, int numOutputNeurons, double learningRate,
            double sigmas, int maxIterations, List<List<Double>> centreVectors) {

    }

    public void run(Map<List<Double>, Double> trainingData, Map<List<Double>, Double> testingData) {

    }

    public double getOutput() {
        return 0.0;
    }

    public List<Double> getWeights() {
        return centresWeights;
    }

    /**
     * Returns a list containing the mean of the squared error per epoch, during
     * training.
     * 
     * @return list containing the mean of the squared error per epoch.
     */
    public List<Double> getTrainErrorList() {
        return trainErrorList;
    }

    /**
     * Returns a list containing the mean of the squared error per epoch, during
     * test.
     * 
     * @return list containing the mean of the squared error per epoch.
     */
    public List<Double> getTestErrorList() {
        return testErrorList;
    }
}