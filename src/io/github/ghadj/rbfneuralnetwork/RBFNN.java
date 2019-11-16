package io.github.ghadj.rbfneuralnetwork;

import java.util.*;

public class RBFNN {
    private List<Centre> centres;
    private List<Double> trainErrorList = new ArrayList<Double>();
    private List<Double> testErrorList = new ArrayList<Double>();

    public RBFNN(int numHiddenLayerNeurons, int numInputNeurons, int numOutputNeurons, double learningRate,
            double sigmas, int maxIterations) {

    }

    public void run(Map<List<Double>, Double> trainingData, Map<List<Double>, Double> testingData){

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