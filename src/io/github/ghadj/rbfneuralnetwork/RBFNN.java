package io.github.ghadj.rbfneuralnetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Map.Entry;

public class RBFNN {
    private List<Double> bias = new ArrayList<>();
    private List<Centre> centres = new ArrayList<>();
    private int numOutputNeurons;
    private double learningRateCenter;
    private double learningRateSigma;
    private double learningRateWeight;
    private int maxIterations;
    private List<Double> trainErrorList = new ArrayList<Double>();
    private List<Double> testErrorList = new ArrayList<Double>();

    public RBFNN(int numHiddenLayerNeurons, int numInputNeurons, int numOutputNeurons, double learningRate,
            double sigma, int maxIterations, List<List<Double>> centreVectors) {
        // initialize bias weights to the output neurons
        for (int i = 0; i < numOutputNeurons; i++)
            bias.add((new Random()).nextDouble() * 2 - 1); // [-1, 1]

        for (int i = 0; i < numHiddenLayerNeurons; i++)
            this.centres.add(new Centre(centreVectors.get(i), sigma, numOutputNeurons));

        this.numOutputNeurons = numOutputNeurons;
        this.learningRateCenter = learningRate;
        this.learningRateSigma = learningRate;
        this.learningRateWeight = learningRate;
        this.maxIterations = maxIterations;
    }

    public void run(Map<List<Double>, List<Double>> trainingData, Map<List<Double>, List<Double>> testingData) {
        for (int i = 0; i < maxIterations; i++) {
            run(trainingData, true);
            run(testingData, false);
        }
    }

    private void run(Map<List<Double>, List<Double>> data, boolean training) {
        double sumError = 0.0, error;
        List<Double> errors;
        for (Entry<List<Double>, List<Double>> e : data.entrySet()) { // for each pattern
            errors = new ArrayList<>();

            for (int outputNeuron = 0; outputNeuron < numOutputNeurons; outputNeuron++) {
                double actual = this.bias.get(outputNeuron);
                for (Centre c : centres) {
                    actual += c.getWeights().get(outputNeuron) * c.gaussianFunction(e.getKey());
                }
                double target = e.getValue().get(outputNeuron);
                error = target - actual;

                errors.add(error);
                sumError += Math.pow(error, 2);
            }

            if (training)
                for (Centre c : centres) {
                    c.updateCoordinates(e.getKey(), errors, learningRateCenter);
                    c.updateSigma(e.getKey(), errors, learningRateSigma);
                    c.updateWeights(errors, learningRateWeight);
                }
            this.updateBias(errors);
        }
        if (training)
            trainErrorList.add(0.5 * sumError);
        else
            testErrorList.add(0.5 * sumError);
    }

    private void updateBias(List<Double> errors) {
        for (Double b : bias)
            b = b + learningRateWeight * errors.get(bias.indexOf(b));
    }

    public List<List<Double>> getWeights() {
        List<List<Double>> weights = new ArrayList<>();
        for (int i = 0; i < numOutputNeurons; i++) {
            List<Double> w = new ArrayList<>();
            for (Centre c : centres)
                w.add(c.getWeights().get(i));
            weights.add(w);
        }
        return weights;
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