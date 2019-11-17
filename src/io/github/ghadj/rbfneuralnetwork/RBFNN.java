package io.github.ghadj.rbfneuralnetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Map.Entry;

/**
 * Implementation of an RBF Neural Network.
 * 
 * @author Georgios Hadjiantonis
 * @since 16-11-2019
 */
public class RBFNN {
    private List<Double> bias = new ArrayList<>();
    private List<Centre> centres = new ArrayList<>();
    private int numOutputNeurons;
    private double learningRateCentre;
    private double learningRateSigma;
    private double learningRateWeight;
    private int maxIterations;
    private List<Double> trainErrorList = new ArrayList<Double>();
    private List<Double> testErrorList = new ArrayList<Double>();

    /**
     * Constructor of an RBF NN instance. Creates centres, bias weights, and set
     * attributes of the instance according to the parameters given. Weights are
     * initialized to random double values [-1, 1]. Learning rate of centres,
     * weights and sigma is set to the same values.
     * 
     * @param numHiddenLayerNeurons number of centres.
     * @param numInputNeurons       number of input neurons (data dimension).
     * @param numOutputNeurons      number of output neurons.
     * @param learningRate          learning rate.
     * @param sigma                 gaussian standard deviation.
     * @param maxIterations         iterations of the training.
     * @param centreVectors         coordinates of each centre.
     */
    public RBFNN(int numHiddenLayerNeurons, int numInputNeurons, int numOutputNeurons, double learningRate,
            double sigma, int maxIterations, List<List<Double>> centreVectors) {
        // initialize bias weights to the output neurons
        for (int i = 0; i < numOutputNeurons; i++)
            bias.add((new Random()).nextDouble() * 2 - 1); // [-1, 1]

        for (int i = 0; i < numHiddenLayerNeurons; i++)
            this.centres.add(new Centre(centreVectors.get(i), sigma, numOutputNeurons));

        this.numOutputNeurons = numOutputNeurons;
        this.learningRateCentre = learningRate;
        this.learningRateSigma = learningRate;
        this.learningRateWeight = learningRate;
        this.maxIterations = maxIterations;
    }

    /**
     * Runs the NN the iterations defined by the maxIterations attribute. Each
     * iteration includes training and testing.
     * 
     * @param trainingData training data.
     * @param testingData  testing data.
     */
    public void run(Map<List<Double>, List<Double>> trainingData, Map<List<Double>, List<Double>> testingData) {
        for (int i = 0; i < this.maxIterations; i++) {
            run(trainingData, true);
            run(testingData, false);
        }
    }

    /**
     * Runs one iterantion of training/testing. After each pattern the centres,
     * weights and sigmas of each centre are updated, in case of training.
     * 
     * @param data     input patterns.
     * @param training true to update network's attributes(centres, weights,
     *                 sigmas), otherwise false.
     */
    private void run(Map<List<Double>, List<Double>> data, boolean training) {
        double sumError = 0.0, error;
        List<Double> errors; // error for each output neuron, for one pattern
        for (Entry<List<Double>, List<Double>> e : data.entrySet()) { // for each pattern
            errors = new ArrayList<>();

            for (int o = 0; o < numOutputNeurons; o++) {
                double actual = this.bias.get(o);
                for (Centre c : centres) {
                    actual += c.getWeights().get(o) * c.gaussianFunction(e.getKey());
                }
                double target = e.getValue().get(o);
                error = target - actual;

                errors.add(error);
                sumError += Math.pow(error, 2);
            }

            if (training) {
                for (Centre c : centres) {
                    c.updateCoordinates(e.getKey(), errors, learningRateCentre);
                    c.updateSigma(e.getKey(), errors, learningRateSigma);
                    c.updateWeights(errors, learningRateWeight);
                }
                this.updateBias(errors);
            }
        }
        if (training)
            trainErrorList.add(0.5 * sumError);
        else
            testErrorList.add(0.5 * sumError);
    }

    /**
     * Updates bias weights to the output neurons.
     * 
     * @param errors for each output neuron, for one pattern.
     */
    private void updateBias(List<Double> errors) {
        for (Double b : bias)
            b = b + learningRateWeight * errors.get(bias.indexOf(b));
    }

    /**
     * Returns a list of the weights from bias unit and centre to each output
     * neuron.
     * 
     * @return a list of the weights from bias unit and centre to each output
     *         neuron.
     */
    public List<List<Double>> getWeights() {
        List<List<Double>> weights = new ArrayList<>();
        weights.add(bias);
        for (int i = 0; i < numOutputNeurons; i++) {
            List<Double> w = new ArrayList<>();
            for (Centre c : centres)
                w.add(c.getWeights().get(i));
            weights.add(w);
        }
        return weights;
    }

    /**
     * Returns a list containing the total error per epoch, during training.
     * 
     * @return list containing the total error per epoch.
     */
    public List<Double> getTrainErrorList() {
        return trainErrorList;
    }

    /**
     * Returns a list containing the total error per epoch, during test.
     * 
     * @return list containing the total error per epoch.
     */
    public List<Double> getTestErrorList() {
        return testErrorList;
    }
}