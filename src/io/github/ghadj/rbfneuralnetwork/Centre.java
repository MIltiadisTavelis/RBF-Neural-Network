package io.github.ghadj.rbfneuralnetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Implementation of a centre of the RBF Neural Network.
 * 
 * @author Georgios Hadjiantonis
 * @since 16-11-2019
 */
public class Centre {
    private List<Double> coordinates;
    private List<Double> weights = new ArrayList<>();
    private double output; // gaussian function result of last epoch
    private double sigma; // standard deviation

    /**
     * Costructos of a Centre instance. Sets inctance's attributes to the given
     * values. Initializes weights to the output neurons to random double values
     * [-1, 1].
     * 
     * @param coordinates      coordinates of the centre.
     * @param sigma            gaussian standard deviation.
     * @param numOutputNeurons number of output neurons.
     */
    public Centre(List<Double> coordinates, double sigma, int numOutputNeurons) {
        this.coordinates = coordinates;
        this.sigma = sigma;
        for (int i = 0; i < numOutputNeurons; i++)
            weights.add((new Random()).nextDouble() * 2 - 1); // [-1, 1]
    }

    /**
     * Calculates output of the gaussian function and sets output attribute to the
     * calculated value.
     * 
     * @param pattern input pattern.
     * @return result of the gaussian function.
     */
    public double gaussianFunction(List<Double> pattern) {
        output = Math.exp((-1) * euclideanDistance(pattern, coordinates) / (2 * Math.pow(sigma, 2)));
        return output;
    }

    /**
     * Calculates the euclidean distance between two vectors.
     * 
     * @param x vector.
     * @param y vector.
     * @return euclidean distance between the given vectors.
     */
    private double euclideanDistance(List<Double> x, List<Double> y) {
        double sum = 0.0;
        for (int i = 0; i < x.size() && i < y.size(); i++)
            sum += Math.pow(x.get(i) - y.get(i), 2);
        return sum;
    }

    /**
     * Updates the coordinates of the centre.
     * 
     * @param x            input pattern.
     * @param errors       error for each output, for one pattern.
     * @param learningRate learning rate.
     */
    public void updateCoordinates(List<Double> x, List<Double> errors, double learningRate) {
        double sum = 0.0;
        for (int i = 0; i < errors.size() && i < weights.size(); i++)
            sum += errors.get(i) * weights.get(i) * output;
        for (Double c : this.coordinates) {
            c = c + learningRate * sum * (x.get(coordinates.indexOf(c)) - c) / Math.pow(sigma, 2);
        }
    }

    /**
     * Updates sigma value of the centre.
     * 
     * @param x            input pattern.
     * @param errors       error for each output, for one pattern.
     * @param learningRate learning rate.
     */
    public void updateSigma(List<Double> x, List<Double> errors, double learningRate) {
        double sum = 0.0;
        for (int i = 0; i < errors.size() && i < weights.size(); i++)
            sum += errors.get(i) * weights.get(i) * output;
        this.sigma = this.sigma
                + learningRate * sum * (euclideanDistance(x, this.coordinates) / Math.pow(this.sigma, 3));
    }

    /**
     * Updates weight of the centre to the output neurons.
     * 
     * @param errors       error for each output, for one pattern.
     * @param learningRate learning rate.
     */
    public void updateWeights(List<Double> errors, double learningRate) {
        for (Double w : this.weights)
            w = w + learningRate * errors.get(this.weights.indexOf(w)) * output;
    }

    /**
     * Returns the weight vector.
     * 
     * @return weight vector.
     */
    public List<Double> getWeights() {
        return weights;
    }
}