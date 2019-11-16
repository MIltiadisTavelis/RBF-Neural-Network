package io.github.ghadj.rbfneuralnetwork;

import java.util.List;
import java.util.Random;

public class Centre {
    private List<Double> coordinates;
    private double sigma; // standard deviation
    private double weight;

    public Centre(List<Double> coordinates, double sigma){
        this.coordinates = coordinates;
        this.sigma = sigma;
        this.weight = (new Random()).nextDouble() * 2 - 1; // [-1, 1]
    }

    public double gaussianFunction(){
        return 0.0;
    }

    public double euclideanDistance(){
        return 0.0;
    }

    public void updateCoordinates(){

    }

    public void updateStandardDeviation(){
        
    }

    public double getWeight(){
        return weight;
    }

    public List<Double> getCoordinates(){
        return coordinates;
    }

    public double getSigma(){
        return sigma;
    }
}