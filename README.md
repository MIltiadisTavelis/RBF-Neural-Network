# RBF-Neural-Network
Implementation of Radial Basis Function (RBF) Neural Network, with moving centers, in java.

## Objectives
* help me to internalize the mathematical description of the algorithm.
* understand the algorithm intimately and discover parameter configurations.
* how the parameters of the algorithm influence its performance. 
* experiment with various datasets and see the behaviour of the algorithm.
* track performance of the algorithm-implementation with different metrics.
* light preprocessing of dataset.
* explore opportunities to make the implementation more efficient.

## Implementation
The following  parameters of the network can be set by a  text file, its path given as command line argument:
* number of centres - hidden neurons
* number of input neurons
* number of output neurons
* learning rate
* standard deviation (sigma)
* number of iterations
* initial coordinates of the centres

## Compile & Run
```
javac -d ./bin ./src/io/github/ghadj/rbfneuralnetwork/*.java

java -cp ./bin io.github.ghadj.rbfneuralnetwork.RBFNNDriver <path to parameters' file>
```
