# SimpleANN
A simple artificial neural network (ANN) written in C++. This project has very 
minimal dependencies. The code shows how to undertake forward and backward 
propagation.

The const global variables in src/ANNStructure.h are used to define the structure
of the ANN:
- numInputs is the number of inputs
- numHiddenNodes is the number of nodes per hidden layer
- numIntermediateLayers is the number of hidden layers. The source code has two
  additional layers defined as FirstHiddenLayer which interfaces with the inputs
  and OutputLayer which interfaces with the outputs. So the total number of hidden
  layers is numIntermediateLayers + 2
- numOutputs is the number of outputs

Each node is a struct that contains:
- Value (e.g. the activated value of the node)
- Bias
- Delta (or used for backpropagation)
- An array of weight that are associated with the node (i.e. which connect to the
  previous layer)

There are two const variables in src/main.cpp which are used to control how
training is undertaken:
- lr which is the learning rate (should be between 0 and 1)
- epochs which is the number of iterations that a given set of training data is
  propagated through the ANN

The code uses the sigmoid function to convert an input into a node into it's
activated value (between 0 and 1).

The code uses mostly compile time variables amd data stored in std:array. The 
intent is to maximise speed (data hopefully stored in the stack) and to avoid 
any dynamic memory allocations during run time.

Command line to compile in Linux:

g++ main.cpp PrintFunctions.cpp -pedantic -Wall -Werror -O3

The intent of this code is to provide a simple example of how to structure, develop, 
structure and train a simple ANN. It could be used as a building block for a
recurrent neural network (possibly the next step).