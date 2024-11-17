#include <array>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "ANNStructure.h"
#include "FirstInnerNode.h"
#include "RemainingInnerNodes.h"
#include "TrainingDataStruct.h"
#include "PrintFunctions.h"

// Activation function and its derivative
double sigmoid(double x) { return 1 / (1 + exp(-x)); }
double dSigmoid(double x) { return x * (1 - x); }

double init_weight() { return ((double)rand()) / ((double)RAND_MAX); }

template <typename T>
void InitialiseLayer(T &Layer)
{
  for (auto &&neuron : Layer)
  {
    for (auto &&weight : neuron.AssociatedWeights)
    {
      weight = init_weight();
    }
    neuron.Bias = init_weight();
  }
}

template <typename T, typename U>
void EstimateDelta(const int n_DownStreamLayerNodes, const T &DownstreamLayer,
                   int n_LayerToCalculateNodes, U &LayerToCalculate)
{
  for (int i = 0; i < n_LayerToCalculateNodes; i++)
  {
    double error = 0.0;
    for (int j = 0; j < n_DownStreamLayerNodes; j++)
    {
      error += DownstreamLayer[j].Delta * DownstreamLayer[j].AssociatedWeights[i];
    }
    LayerToCalculate[i].Delta = error * dSigmoid(LayerToCalculate[i].Value);
  }
}

template <typename T, typename U>
void ChangeBiasAndWeights(const int n_UpstreamLayerNodes, const U &UpstreamLayer,
                          int n_LayerToChangeNodes, T &LayerToChange, double lr)
{
  for (int i = 0; i < n_LayerToChangeNodes; i++)
  {
    LayerToChange[i].Bias += LayerToChange[i].Delta * lr;
    for (int j = 0; j < n_UpstreamLayerNodes; j++)
    {
      LayerToChange[i].AssociatedWeights[j] += UpstreamLayer[j].Value * LayerToChange[i].Delta * lr;
    }
  }
}

template <typename T, typename U>
void ForwardPassOneLayer(const int n_PreviousLayerNodes, const T &PreviousLayer,
                         U &LayerToCalculate)
{
  for (auto &&neuron : LayerToCalculate)
  {
    double activation = neuron.Bias;
    for (int j = 0; j < n_PreviousLayerNodes; j++)
    {
      activation += PreviousLayer[j].Value * neuron.AssociatedWeights[j];
    }
    neuron.Value = sigmoid(activation);
  }
}

void ForwardPassAllLayers(
    TrainingDataStruct &t,
    std::array<FirstInnerNode, numHiddenNodes> &FirstHiddenLayer,
    std::array<std::array<RemainingInnerNodes, numHiddenNodes>, numIntermediateLayers> &IntermediateHiddenLayers,
    std::array<RemainingInnerNodes, numOutputs> &OutputLayer)
{
  // Input training data into the first hidden layer
  for (auto &&neuron : FirstHiddenLayer)
  {
    double activation = neuron.Bias;
    for (int j = 0; j < numInputs; j++)
    {
      activation += t.inputs[j] * neuron.AssociatedWeights[j];
    }
    neuron.Value = sigmoid(activation);
  }

  // Propagate through all of the intermediate hidden layers and finally into
  // the output layer
  if (numIntermediateLayers < 1)
  {
    std::cerr << "Insufficient intermediate layers - aborting\n";
    exit(-1);
  }
  else if (numIntermediateLayers == 1)
  {
    ForwardPassOneLayer(numHiddenNodes, FirstHiddenLayer, IntermediateHiddenLayers[0]);
    ForwardPassOneLayer(numHiddenNodes, IntermediateHiddenLayers[0], OutputLayer);
  }
  else
  {
    ForwardPassOneLayer(numHiddenNodes, FirstHiddenLayer, IntermediateHiddenLayers[0]);
    for (int i = 1; i < numIntermediateLayers; i++)
    {
      ForwardPassOneLayer(numHiddenNodes, IntermediateHiddenLayers[i - 1], IntermediateHiddenLayers[i]);
    }
    ForwardPassOneLayer(numHiddenNodes, IntermediateHiddenLayers[numIntermediateLayers - 1], OutputLayer);
  }
}

template <typename T>
void PrintNeuronWeightsAndBias(T &neuron)
{
  std::cout << "W ";
  for (auto &&weight : neuron.AssociatedWeights)
  {
    std::cout << weight << " ";
  }
  std::cout << "\n";
  std::cout << "B " << neuron.Bias << "\n";
}

int main(void)
{
  const double lr = 0.25;

  const int numberOfEpochs = 10 * 1000 + 1;
  const int IntervalsBetweenEpochsToPrintOutputs = 100;

  std::array<FirstInnerNode, numHiddenNodes> FirstHiddenLayer; // This interfaces with the inputs
  std::array<std::array<RemainingInnerNodes, numHiddenNodes>, numIntermediateLayers> IntermediateHiddenLayers;
  std::array<RemainingInnerNodes, numOutputs> OutputLayer; // This interfaces with output neurons

  std::vector<TrainingDataStruct> TrainingData;

  // Training data consists of:
  // - numInputs of inputs
  // - numOutputs of outputs

  // The inputs can be an integer or real number. The output is a real number
  // between 0 and 1. Think of the output (or outputs) as being a representation
  // of the actual output you're wanting.

  // e.g. for a simple XOR function that only produces either 0 or 1, the output
  // can just be a single output between 0-1. i.e. {0} represents 0 and {1}
  // represents 1.

  // For a case where you're adding two numbers together, you'll need some type
  // of encoding scheme. In the example given below, the output {0,0,0,0}
  // represents the output of 0, {0,1,0,0} is 1, {0,0,1,0} is 2 and {0,0,0,1}
  // is 3.

  TrainingData.push_back({{0, 0}, {0, 0, 0, 0}});
  TrainingData.push_back({{1, 0}, {0, 1, 0, 0}});
  TrainingData.push_back({{0, 1}, {0, 1, 0, 0}});
  TrainingData.push_back({{0, 2}, {0, 0, 1, 0}});
  TrainingData.push_back({{1, 1}, {0, 0, 1, 0}});
  TrainingData.push_back({{2, 0}, {0, 0, 1, 0}});
  TrainingData.push_back({{0, 3}, {0, 0, 0, 1}});
  TrainingData.push_back({{1, 2}, {0, 0, 0, 1}});
  TrainingData.push_back({{2, 1}, {0, 0, 0, 1}});
  TrainingData.push_back({{3, 0}, {0, 0, 0, 1}});

  // Let's initialise the weights and biases of all layers
  InitialiseLayer(FirstHiddenLayer);
  for (auto &&Layer : IntermediateHiddenLayers)
  {
    InitialiseLayer(Layer);
  }
  InitialiseLayer(OutputLayer);

  // Let's train the neural network for the number of epochs
  for (int epochs = 0; epochs < numberOfEpochs; epochs++)
  {
    for (auto &&t : TrainingData)
    {
      // Let's take an input (in this case t is from the training data) and
      // let's calculate the values for each node in each layer
      ForwardPassAllLayers(t, FirstHiddenLayer, IntermediateHiddenLayers, OutputLayer);

      // Print the results from forward pass
      if (epochs % IntervalsBetweenEpochsToPrintOutputs == 0)
      {
        PrintInputOutputAndError(epochs, t, OutputLayer);
      }

      // Back propagation

      // First let's estimate the error (i.e. the difference between what we
      // expect from the training data to what the ANN has predicted)
      for (int i = 0; i < numOutputs; i++)
      {
        double errorOutput = t.outputs[i] - OutputLayer[i].Value;
        OutputLayer[i].Delta = errorOutput * dSigmoid(OutputLayer[i].Value);
      }

      // Let's now estimate the change needed for each of the preceding layers
      if (numIntermediateLayers < 1)
      {
        std::cout << "Insufficient intermediate layers";
        exit(-1);
      }
      else if (numIntermediateLayers == 1)
      {
        EstimateDelta(numOutputs, OutputLayer, numHiddenNodes, IntermediateHiddenLayers[0]);
        EstimateDelta(numHiddenNodes, IntermediateHiddenLayers[0], numHiddenNodes, FirstHiddenLayer);
      }
      else
      {
        EstimateDelta(numOutputs, OutputLayer, numHiddenNodes, IntermediateHiddenLayers[numIntermediateLayers - 1]);
        for (int i = (numIntermediateLayers - 1); i > 0; i--)
        {
          EstimateDelta(numHiddenNodes, IntermediateHiddenLayers[i], numHiddenNodes, IntermediateHiddenLayers[i - 1]);
        }
        EstimateDelta(numHiddenNodes, IntermediateHiddenLayers[0], numHiddenNodes, FirstHiddenLayer);
      }

      // Let's now change the weights and the biases of all layers
      if (numIntermediateLayers < 1)
      {
        std::cerr << "Insufficient intermediate layers - aborting\n";
        exit(-1);
      }
      else if (numIntermediateLayers == 1)
      {
        ChangeBiasAndWeights(numHiddenNodes, IntermediateHiddenLayers[0], numOutputs, OutputLayer, lr);
        ChangeBiasAndWeights(numHiddenNodes, FirstHiddenLayer, numHiddenNodes, IntermediateHiddenLayers[0], lr);
      }
      else
      {
        ChangeBiasAndWeights(numHiddenNodes, IntermediateHiddenLayers[numIntermediateLayers - 1], numOutputs, OutputLayer, lr);
        for (int i = (numIntermediateLayers - 1); i > 0; i--)
        {
          ChangeBiasAndWeights(numHiddenNodes, IntermediateHiddenLayers[i - 1], numHiddenNodes, IntermediateHiddenLayers[i], lr);
        }
        ChangeBiasAndWeights(numHiddenNodes, FirstHiddenLayer, numHiddenNodes, IntermediateHiddenLayers[0], lr);
      }

      for (int i = 0; i < numHiddenNodes; i++)
      {
        FirstHiddenLayer[i].Bias += FirstHiddenLayer[i].Delta * lr;
        for (int j = 0; j < numInputs; j++)
        {
          FirstHiddenLayer[i].AssociatedWeights[j] += t.inputs[j] * FirstHiddenLayer[i].Delta * lr;
        }
      }
      // At this point we've completed back propagation. We'll now continue
      // training by going back to the for loop.
    }
  }

  // We've now trained the ANN. Let's do some tests with data it's not seen
  // before

  TrainingDataStruct test = {{1.5, 1.5}, {0, 0, 0, 0}};
  ForwardPassAllLayers(test, FirstHiddenLayer, IntermediateHiddenLayers, OutputLayer);
  PrintTrainingDataStructInput(test);
  std::cout << " ";
  PrintNeuronValues(OutputLayer);
  std::cout << "\n";

  test = {{0.1, 0.9}, {0, 0, 0, 0}};
  ForwardPassAllLayers(test, FirstHiddenLayer, IntermediateHiddenLayers, OutputLayer);
  PrintTrainingDataStructInput(test);
  std::cout << " ";
  PrintNeuronValues(OutputLayer);
  std::cout << "\n";

  test = {{0.6, 1.4}, {0, 0, 0, 0}};
  ForwardPassAllLayers(test, FirstHiddenLayer, IntermediateHiddenLayers, OutputLayer);
  PrintTrainingDataStructInput(test);
  std::cout << " ";
  PrintNeuronValues(OutputLayer);
  std::cout << "\n";

  return 0;
}