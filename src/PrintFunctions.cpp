#include "PrintFunctions.h"

void PrintNeuronValues(const std::array<RemainingInnerNodes, numOutputs> &layer)
{
  std::cout << "Neuron values = [";
  bool first = true;
  for (auto &&neuron : layer)
  {
    if (first == false)
    {
      std::cout << ", ";
    }
    std::cout << std::fixed << std::setprecision(2) << neuron.Value;
    first = false;
  }
  std::cout << "]";
  std::cout << std::defaultfloat;
}

void PrintTrainingDataStructInput(const TrainingDataStruct &t)
{
  std::cout << "Input = [";
  bool first = true;
  for (auto &&i : t.inputs)
  {
    if (first == false)
    {
      std::cout << ", ";
    }
    std::cout << std::setw(3) << std::fixed << std::setprecision(2) << i;
    first = false;
  }
  std::cout << "]";
  std::cout << std::defaultfloat;
}

void PrintTrainingDataStructOutput(const TrainingDataStruct &t)
{
  std::cout << "Output = [";
  bool first = true;
  for (auto &&i : t.outputs)
  {
    if (first == false)
    {
      std::cout << ", ";
    }
    std::cout << std::setw(3) << std::fixed << std::setprecision(2) << i;
    first = false;
  }
  std::cout << "]";
  std::cout << std::defaultfloat;
}

void PrintInputOutputAndError(
    int epochs,
    const TrainingDataStruct &t,
    const std::array<RemainingInnerNodes, numOutputs> &OutputLayer)
{
  std::cout << epochs << " ";
  PrintTrainingDataStructInput(t);
  std::cout << " ";
  PrintTrainingDataStructOutput(t);
  std::cout << " ";
  PrintNeuronValues(OutputLayer);

  std::cout << " ";
  std::cout << "Error = [";
  bool first = true;
  for (int i = 0; i < numOutputs; i++)
  {
    if (first == false)
    {
      std::cout << ", ";
    }
    std::cout << std::setw(5) << std::fixed << std::setprecision(2) << t.outputs[i] - OutputLayer[i].Value;
    first = false;
  }
  std::cout << "]\n";
  std::cout << std::defaultfloat;
}

void PrintInputAndOutputAndError(
    const TrainingDataStruct &t,
    const std::array<RemainingInnerNodes, numOutputs> &OutputLayer)
{
  PrintTrainingDataStructInput(t);
  std::cout << " ";
  PrintTrainingDataStructOutput(t);
  std::cout << " ";
  PrintNeuronValues(OutputLayer);

  std::cout << " ";
  std::cout << "Error = [";
  bool first = true;
  for (int i = 0; i < numOutputs; i++)
  {
    if (first == false)
    {
      std::cout << ", ";
    }
    std::cout << t.outputs[i] - OutputLayer[i].Value;
    first = false;
  }
  std::cout << "]\n";
}
