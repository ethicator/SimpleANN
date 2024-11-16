#ifndef __TrainingDataStruct_h__
#define __TrainingDataStruct_h__

#include "ANNStructure.h"

struct TrainingDataStruct
{
  std::array<double, numInputs> inputs;
  std::array<double, numOutputs> outputs;
};

#endif