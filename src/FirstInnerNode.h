#ifndef __FirstInnerNode_h__
#define __FirstInnerNode_h__

#include "ANNStructure.h"

struct FirstInnerNode
{
  double Value = 0.0;
  double Bias = 0.0;
  double Delta = 0.0;
  std::array<double, numInputs> AssociatedWeights;
};

#endif