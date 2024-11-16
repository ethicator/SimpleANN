#ifndef __RemainingInnerNode_h__
#define __RemainingInnerNode_h__

#include "ANNStructure.h"

struct RemainingInnerNodes
{
  double Value = 0.0;
  double Bias = 0.0;
  double Delta = 0.0;
  std::array<double, numHiddenNodes> AssociatedWeights;
};

#endif