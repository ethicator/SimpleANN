#ifndef __PrintFunctions_h__
#define __PrintFunctions_h__

#include <iostream>
#include <iomanip>
#include <array>
#include "RemainingInnerNodes.h"
#include "TrainingDataStruct.h"

void PrintNeuronValues(const std::array<RemainingInnerNodes, numOutputs> &layer);
void PrintTrainingDataStructInput(const TrainingDataStruct &t);
void PrintTrainingDataStructOutput(const TrainingDataStruct &t);
void PrintInputOutputAndError(
    int epochs,
    const TrainingDataStruct &t,
    const std::array<RemainingInnerNodes, numOutputs> &OutputLayer);
void PrintInputAndOutputAndError(
    const TrainingDataStruct &t,
    const std::array<RemainingInnerNodes, numOutputs> &OutputLayer);

#endif