#pragma once

#define DEFAULT_FILL -1000

#include <climits>
#include <map>
#include <string>
#include <vector>
#include <iostream>

#include "AtomParams.h"
#include "GPUArrayGlobal.h"
#include "Fix.h"
#include "xml_func.h"
#include "SquareVector.h"
#include "BoundsGPU.h"
class EvaluatorWrapper;
void export_FixPair();

class State;

class FixThreeBody : public Fix {
    FixThreeBody(boost::shared_ptr<State>(State) state_, std::string handle_, std::string groupHandle_,
            std::string type_, bool forceSingle_, bool requiresCharges_, int applyEvery_)
        : Fix(state_, handle_, groupHandle_, type_, forceSingle_, false, requiresCharges_, applyEvery_), chargeCalcFix(nullptr)
        {
            // Empty constructor
        };
    //okay so options for this include
    //generalize FixPair to 
    //FixNBody which FixPair inherits and FixThreeBody inherits as well
    //
    //but basically the functionality that needs to be here is that for setting parameters, copying these parameters to linear memory and then to the gpu
    //and also that for handling the evaluators 
    //
    //I suggest we just copy the functionality for right now, then see what commonality there ends up being and build base classes from there
