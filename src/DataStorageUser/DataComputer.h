#pragma once
#ifndef DATACOMPUTER_H
#define DATACOMPUTER_H
#include "Python.h"
#include <vector>
#include <map>
#include "globalDefs.h"
#include <boost/shared_ptr.hpp>
#include <boost/python.hpp>
#include <iostream>
#include "boost_for_export.h"
#include "BoundsGPU.h"
#include "Virial.h"

class State;
namespace MD_ENGINE {

    class DataComputer {
    public:
        State *state;
        uint32_t lastGroupTag;
        //note that transfer flag doesn't make it sync.  

        //used for computers which are shared between multiple fixes.  Only need to have one computer per sim
        virtual void computeScalar_GPU(bool transferToCPU, uint32_t groupTag) = 0;
        virtual void computeTensor_GPU(bool transferToCPU, uint32_t groupTag) = 0;

        //after values transferred, can compute final value on host and store interally.  Retrieval implemented by each compute
        virtual void computeScalar_CPU() = 0;
        virtual void computeTensor_CPU() = 0;






        virtual void appendScalar(boost::python::list &) = 0;
        virtual void appendTensor(boost::python::list &) = 0;

        bool computingScalar; //will determine what memory is allocated.  fixes need to tell their computers what they need in prepareforrun
        bool computingTensor;
        virtual void prepareForRun(){};
        DataComputer(){};
        DataComputer(State *, bool, bool);
    };

}
#endif
