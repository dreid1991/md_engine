#pragma once
#ifndef DATASET_H
#define DATASET_H
#include "Python.h"
#include <vector>
#include <map>
#include "globalDefs.h"
#include <boost/shared_ptr.hpp>
#include <iostream>
#include "boost_for_export.h"
#include "BoundsGPU.h"
#include "Virial.h"
/*
 *okay, so need to have per-group data so that we can compute all the data, sync, then use cpu-side values.

 On the other hand, we don't know in advance what data the fixes will be using.  Fixes will generally be using the data right away, so
 how about we have like group-agnostic data sets stored in data manager, then group-specific ones which are instantiated ONLY when a DataSetUser needs it.

 That sounds good.  Good night.
 *
 * /
namespace MD_ENGINE {
    class State;

    class DataSet {
    public:
        State *state;
        uint32_t groupTag;

        //if not already computed for turn requested, will compute value.
        //note that transfer flag doesn't make it sync.  Use if (not state->isSynced()) {state->sync()}
        virtual void computeScalar(bool transferToCPU) = 0;
        virtual void computeVector(bool transferToCPU) = 0;






        virtual void appendValues(boost::python::list) = 0;
        int collectEvery;		
        boost::python::object collectGenerator;
        bool collectModeIsPython;

        virtual void prepareForRun(){};
        DataSet(){};
        DataSet(State *, uint32_t groupTag_);
    };

}
#endif
