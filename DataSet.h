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
void export_DataSet();
class DataSet {
	public:
        std::vector<int64_t> turns;
        //can't store data in here b/c is of different types, wouldn't be able to store all base pointers in a vector.  Maybe pass data with void pointers?
		uint32_t groupTag;
        bool requiresVirials;
        bool requiresKineticEng;
        bool requiresEng;

		virtual void collect(int64_t turn, BoundsGPU &, int nAtoms, float4 *xs, float4 *vs, float4 *fs, float *engs, Virial *) = 0;
        virtual void appendValues() = 0;
        int64_t nextCollectTurn;
        int collectEvery;		
        boost::python::object collectGenerator;
        bool collectModeIsPython;

        void setCollectMode(); 
        virtual void prepareForRun(){};
		DataSet(){};
		DataSet(uint32_t groupTag_){
            groupTag = groupTag_;
            requiresVirials = false;
            requiresEng = false;
            collectEvery = -1;
        };
        int64_t getNextCollectTurn(int64_t turn);
        void setNextCollectTurn(int64_t turn);
        //okay, so default arguments will be set in python wrapper.  In C++, will just have to send Py_None as arg for collectGenerator
        bool sameGroup(uint32_t other) {
            return other == groupTag;
        }
        void takeCollectValues(int collectEvery_, boost::python::object collectGenerator_);
        boost::python::list turnsPy;
        boost::python::list valsPy;
};


#endif
