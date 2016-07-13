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
        bool requiresEng;
        
        //if not already computed for turn requested, will compute value.
        virtual void computeScalar(int64_t turn, bool transferToCPU);
        virtual void computeVector(int64_t turn, bool transferToCPU);

        bool lastScalarOnCPU;
        bool lastVectorOnCPU;

        int64_t turnScalarComputed;
        int64_t turnVectorComputed;
        
        //each data set will implement its own stored scalar and vector to be stored and appended to py list if asked by integrator

        bool computingScalar;
        bool computingVector;


        //external function to be called by integrator
		virtual void collect(int64_t turn, BoundsGPU &, int nAtoms, float4 *xs, float4 *vs, float4 *fs, float *engs, Virial *, cudaDeviceProp &) = 0;
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
            turnScalarComputed = -1;
            turnVectorComputed = -1;
        };
        int64_t getNextCollectTurn(int64_t turn);
        void setNextCollectTurn(int64_t turn);
        //okay, so default arguments will be set in python wrapper.  In C++, will just have to send Py_None as arg for collectGenerator
        bool sameGroup(uint32_t other) {
            return other == groupTag;
        }
        void takeCollectValues(int collectEvery_, boost::python::object collectGenerator_);
        boost::python::list turnsPy;
        boost::python::list scalarsPy;
        boost::python::list vectorsPy;
};


#endif
