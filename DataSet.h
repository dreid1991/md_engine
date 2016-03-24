#ifndef DATASET_H
#define DATASET_H
#include "Python.h"
#include <vector>
#include <map>
#include <string>
#include "globalDefs.h"
#include <boost/shared_ptr.hpp>
#include <iostream>
#include "boost_for_export.h"
#include "BoundsGPU.h"
#include "Virial.h"
using namespace std;
class DataSet {
	public:
		vector<int64_t> turns;
        //can't store data in here b/c is of different types, wouldn't be able to store all base pointers in a vector.  Maybe pass data with void pointers?
		uint32_t groupTag;
        bool requiresVirials;
        bool requiresKineticEng;
        bool requiresEng;

		virtual void collect(int64_t turn, BoundsGPU &, int nAtoms, float4 *vs, float *engs, Virial *){};
        int64_t nextCollectTurn;
        int collectEvery;		
        PyObject *collectGenerator;
        bool collectModeIsPython;

        void setCollectMode(); 
        void prepareForRun();
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
        void takeCollectValues(int collectEvery_, PyObject *collectGenerator_);
        /*
		DataSet(uint32_t groupTag_, int collectEvery_, PyObject *collectGenerator_) {
            groupTag = groupTag_;
            collectEvery = collectEvery_;
            collectGenerator = collectGenerator;
            setCollectMode();
            requiresVirials = false;
            requiresEng = false;
        };
        */
};


#endif
