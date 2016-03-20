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
using namespace std;
class DataSet {
	public:
		vector<int64_t> turns;
        //can't store data in here b/c is of different types, wouldn't be able to store all base pointers in a vector.  Maybe pass data with void pointers?
		string handle;
        bool requiresVirials;

		virtual void collect(int64_t turn){};
        int64_t nextCollectTurn;
        int collectEvery;		
        PyObject *collectGenerator;
        bool collectModeIsPython;

        void setCollectMode() {
            if (PyCallable_Check(collectGenerator)) {
                collectModeIsPython = true;
            } else {
                collectModeIsPython = false;
                assert(collectEvery > 0);
            }
        }
        void prepareForRun() {
            setCollectMode();
        }
		DataSet(){};
        //okay, so default arguments will be set in python wrapper.  In C++, will just have to send Py_None as arg for collectGenerator
        
		DataSet(string handle_, int collectEvery_, PyObject *collectGenerator_) {
            handle = handle_;
            collectEvery = collectEvery_;
            collectGenerator = collectGenerator;
            setCollectMode();
        };
};


#endif
