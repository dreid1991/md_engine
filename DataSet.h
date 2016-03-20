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
class State;
using namespace std;
void export_DataSet();
class DataSet {
	public:
		State *state;
		vector<int> turns;
		vector<num> data;
		int turnInit;
		string handle;
		virtual void process(int turn){};
		num accumulator;
		int accumulateEvery;
		int computeEvery;
		PyObject *pyProcess;
		
		DataSet(){};
		DataSet(State *state_, string handle_, int accumulateEvery_, int computeEvery_);
};


#endif
