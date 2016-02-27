#ifndef ATOM_PARAMS_H
#define ATOM_PARAMS_H
#include "Python.h"
#include <math.h>
#include <iostream>
#include <boost/shared_ptr.hpp>
#include <assert.h>

//herm, would like to allow for different force fields
//
//
//id maps to index of data in mass, sigmas, epsilons
//
//this class does not hold per-atom info
#include "boost_for_export.h"
void export_AtomParams();
using namespace boost;
using namespace std;
class State;

class AtomParams {
	public:

        State *state;
		AtomParams() : numTypes(0) {};
		AtomParams(State *s) : state(s), numTypes(0) {};
		vector<string> handles;
		vector<double> masses;
        vector<double> atomicNums; //for xyz in vmd, etc
        int typeFromHandle(string handle);
		int numTypes;


		int addSpecies(string handle, double mass, double atomicNum=6);
		void clear();
};



#endif
