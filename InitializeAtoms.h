#ifndef INITIALIZE_H
#define INTIIALIZE_H
#include "Python.h"
#include <math.h>
#include <vector>
#include <map>
#include <random>
#include "State.h"
#include "boost_for_export.h"
void export_InitializeAtoms();
using namespace std;


class InitializeAtomsPythonWrap {};

namespace InitializeAtoms {
	extern default_random_engine generator;
	void populateOnGrid(SHARED(State) state, Bounds &bounds, string handle, int n);

	void populateRand(SHARED(State) state, Bounds &bounds, string handle, int n, num distMin);

	void initTemp(SHARED(State) state, string groupHandle, num temp);
}

#endif
