#pragma once

#ifndef FIX_WALL_H
#define FIX_WALL_H

#include <map>
#include <string>
#include <vector>

#include "GPUArrayGlobal.h"
#include "Fix.h"

class State;

void export_FixWall();



// at what point in the inheritance should a noncopyable attribute be added?
// i.e., where should boost::noncopyable be declared when exporting the class?
// ^TODO
class FixWall: public Fix {

public:

	// constructor
	FixWall(boost::shared_ptr<State> state_, std::string handle_, 
			std::string groupHandle_, std::string type_, 
			bool forceSingle_, bool requiresCharges_, int applyEvery_)
		: Fix(state_, handle_, groupHandle_, type_, true, false, false, 
		applyEvery_)

		{

		};
	// what would this need to do? consider..
	bool prepareForRun();

	// each derived class will need their own compute
	virtual void compute(bool) {};

	// all will have origin
	Vector origin;
	// direction the force projects	
	Vector forceDir;
	// cutoff distance of the wall force
	double dist;

	

};

#endif
