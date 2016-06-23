#pragma once
#ifndef FIXWALLHARMONIC_TEMP_H
#define FIXWALLHARMONIC_TEMP_H

#include "FixWall.h"
#include "WallEvaluatorHarmonic.h"



void export_WallHarmonic_temp();

// look at FixLJCut.h for analogous process, with considerations for FixWallHarmonic.h's original implementation

// this is a fix for harmonic wall boundary conditions

class FixWallHarmonic_temp : public FixWall {
	public:
		FixWallHarmonic_temp(boost::shared_ptr<State>, std::string handle_, std::string groupHandle_,
							Vector origin_, Vector forceDir_, double dist_, double k);

		// assign to the base class vars
		// is this necessary / correct? check other parts of the code for analogous processes TODO
		dist = dist_;
		origin = origin_;
		forceDir = forceDir_;
	    
		void compute(bool);

//		bool prepareForRun();

//		bool postRun();

		// spring constant
		double k;
		WallHarmonicEvaluator evaluator; // evaluator for harmonic wall interactions

};

#endif



