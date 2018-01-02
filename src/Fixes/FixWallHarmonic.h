#pragma once
#ifndef FIXWALLHARMONIC_H
#define FIXWALLHARMONIC_H

#include "FixWall.h"
#include "WallEvaluatorHarmonic.h"



void export_FixWallHarmonic();

class FixWallHarmonic : public FixWall {
	public:
		FixWallHarmonic(SHARED(State), std::string handle_, std::string groupHandle_,
							Vector origin_, Vector forceDir_, real dist_, real k_);
		// cutoff distance
        real dist;

        // spring constant
		real k;
		
        void compute(int);

		bool prepareForRun();
		
        bool postRun();

        void singlePointEng(real *);

		EvaluatorWallHarmonic evaluator; // evaluator for harmonic wall interactions

};

#endif



