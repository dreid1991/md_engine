#pragma once
#ifndef FIXWALLLJ126_H
#define FIXWALLLJ126_H

#include "FixWall.h"
#include "WallEvaluatorLJ126.h"

void export_FixWallLJ126();

class FixWallLJ126 : public FixWall {

    public: 
        
		FixWallLJ126(SHARED(State), std::string handle_, std::string groupHandle_,
							Vector origin_, Vector forceDir_, real dist_, real sigma_, real epsilon_);
		real dist;

		real sigma;
		
        real epsilon;

        void compute(int);

		bool prepareForRun();
		
        bool postRun();

        void singlePointEng(real *);
		
        EvaluatorWallLJ126 evaluator; // evaluator for LJ 12-6 wall interactions
};

#endif
