#pragma once
#ifndef FIXDPD_T_H
#define FIXDPD_T_H

#include "FixDPD.h"

void export_FixDPD_T();

// class implementing isothermal dissipative particle dynamics
class FixDPD_T : public FixDPD {

    public:
        // here, constructors that are specific to the integrator being used
        // implement try-catch statements to ensure user implementation is 
        // correct; there is a difference if they are using the LGJF integrator
        // as opposed to Verlet integrator.
        
        // an empty constructor, because these are problems for future brian
        FixDPD_T () {};

        // additionally, we will need separate constructors for the various
        // methods by which we allow temperature inputs

        void compute(bool);

		bool prepareForRun();
		
        bool postRun();

        void singlePointEng(float *);
        
        //EvaluatorDPD_T evaluator;
}

#endif
