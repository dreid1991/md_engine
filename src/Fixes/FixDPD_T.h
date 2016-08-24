#pragma once
#ifndef FIXDPD_T_H
#define FIXDPD_T_H

#include "FixDPD.h"

void export_FixDPD_T();

// class implementing isothermal dissipative particle dynamics
class FixDPD_T : public FixDPD {

    public:
        // initialize the friction coefficient to some impractically large default value
        float gamma = std::numeric_limits<float>::max();

        // initialize the random force coefficient to some impractically large default value
        float sigma = std::numeric_limits<float>::max();
        float dt = std::numeric_limits<float>::max();
        
        // a constructor in which the friction coefficient is specified
        FixDPD_T (State* state_, float gamma_, float rcut_, int s_) ;
        
        // a constructor in which the amplitude of the thermal fluctuations is specified
        FixDPD_T (State* state_, float sigma_, float rcut_, int s_) ; 
        
        // our destructor
        ~FixDPD_T () {};

        // we compute the random and dissipative forces in compute
        void compute(bool);

        // we update the dissipative forces in stepFinal
        bool stepFinal();
        
        // hand some stuff to the evaluator (whatever it might need)
		bool prepareForRun();
		
        // here we might delete stuff?
        bool postRun();

        void singlePointEng(float *);
        
        EvaluatorDPD_T evaluator;
}

#endif
