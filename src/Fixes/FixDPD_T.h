#pragma once
#ifndef FIXDPD_T_H
#define FIXDPD_T_H

#include "FixDPD.h"

void export_FixDPD_T();

// class implementing isothermal dissipative particle dynamics
class FixDPD_T : public FixDPD {

    public:
        // our friction coefficient gamma
        float gamma;

        // amplitude of the thermal noise given by sigma
        float sigma;
        // note that we will need to pass dt, timestep, and the temperature setpoint
        // to the fix compute somewhere
        // where temperature is specified by the interpolator class
        //
        // boolean updateGamma: denotes whether we update gamma when setpoint
        // temperature is changed.  True if we update gamma, false if we update sigma
        // (dependent on whether the user specifies sigma or gamma at run time)
        bool updateGamma;
        // a constructor in which the friction coefficient is specified
        // in the body of this constructor, we calculate sigma
        FixDPD_T (State* state_, std::string handle_, std::string groupHandle_, 
                  float gamma_, float rcut_, int s_) ;
        
        // a constructor in which the amplitude of the thermal fluctuations is specified
        // in the body of this constructor, we calculate gamma
        FixDPD_T (State* state_, std::string handle_, std::string groupHandle_,
                  float sigma_, float rcut_, int s_) ; 
        
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
