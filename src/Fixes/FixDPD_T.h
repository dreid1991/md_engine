#pragma once
#ifndef FIXDPD_T_H
#define FIXDPD_T_H

#include "FixDPD.h"

void export_FixDPD_T();

// class implementing isothermal dissipative particle dynamics
class FixDPD_T : public FixDPD {

    public:
        // our friction coefficient gamma
        double gamma;

        // amplitude of the thermal noise given by sigma
        double sigma;
        // note that we will need to pass dt, timestep, and the temperature setpoint
        // to the fix compute somewhere
        // where temperature is specified by the interpolator class
        //
        // boolean updateGamma: denotes whether we update gamma when setpoint
        // temperature is changed.  True if we update gamma, false if we update sigma
        // (dependent on whether the user specifies sigma or gamma at run time)
        bool updateGamma;
        
        // given gamma, we make three constructors for the different implementations of interpolator
        FixDPD_T (boost::shared_ptr<State> state_, std::string handle_, std::string groupHandle_,
                  double gamma_, double rcut_, double s_, boost::python::list intervals_,
                  boost::python::list temps_);
        FixDPD_T (boost::shared_ptr<State> state_, std::string handle_, std::string groupHandle_,
                  double gamma_, double rcut_, double s_, boost::python::object tempFunc_);
        FixDPD_T (boost::shared_ptr<State> state_, std::string handle_, std::string groupHandle_,
                  double gamma_, double rcut_, double s_, double temp_);
        

        // same, but now we are given the thermal noise coefficient sigma instead
        FixDPD_T (boost::shared_ptr<State> state_, std::string handle_, std::string groupHandle_,
                  double sigma_, double rcut_, double s_, boost::python::list intervals_,
                  boost::python::list temps_);
        FixDPD_T (boost::shared_ptr<State> state_, std::string handle_, std::string groupHandle_,
                  double sigma_, double rcut_, double s_, boost::python::object tempFunc_);
        FixDPD_T (boost::shared_ptr<State> state_, std::string handle_, std::string groupHandle_,
                  double sigma_, double rcut_, double s_, double temp_);
        
        // our destructor
        ~FixDPD_T () {};

        // we compute the random and dissipative forces in compute
        void compute(bool);

        // we update the dissipative forces in stepFinal
        bool stepFinal();
        
        // hand some stuff to the evaluator (whatever it might need)
		bool prepareForRun();
		
        // 
        bool postRun();
        // this fix does not contribute to the single point energy
        void singlePointEng(float *);
        
        EvaluatorDPD_T evaluator;
}

#endif
