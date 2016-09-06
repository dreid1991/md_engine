#pragma once
#ifndef FIXDPD_H
#define FIXDPD_H

#include "boost_for_export.h"
#include "Interpolator.h"
#include "Fix.h"
//! Make FixDPD available to the pair base class in boost
void export_FixDPD();
//! Base class for dissipative particle dynamics fixes
/*!
 * Some generalizations will go here once the common features of 
 * DPD ensembles are determined
 */
namespace py = boost::python;
class FixDPD : public Interpolator, public Fix {
    public:

        // some constructor here
        // this will (hopefully) become more useful as we incorporate more ensembles of DPD
        // we receive a list of intervals and setpoint values
	FixDPD(State* state_, std::string handle_, 
			std::string groupHandle_, std::string type_, 
		    py::list intervals_, py::list temps_) : 
          Interpolator(intervals_, temps_), 
          Fix(state_, handle_, groupHandle_, type_, false, false, false, applyEvery=1)
        {

        };
    FixDPD(State* state_, std::string handle_, 
           std::string groupHandle_, std::string type_,
           py::object tempFunc_) : 
          Interpolator(tempFunc_), 
          Fix(state_, handle_, groupHandle_, type_, false, false, false, applyEvery=1)
    {

    };

    FixDPD(State* state_, std::string handle_,
           std::string groupHandle_, std::string type_, 
           double temp_): 
          Interpolator(temp_), 
          Fix(state_, handle_, groupHandle_, type_, false, false, false, applyEvery=1)
    {

    }; 


};

#endif




