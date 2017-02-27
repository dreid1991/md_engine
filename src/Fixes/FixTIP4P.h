#pragma once
#ifndef FixTIP4P_H
#define FixTIP4P_H

#include "Fix.h"
#include "GPUArrayGlobal.h"

//! Make FixTIP4P available to the python interface
void export_FixTIP4P();

//! TIP4P/2005 water model

class FixTIP4P: public FixPair {

    public:
        // delete the default ctor
        FixTIP4P = delete();
    
        // what information will be required here?
        // --- we note that TIP4P has one LJ site and 3 charges
        // --- one of the charged sites has no mass (or some nominal mass)
        // --- the force exerted by (on) the ghost site must be distributed 
        //     to the real atoms s.t. we are physically modelling water
        // --- 




};

#endif /* FixTIP4P_H */
