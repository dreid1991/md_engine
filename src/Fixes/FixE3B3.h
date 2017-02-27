#pragma once
#ifndef FIXE3B3_H
#define FIXE3B3_H

/* inherits from Fix.h, has a FixTIP4P member */
#include "Fix.h"
#include "FixTIP4P.h"
#include "GPUArrayGlobal.h"

//! Make FixE3B3 available to the python interface
void export_FixE3B3();

//! Explicit 3-Body Potential, v3 (E3B3) for Water
/*
 * This fix implements the E3B3 water for water as 
 * described by Tainter, Shi, & Skinner in 
 * J. Chem. Theory Comput. 2015, 11, 2268-2277
 *
 * Note that this fix incorporates FixTIP4P as a member,
 * and so FixTIP4P should not be issued as a separate 
 * command within the input file.
 */

class FixE3B3: public Fix {
    
    public:

        // delete the default constructor
        FixE3B3() = delete;
        /* FixE3B3 constructor
         * -- pointer to state
         * -- handle for the fix
         * -- group handle
         * -- cutoff, & cutswitch?
         * -- two-body cutoff
         * ---???????????
         */
    FixE3B3(boost::shared_ptr<State> state,
                  std::string handle,
                  std::string groupHandle,
                  double _moreParmsHere); // TODO




    private:
        // are there any adjustable parameters within the TIP4P model?
        // how will we specify the water as input?
        FixTIP4P *twobody; // twobody reference potential is TIP4P/2005
        




};



#endif /* FIXE3B3_H */
