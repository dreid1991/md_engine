#include "Fix.h"
#include "GPUArray.h"
/*
 * Okay, here's some ambiguity.  You initialize spring static for group a, and group a.  Tether positions are computed for those atoms, and you run the simulation.  
 * Now, you change who's in group a.  What does the simulation do? Does it call thetherFunc for the new atoms?  No, that is not explicit.  Could do unpredictable things for poor user.
 *
 * Instead, it will keep its current tetherPos list until updateTetherPositions is called, at which point the entire list is recomputed.
 *
 * Would also be nice to have some way of directly editing the tethers. Just exposing h_data of tetherPos is worthless b/c bitcast id
 *
 * For future optimization: if you don't have a lot of tethered atoms, you're likely overloading some SMUs while not using others, so try to spread the work out over more blocks if not many atoms are involved
 */

void export_FixSpringStatic();
class FixSpringStatic : public Fix {
    //these will be of only those in the group, so no need to 
    public:
        GPUArray<float4> tethers;
        double k;
        void compute(bool);
        void updateTethers();
        bool dataToDevice();
        bool prepareForRun();
        PyObject *tetherFunc;
        Vector multiplier;
        FixSpringStatic(SHARED(State), string handle_, string groupHandle_, double k_, PyObject *tetherFunc_, Vector multiplier=Vector(1, 1, 1));
    
};
