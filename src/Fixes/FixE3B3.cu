#include "FixE3B3.h"

#include "BoundsGPU.h"
#include "GridGPU.h"
#include "State.h"
#include "boost_for_export.h"
#include "cutils_math.h"
#include "ThreeBodyEvaluateIso.h"

/* Constructors


   */





/* Computes
   */

void FixE3B3::compute(bool computeVirials) {
    // send the molecules to the e3b3 evaluator, where we compute both the two-body correction
    // and the three-body interactions.
    // -- send the correct neighbor list (specific to this potential) and the array of water molecules
    //    local to this gpu

    // note that at this point, the evaluator should have been instantiated, and with the proper units for this 
    // simulation.
    int nAtoms = state->atoms.size();
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();

    evalWrap->compute(nAtoms, gpd.xs(activeIdx), gpd.fs(activeIdx),
                      neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(),
                      state->devManager.prop.warpSize, numTypes, state->boundsGPU,
                      gpd.virials.d_data.data(), computeVirials);

}

void FixE3B3::stepInit(bool computeVirialsInForce){
    // we use this as an opportunity to re-create the local neighbor list, if necessary
    int periodicInterval = state->periodicInterval;
    if (state->turn % periodicInterval == 0) {
        // do the re-creation of the neighborlist for E3B3
        E3B3Grid.periodicBoundaryConditions();
    }
}


/* Single Point Eng
   */
void FixE3B3::singlePointEng(float *perParticleEng) {
    // and, the three body contribution
    // -- we still pass everything molecule by molecule... but add it to their particle arrays

    // gonna need to look up how this is done..


}



/* prepareForRun

   */



/* restart chunk?


   */



/* postRun


   */




/* exports

   */


