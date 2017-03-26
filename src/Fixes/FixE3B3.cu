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
    // first, the two-body contribution to force from E3B3..
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


    // and, the three body contribution

}



/* Single Point Eng
   */
void FixE3B3::singlePointEng(float *perParticleEng) {
    // first, the two-body contribution to the energy from E3B3..



    // and, the three body contribution

}



/* prepareForRun

   */



/* restart chunk?


   */



/* postRun


   */




/* exports

   */


