#include "FixE3B3.h"


#include "BoundsGPU.h"
#include "GridGPU.h"
#include "State.h"
#include "boost_for_export.h"
#include "cutils_math.h"
#include "ThreeBodyEvaluateIso.h"

const string E3B3Type = "E3B3";
/* Constructor
 * Makes an instance of the E3B3 fix
 */

FixE3B3::FixE3B3(boost::shared_ptr<State> state,
                  std::string handle,
                  std::string groupHandle): Fix(state_, handle_, groupHandle_, E3B3Type, true, true, false, 1) { 
    // set the cutoffs used in this potential
    rf = 5.2; // far cutoff for threebody interactions
    rs = 5.0; // short cutoff for threebody interactions
    rc = 7.2; // cutoff for our local neighborlist
    int largeNumber = 500; // we don't need any exclusions, which are denoted by enum EXCLUSIONMODE in State.h

    // gridGPU for E3B3 now instantiated
    gridGPU = GridGPUE3B3(state, rc, rc, rc, rc, largeNumber);

};
    

void FixE3B3::compute(bool computeVirials) {
    // send the molecules to the e3b3 evaluator, where we compute both the two-body correction
    // and the three-body interactions.
    // -- send the correct neighbor list (specific to this potential) and the array of water molecules
    //    local to this gpu

    // note that at this point, the evaluator should have been instantiated, and with the proper units for this 
    // simulation.
    
    /*
    int nAtoms = state->atoms.size();
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    */

    int nMolecules = waterMolecules.size();
    GPUData &gpd = state->gpd;
    // some GPU data for the molecules?


    // and we have our GridGPU already


    int activeIdx = gpd.activeIdx();
    // we need a perMolecArray!
    // the original:
    //uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    
    // our grid data holding our molecule-by-molecule neighbor list
    // -- we need to copy over the molecule array as well.
    uint16_t *neighborCounts = grid.perMolecArray.d_data.data();
    /*
    evalWrap->compute(nAtoms, gpd.xs(activeIdx), gpd.fs(activeIdx),
                      neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(),
                      state->devManager.prop.warpSize, numTypes, state->boundsGPU,
                      gpd.virials.d_data.data(), computeVirials);
    */
    
    // we will pass the following to our compute function:
    



}

void FixE3B3::stepInit(bool computeVirialsInForce){
    // we use this as an opportunity to re-create the local neighbor list, if necessary
    int periodicInterval = state->periodicInterval;
    if (state->turn % periodicInterval == 0) {
        // do the re-creation of the neighborlist for E3B3
        // -- make vector of the centers of mass (~ the positions of the molecules..)
        

        // -----> this should not return anything. it should operate on class variables.
        //   So, maybe have the grid take in a pointer to a class; default option is to have 
        //   this pointer be to our 'state' intance; 
        //    ---- this is for future development
        E3B3Grid.periodicBoundaryConditions(waterMolecules);


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
void FixE3B3::prepareForRun(){
    /* TODO:
     * -- evaluator
     * -- assorted GPU data stuff?!
     * -- GridGPUE3B3?? (for the first turn)
     * -- ????
     */ 
    return;

}


/* restart chunk?


   */



/* postRun


   */


// the atom ids are presented as the input; assembled into a molecule
void FixE3B3::addMolecule(int id_O, int id_H1, int id_H2, int id_M) {
    
    // id's are arranged as O, H, H, M
    std::vector<int> waterIds;

    // add to waterIds vector the four atom ids
    waterIds.push_back(id_O);
    waterIds.push_back(id_H1);
    waterIds.push_back(id_H2);
    waterIds.push_back(id_M);

    // mass of O > mass H1 == mass H2 > mass M
    bool ordered = true;
    double massO = state->atoms[idToIdx[id_O]].mass;
    double massH1 = state->atoms[idToIdx[id_H1]].mass;
    double massH2 = state->atoms[idToIdx[id_H2]].mass;
    double massM = state->atoms[idToIdx[id_M]].mass;

    if (! (massO > massH1 && massO > massH2)) {
        ordered = false;
    }
    if (massH1 != massH2) ordered = false;
    if (!(massH1 > massM)) ordered = false;

    if (! (ordered)) mdError("Ids in FixE3B3::addMolecule must be as O, H1, H2, M");

    // assemble them in to a molecule
    Molecule thisWater = Molecule(state, waterIds);

    // append this molecule to the class variable waterMolecules
    // -- molecule id is implicit as the index in this list
    waterMolecules.push_back(thisWater);
}

/* exports

   */

void export_FixE3B3() {
  py::class_<FixE3B3, boost::shared_ptr<FixE3B3>, py::bases<Fix> > ( 
								      "FixE3B3",
								      py::init<boost::shared_ptr<State>, std::string, std::string>
								      (py::args("state", "handle", "groupHandle", "rcut", "rs", "rf")
								       ))
    .def("createRigid", &FixRigid::createRigid,
	     (py::arg("id_a"), 
          py::arg("id_b"), 
          py::arg("id_c")
         )
	 );
}
