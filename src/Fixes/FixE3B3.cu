#include "FixE3B3.h"
#include "BoundsGPU.h"
#include "GridGPU.h"
#include "State.h"
#include "boost_for_export.h"
#include "cutils_math.h"
#include "list_macro.h"

const std::string E3B3Type = "E3B3";
namespace py = boost::python;
/* Constructor
 * Makes an instance of the E3B3 fix
 */

FixE3B3::FixE3B3(boost::shared_ptr<State> state_,
                  std::string handle_,
                  std::string groupHandle_): Fix(state_, handle_, groupHandle_, E3B3Type, true, true, false, 1) { 
    // set the cutoffs used in this potential
    rf = 5.2; // far cutoff for threebody interactions (Angstroms)
    rs = 5.0; // short cutoff for threebody interactions (Angstroms)
    rc = 7.2; // cutoff for our local neighborlist (Angstroms)
    padding = 2.0; // implied since rc - rf = 2.0; pass this to local GridGPU on instantiation
    // to do: set up the local gridGPU for this set of GPUData; 
    // ---- which means we need to set up the local GPUData;
    // ------- can't do this until we have all the atoms in simulation; so do it in prepareForRun
};
    

void FixE3B3::compute(bool computeVirials) {
    // send the molecules to the e3b3 evaluator, where we compute both the two-body correction
    // and the three-body interactions.
    // -- send the correct neighbor list (specific to this potential) and the array of water molecules
    //    local to this gpu

    
    // some GPU data for the molecules?
    // --- TODO: change to myGpd; make sure that this is changed every (turn%periodicInterval == 0) in stepInit
    GPUData &gpd = state->gpd;

    // obviously, we also need the gpu data for the /atoms/


    // and we have our GridGPU already


    int activeIdx = gpd.activeIdx();
    // we need a perMolecArray!
    // the original:
    //uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    
    // our grid data holding our molecule-by-molecule neighbor list
    // -- we need to copy over the molecule array as well.
    uint16_t *neighborCounts = gridGPU.perAtomArray.d_data.data();
    /*
    evalWrap->compute(nAtoms, gpd.xs(activeIdx), gpd.fs(activeIdx),
                      neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(),
                      state->devManager.prop.warpSize, numTypes, state->boundsGPU,
                      gpd.virials.d_data.data(), computeVirials);
    */
    
    // we will pass the following to our compute function:
    
}

bool FixE3B3::stepInit(){
    // we use this as an opportunity to re-create the local neighbor list, if necessary
    int periodicInterval = state->periodicInterval;
    if (state->turn % periodicInterval == 0) {
        // do the re-creation of the neighborlist for E3B3
        // -- make vector of the centers of mass (~ the positions of the molecules..)
        

        // -----> this should not return anything. it should operate on class variables.
        //   So, maybe have the grid take in a pointer to a class; default option is to have 
        //   this pointer be to our 'state' intance; 
        //    ---- this is for future development
        gridGPU.periodicBoundaryConditions();


    }
    return true;
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
bool FixE3B3::prepareForRun(){
    /* TODO:
     * -- assorted GPU data stuff?!
     * -- GridGPUE3B3?? (for the first turn)
     * -- ????
     */ 
    float rs = 5.0;
    float rf = 5.2;

    /* TODO: put in the real values for these parameters */
    // the values for our E3B3 parameters.  should we call state->units here?
    // ---- probably for the Ea, Eb, Ec, E2 constants! But, distance is always in angstroms.
    float E2 = 1.0000;
    float Ea = 1.0000;
    float Eb = 1.0000;
    float Ec = 1.0000;
    float k2 = 1.0000;
    float k3 = 1.0000;
    evaluator = EvaluatorE3B3(rs, rf, E2,
                              Ea, Eb, Ec,
                              k2, k3);
    
    // for molecule in molecules list (we can assume it is populated at this point)
    

    
    /*
    int nAtoms = state->atoms.size();
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    */

    // see State.cpp: State::prepareForRun(); most of this code is taken from there;
    // but with molecules now!
    int nMolecules = waterMolecules.size();
    std::vector<float4> xs_vec;
    std::vector<uint> ids;

    xs_vec.reserve(nMolecules);
    ids.reserve(nMolecules);

    for (const auto &molecule: waterMolecules)  {
        Vector this_xs = molecule.COM();
        float4 new_xs = make_float4(this_xs[0], this_xs[1], this_xs[2], 0);
        xs_vec.push_back(new_xs);
        ids.push_back(molecule.id);
    }

    // note that gpd is the /local/ gpd
    gpd.xs.set(xs_vec);
    gpd.ids.set(ids);
   

    std::vector<int> id_vec = LISTMAPREF(Molecule, int, m, waterMolecules, m.id);
    std::vector<int> idToIdxs_vec;
    int size = *std::max_element(id_vec.begin(), id_vec.end()) + 1;
    idToIdxs_vec.reserve(size);
    for (int i=0; i<size; i++) {
        idToIdxs_vec.push_back(-1);
    }
    for (int i=0; i<id_vec.size(); i++) {
        idToIdxs_vec[id_vec[i]] = i;
    }

    gpd.idToIdxsOnCopy = idToIdxs_vec;
    gpd.idToIdxs.set(idToIdxs_vec);
    //gridGPU = grid.makeGPU(maxRCut);  // uses os, ns, ds, dsOrig from AtomGrid
    double maxRCut = rf;// cutoff of our potential (5.2 A)
    double padding = 2.0;
    double gridDim = maxRCut + padding;

    GPUData *gpuData = &gpd;

    // this number has no meaning whatsoever; it is completely arbitrary;
    // -- we are not using exclusionMode for this grid or set of GPUData
    int exclusionMode = 30;
    // I think this is doubly irrelevant, since we use a doExclusions(false) method later (below)

    gridGPU = GridGPU(state, gridDim, gridDim, gridDim, gridDim, exclusionMode, padding, gpuData);

    // tell gridGPU that the only GPUData we need to sort are positions (and, of course, the molecule/atom id's)
    gridGPU.onlyPositions(true);

    // tell gridGPU not to do any exclusions stuff
    gridGPU.doExclusions(false);

    // so, the only buffers that we need are the xs and ids!
    gpd.xsBuffer = GPUArrayGlobal<float4>(nMolecules);
    //gpd.vsBuffer = GPUArrayGlobal<float4>(nMolecules);
    //gpd.fsBuffer = GPUArrayGlobal<float4>(nMolecules);
    gpd.idsBuffer = GPUArrayGlobal<uint>(nMolecules);
    
    return true;
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
    double massO = state->idToAtom(id_O).mass; 
    double massH1 = state->idToAtom(id_H1).mass;
    double massH2 = state->idToAtom(id_H2).mass;
    double massM = state->idToAtom(id_M).mass;

    // check the ordering
    if (! (massO > massH1 && massO > massH2 )) {
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
								      (py::args("state", "handle", "groupHandle")
								       ))
    .def("addMolecule", &FixE3B3::addMolecule,
	     (py::arg("id_O"), 
          py::arg("id_H1"), 
          py::arg("id_H2"),
          py::arg("id_M")
         )
	 );
}
