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

//
// what arguments do we need here? we are updating the molecule positions from 
// the current atom positions
// __global__ void compute_COM(int4 *waterIds, float4 *xs, float4 *vs, int *idToIdxs, int nMols, float4 *com, BoundsGPU bounds) {

// from FixRigid.cu
__device__ inline float3 positionsToCOM(float3 *pos, float *mass, float ims) {
  return (pos[0]*mass[0] + pos[1]*mass[1] + pos[2]*mass[2] + pos[3]*mass[3])*ims;
}


// see FixRigid.cu! does the same thing. but now, we store it in their own gpd..
 __global__ void update_xs(int nMolecules, int4 *waterIds, float4 *mol_xs,
                           float4 *xs, float4 *vs, int *idToIdxs, BoundsGPU bounds) {

     // now do pretty much the same as FixRigid computeCOM()
     // --- remember to account for the M-site
    int idx = GETIDX();
    
    if (idx < nMolecules) {

        // may as well make these arrays
        int theseIds[4]; 
        float3 pos[4];
        float mass[4];

        theseIds[0] = waterIds[idx].x;
        theseIds[1] = waterIds[idx].y;
        theseIds[2] = waterIds[idx].z;
        theseIds[3] = waterIds[idx].w;

        float ims = 0.0f;
        for (int i = 0; i < 4; i++) {
            int thisId = theseIds[i];
            int thisIdx = idToIdxs[thisId];
            float3 p = make_float3(xs[thisIdx]);
            pos[i] = p;
            ims += vs[thisIdx].w;
            mass[i] = 1.0f / vs[thisIdx].w;
        }

        for (int i = 1; i < 4; i++) {
            float3 delta = pos[i] - pos[0];
            delta = bounds.minImage(delta);
            pos[i] = pos[0] + delta;
        }

        xs[idx]  = make_float4(positionsToCOM(pos, mass,ims));
        xs[idx].w = ims;
    }

}

void FixE3B3::compute(bool computeVirials) {
    
    // send the molecules to the e3b3 evaluator, where we compute both the two-body correction
    // and the three-body interactions.
    // -- send the correct neighbor list (specific to this potential) and the array of water molecules
    //    local to this gpu
    // -- still need to send the global simulation data, which contains the atoms itself
    
    // get the activeIdx for our local gpd (the molecule-by-molecule stuff);
    int activeIdx = gpd.activeIdx();

    // and the global gpd
    // --- IMPORTANT: the virials must be taken from the /global/ gpudata!
    GPUData &gpdGlobal = state->gpd;
    int globalActiveIdx = gpdGlobal.activeIdx();
    
    // our grid data holding our molecule-by-molecule neighbor list
    // -- we need to copy over the molecule array as well.
    
    // although it says 'perAtomArray', note that all of this gpd for this grid is by molecule
    // so, its just a misnomer in this instance. its a count of neighboring molecules.
    uint16_t *neighborCounts = gridGPU.perAtomArray.d_data.data();

    /* data required for compute_e3b3:
       - nMolecules
       - moleculesIdsToIdxs
       - waterIds (atom IDS in a given molecule)
       - molecules neighborcounts
       - molecules nlist
       - molecules - cumulSumMaxPerBlock (grid.perBlockArray.d_data.data())a
       - warpsize
       - atom idsToIdxs
       - atom positions
       - boundsGPU (state)
       - virials (global)
    */

    if (computeVirials) {
        compute_E3B3<EvaluatorE3B3, true> <<<NBLOCK(nMolecules), PERBLOCK>>> (
            nMolecules, 
            gpd.idToIdxs.d_data.data(), 
            waterIdsGPU.data(),
            gridGPU.perAtomArray.d_data.data(),
            gridGPU.neighborlist.data(), 
            gridGPU.perBlockArray.d_data.data(),
            state->devManager.prop.warpSize,
            gpdGlobal.idToIdxs.d_data.data(), 
            gpdGlobal.xs(globalActiveIdx), 
            gpdGlobal.fs(globalActiveIdx),
            state->boundsGPU, 
            gpdGlobal.virials.d_data.data(),
            evaluator);
        /*
        (int nMolecules, 
         const int *__restrict__ molIdToIdxs,
         const float4 *__restrict__ atomsFromMolecule,
         const uint16_t *__restrict__ neighborCounts, 
         const uint *__restrict__ neighborlist, 
         const uint32_t * __restrict__ cumulSumMaxPerBlock, 
         int warpSize, 
         const int *__restrict__ idToIdxs,
         const float4 *__restrict__ xs, 
         float4 *__restrict__ fs, 
         BoundsGPU bounds, 
         Virial *__restrict__ virials,
         EVALUATOR eval)
        */
    } else {
        compute_E3B3<EvaluatorE3B3, false> <<<NBLOCK(nMolecules), PERBLOCK>>> (
            nMolecules, 
            gpd.idToIdxs.d_data.data(), 
            waterIdsGPU.data(),
            neighborCounts, 
            gridGPU.neighborlist.data(), 
            gridGPU.perBlockArray.d_data.data(),
            state->devManager.prop.warpSize,
            gpdGlobal.idToIdxs.d_data.data(), 
            gpdGlobal.xs(globalActiveIdx), 
            gpdGlobal.fs(globalActiveIdx),
            state->boundsGPU, 
            gpdGlobal.virials.d_data.data(),
            evaluator);
    };
            /*
    evalWrap->compute(nAtoms, gpd.xs(activeIdx), gpd.fs(activeIdx),
                      neighborCounts, gridGPU.neighborlist.data(), gridGPU.perBlockArray.d_data.data(),
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
        // -- the xs of the molecules is /not/ updated with the atoms!
        //    but this is what we form our neighborlist off of (for the molecule-by-molecule neighborlist)
        //    so, do a kernel call here to update them to the current positions
        //    of their constituent atoms

        // for each thread, we have one molecule
        // -- get the atoms for this idx, compute COM, set the xs to the new value, and return
        //    -- need idToIdx for atoms? I think so.  Also, this is easy place to check 
        //       accessing the data arrays
        uint activeIdx = gpd.activeIdx();

        // get the global gpd and the bounds
        uint globalActiveIdx = state->gpd.activeIdx();
        GPUData &gpdGlobal = state->gpd;
        BoundsGPU &bounds = state->boundsGPU;

        // pass the local gpd (molecule by molecule) and the global (atom by atom) gpd
        // -- -with this, our local gpd data for the molecule COM is up to date with 
        //     the current atomic data
        update_xs<<<NBLOCK(nMolecules), PERBLOCK>>>(
            nMolecules, waterIdsGPU.data(), gpd.xs(activeIdx), 
            gpdGlobal.xs(globalActiveIdx), gpdGlobal.vs(globalActiveIdx), gpdGlobal.idToIdxs.d_data.data(),
            bounds
        );

        
        // our grid now operates on the updated molecule xs to get a molecule by molecule neighborlist    
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
    
    // units for distance are always angstroms, 
    // in the context of simulations that would use E3B3
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
    
    // instantiate the evaluator
    evaluator = EvaluatorE3B3(rs, rf, E2,
                              Ea, Eb, Ec,
                              k2, k3);
    
    // set up the int4 waterMoleculeIds
    int nMolecules = waterMolecules.size();

    waterIdsGPU = GPUArrayDeviceGlobal<int4>(nMolecules);
    waterIdsGPU.set(waterIds.data()); // waterIds vector populated as molecs added
    
    /*
    int nAtoms = state->atoms.size();
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    */

    // see State.cpp: State::prepareForRun(); most of this code is taken from there;
    // but with molecules now!
    nMolecules = waterMolecules.size();
    
    std::vector<float4> xs_vec;
    std::vector<uint> ids;

    xs_vec.reserve(nMolecules);
    ids.reserve(nMolecules);

    for (auto &molecule: waterMolecules)  {
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

    // make a pointer to this gpudata, which we will pass to the local grid
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
   * nothing to do hereeeee

   */


// the atom ids are presented as the input; assembled into a molecule
void FixE3B3::addMolecule(int id_O, int id_H1, int id_H2, int id_M) {
    
    // id's are arranged as O, H, H, M
    std::vector<int> localWaterIds;

    
    // add to waterIds vector the four atom ids
    localWaterIds.push_back(id_O);
    localWaterIds.push_back(id_H1);
    localWaterIds.push_back(id_H2);
    localWaterIds.push_back(id_M);


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
    Molecule thisWater = Molecule(state, localWaterIds);

    // append this molecule to the class variable waterMolecules
    // -- molecule id is implicit as the index in this list
    waterMolecules.push_back(thisWater);

    int4 idsAsInt4 = make_int4(localWaterIds[0], localWaterIds[1], localWaterIds[2], localWaterIds[3]);
    // and add to the global list
    waterIds.push_back(idsAsInt4);


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
