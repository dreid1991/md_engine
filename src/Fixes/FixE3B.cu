#include "FixE3B.h"

#include "BoundsGPU.h"
#include "State.h"
#include "boost_for_export.h"
#include "cutils_math.h" 
#include "list_macro.h"

const std::string E3BType = "E3B";
namespace py = boost::python;
/* Constructor
 * Makes an instance of the E3B fix
 */

FixE3B::FixE3B(boost::shared_ptr<State> state_,
                  std::string handle_,
                  std::string groupHandle_): Fix(state_, handle_, groupHandle_, E3BType, true, true, false, 1) { 
    // set the cutoffs used in this potential
    rf = 5.2; // far cutoff for threebody interactions (Angstroms)
    rs = 5.0; // short cutoff for threebody interactions (Angstroms)
    rc = 6.2; // cutoff for our local neighborlist (Angstroms)
    padding = rc - rf; // rc - rf = 1.0; pass this to local GridGPU on instantiation
    style = "E3B3";
    requiresForces = false;
    requiresPerAtomVirials = false;
    prepared = false;
    nThreadPerAtom(state->devManager.prop.warpSize);

};

//
// what arguments do we need here? we are updating the molecule positions from 
// the current atom positions

// from FixRigid.cu
__device__ inline real3 positionsToCOM_E3B(real3 *pos, real *mass, real ims) {
  return (pos[0]*mass[0] + pos[1]*mass[1] + pos[2]*mass[2] + pos[3]*mass[3])*ims;
}


__global__ void updateMoleculeIdxsMap(int nMolecules, int4 *waterIds, 
                                      int4 *waterIdxs, int *mol_idToIdxs, int *idToIdxs) {

    int idx = GETIDX();

    if (idx < nMolecules) {
        // let this 'idx' represent the molecule id...
        // then, get the molecule idx 
        int molId = idx;
        // here is the index in the waterIdxs array that we will put the atomic idxs
        int molIdx = mol_idToIdxs[molId];

        // now get the atom ids in this molecule
        int4 atomIds = waterIds[molId];

        // convert them to idxs using the atom's idToidxs array
        int idx_O = idToIdxs[atomIds.x];
        int idx_H1= idToIdxs[atomIds.y];
        int idx_H2= idToIdxs[atomIds.z];
        int idx_M = idToIdxs[atomIds.w];

        // and here is the data that we will store in our waterIdxs, to be referenced by EvaluatorE3B
        int4 atomIdxs = make_int4(idx_O,idx_H1,idx_H2,idx_M);
        waterIdxs[molIdx] = atomIdxs;

    }
}

// see FixRigid.cu! does the same thing. but now, we store it in their own gpdLocal..
// -- nothing fancy, dont need the neighborlist here.
__global__ void update_xs(int nMolecules, int4 *waterIdxs, real4 *mol_xs, 
                          real4 *xs, real4 *vs, BoundsGPU bounds) {

     // now do pretty much the same as FixRigid computeCOM()
     // --- remember to account for the M-site
    int idx = GETIDX();
    
    if (idx < nMolecules) {
        // may as well make these arrays

        // just for clarity: we are looking at molecule /id/
        int4 atomIdxs = waterIdxs[idx];

        real4 pos_O_whole = xs[atomIdxs.x];
        real4 pos_H1_whole= xs[atomIdxs.y];
        real4 pos_H2_whole= xs[atomIdxs.z];
        real4 pos_M_whole = xs[atomIdxs.w];

        real4 vs_O  = vs[atomIdxs.x];
        real4 vs_H1 = vs[atomIdxs.y];
        real4 vs_H2 = vs[atomIdxs.z];
        real4 vs_M  = vs[atomIdxs.w];

        real mass_O = 1.0 / vs_O.w;
        real mass_H1= 1.0 / vs_H1.w;
        real mass_H2= 1.0 / vs_H2.w;
        real mass_M = 1.0 / vs_M.w;

        real totalMass = mass_O + mass_M + mass_H1 + mass_H2;
        real invMass   =   1.0 / totalMass;
        
        //real weightO = mass_O  * invMass; // not needed
        real weightH1= mass_H1 * invMass;
        real weightH2= mass_H2 * invMass;
        real weightM = mass_M  * invMass;

        real3 posO = make_real3(pos_O_whole);
        real3 posH1= make_real3(pos_H1_whole);
        real3 posH2= make_real3(pos_H2_whole);
        real3 posM = make_real3(pos_M_whole);

        real3 posH1_min = posO + bounds.minImage(posH1 - posO);
        real3 posH2_min = posO + bounds.minImage(posH2 - posO);
        real3 posM_min  = posO + bounds.minImage(posM  - posO);

        // more accurate way to calculate COM
        real3 pos_COM =  posO - (posH1_min * weightH1) - (posH2_min * weightH2) - (posM_min * weightM);
        // make pos_H1, pos_H2, pos_M the minImage w.r.t. pos_O
        real4 value_to_store = make_real4(pos_COM.x, pos_COM.y, pos_COM.z, invMass);
        mol_xs[idx] = value_to_store;

    }

}

void FixE3B::compute(int virialMode) {
    
    // send the molecules to the e3b3 evaluator, where we compute both the two-body correction
    // and the three-body interactions.
    // -- send the correct neighbor list (specific to this potential) and the array of water molecules
    //    local to this gpu
    // -- still need to send the global simulation data, which contains the atoms itself
 

    // get the activeIdx for our local gpdLocal (the molecule-by-molecule stuff);
    int activeIdx = gpdLocal.activeIdx();
    int warpSize = state->devManager.prop.warpSize;
    bool computeVirials = false;
    if (virialMode == 2 or virialMode == 1) computeVirials = true;
    // else false

    // and the global gpd
    // --- IMPORTANT: the virials must be taken from the /global/ gpudata!
    GPUData &gpdGlobal = state->gpd;
    int globalActiveIdx = gpdGlobal.activeIdx();
    
    // our grid data holding our molecule-by-molecule neighbor list
    // -- we need to copy over the molecule array as well.
    
    // although it says 'perAtomArray', note that all of this gpd for this grid is by molecule
    // so, its just a misnomer in this instance. its a count of neighboring molecules.
    
    // So, we actually to take a pair of molecules for i, j = neighbors [0, N-1], k = [j + 1, N]
    // -- we are limited to 1024 threads per block ...
    //    -- actually, only 256 can run concurrently, so at that point it is effectively saturated;
    //       Moreover, total number of active threads is 2048 * (# SM) == 57344 on GTX 1080 Ti FE;
    //       Or, 1792 molecules can be computed concurrently.


    /* data required for compute_e3b_force:
       - nMolecules
       - atomIdxsFromMoleculeIdx
       - molecules neighborcounts // by moleculeIdx
       - molecules nlist          // as moleculeIdx, by moleculeIdx
       - molecules - cumulSumMaxPerBlock (grid.perBlockArray.d_data.data())a
       - warpsize                 // device constant
       - atom positions           // as atomIdx
       - atom forces (....)       // as atomIdx
       - boundsGPU (state)
       - virials (global)         // as atomIdx
       - the evaluator
    */

    int maxNumNeighbors = gridGPULocal.computeMaxNumNeighbors();
    

    // So, we can have 256 concurrently computed threads per streaming multiprocessor; a warp is 32 threads;
    // per SM, we can therefore have 8 molecules; 
    //  -- we will do intra-warp reduction for forces and virials, rather than block reduction
    //  shared memory is used to store nlist molecule - atom positions, so we don't need to access 
    //  global memory for those except the one time
    size_t sharedMemSize = maxNumNeighbors * warpsPerBlock * sizeof(real4);
    if (computeVirials) {
            // numBlocks, threadsPerBlock defined in prepareForRun()
        compute_E3B_force<true><<<numBlocks,
                                threadsPerBlock,
                                sharedMemSize>>>(nMolecules,              // nMolecules in E3B potential
                                waterIdxsGPU.data(),                      // atomIdxs for molecule idx
                                gridGPULocal.perAtomArray.d_data.data(),  // neighbor counts for molecule idx
                                gridGPULocal.neighborlist.data(),         // neighbor idx for molecule idx
                                gridGPULocal.perBlockArray.d_data.data(), // cumulSumMaxPerBlock
                                warpSize,
                                gpdGlobal.xs(globalActiveIdx),            // as atom idxs 
                                gpdGlobal.fs(globalActiveIdx),            // as atom idxs
                                state->boundsGPU,
                                gpdGlobal.virials.d_data.data(),          // as atom idxs
                                warpsPerBlock,                            // used to compute molecule idx
                                maxNumNeighbors,                          // defines beginning index in smem
                                evaluator);


        // probably going to need all the threads
        // parallel reduction for virial contribution and forces
            //compute_E3B_force<EvaluatorE3B,true><<<numBlocks,threadsPerBlock,sharedmem>>>()

    } else {

            // numBlocks, threadsPerBlock defined in prepareForRun()
        compute_E3B_force<false><<<numBlocks,
                                threadsPerBlock,
                                sharedMemSize>>>(nMolecules,              // nMolecules in E3B potential
                                waterIdxsGPU.data(),                      // atomIdxs for molecule idx
                                gridGPULocal.perAtomArray.d_data.data(),  // neighbor counts for molecule idx
                                gridGPULocal.neighborlist.data(),         // neighbor idx for molecule idx
                                gridGPULocal.perBlockArray.d_data.data(), // cumulSumMaxPerBlock
                                warpSize,
                                gpdGlobal.xs(globalActiveIdx),            // as atom idxs 
                                gpdGlobal.fs(globalActiveIdx),            // as atom idxs
                                state->boundsGPU,
                                gpdGlobal.virials.d_data.data(),          // as atom idxs
                                warpsPerBlock,                            // used to compute molecule idx
                                maxNumNeighbors,                          // defines beginning index in smem
                                evaluator);

        //compute_E3B<EvaluatorE3B,false><<<numBlocks,threadsPerBlock,sharedmem>>>(

    }
}

void FixE3B::handleLocalData() {

    if (prepared) {
        uint activeIdx = gpdLocal.activeIdx();
        uint globalActiveIdx = state->gpd.activeIdx();
        GPUData &gpdGlobal = state->gpd;
        // calls a kernel that populates our waterIdxsGPU with current data
        // --- this has nothing to do with the positions; this should be called 
        //     every time 
        updateMoleculeIdxsMap<<<NBLOCK(nMolecules),PERBLOCK>>>(nMolecules,
                                                               waterIdsGPU.data(),
                                                               waterIdxsGPU.data(),
                                                               gpdLocal.idToIdxs.d_data.data(),
                                                               gpdGlobal.idToIdxs.d_data.data());
    }

}


bool FixE3B::stepInit(){
    // we use this as an opportunity to re-create the local neighbor list, if necessary
    //int periodicInterval = state->periodicInterval;
    // XXX on second thought, we want the list as short as possible - so re-make the 
    // threebody neighborlist every turn
    uint activeIdx = gpdLocal.activeIdx();

    // get the global gpd and the bounds
    uint globalActiveIdx = state->gpd.activeIdx();
    GPUData &gpdGlobal = state->gpd;
    BoundsGPU &bounds = state->boundsGPU;

    // do the re-creation of the neighborlist for E3B
    // -- the xs of the molecules is /not/ updated with the atoms!
    //    but this is what we form our neighborlist off of (for the molecule-by-molecule neighborlist)
    //    so, do a kernel call here to update them to the current positions
    //    of their constituent atoms

    // update the positions of our molecules
    // --- if we assume that waterIdxsMap is up-to-date, we can store the positions of the molecule 
    //     directly in waterIdxsGPU
    update_xs<<<NBLOCK(nMolecules), PERBLOCK>>>(nMolecules, 
                                                waterIdxsGPU.data(),           
                                                gpdLocal.xs(activeIdx),        // to store COMs
                                                gpdGlobal.xs(globalActiveIdx), // positions of atoms
                                                gpdGlobal.vs(globalActiveIdx), // for masses
                                                bounds
                                                );



    // for each thread, we have one molecule
    // -- get the atoms for this idx, compute COM, set the xs to the new value, and return
    //    -- need idToIdx for atoms? I think so.  Also, this is easy place to check 
    //       accessing the data arrays

    // pass the local gpdLocal (molecule by molecule) and the global (atom by atom) gpd
    // -- -with this, our local gpdLocal data for the molecule COM is up to date with 
    //     the current atomic data

    // our grid now operates on the updated molecule xs to get a molecule by molecule neighborlist    
    gridGPULocal.periodicBoundaryConditions(-1,true);
    
    // update the molecule idxs map
    updateMoleculeIdxsMap<<<NBLOCK(nMolecules),PERBLOCK>>>(nMolecules,
                                                           waterIdsGPU.data(),
                                                           waterIdxsGPU.data(),
                                                           gpdLocal.idToIdxs.d_data.data(),
                                                           gpdGlobal.idToIdxs.d_data.data());
        
    return true;
}

/* Single Point Eng
 *
 *
 *
 */
void FixE3B::singlePointEng(real *perParticleEng) {
    

    // get the activeIdx for our local gpdLocal (the molecule-by-molecule stuff);
    int activeIdx = gpdLocal.activeIdx();
    int warpSize = state->devManager.prop.warpSize;

    // and the global gpd
    // --- IMPORTANT: the virials must be taken from the /global/ gpudata!
    GPUData &gpdGlobal = state->gpd;
    int globalActiveIdx = gpdGlobal.activeIdx();
    int maxNumNeighbors = gridGPULocal.computeMaxNumNeighbors();
    

    // So, we can have 256 concurrently computed threads per streaming multiprocessor; a warp is 32 threads;
    // per SM, we can therefore have 8 molecules; 
    //  -- we will do intra-warp reduction for forces and virials, rather than block reduction
    //  shared memory is used to store nlist molecule - atom positions, so we don't need to access 
    //  global memory for those except the one time
    size_t sharedMemSize = maxNumNeighbors * warpsPerBlock * sizeof(real4);
    compute_E3B_energy<<<numBlocks,
                        threadsPerBlock,
                        sharedMemSize>>>(nMolecules,              // nMolecules in E3B potential
                        waterIdxsGPU.data(),                      // atomIdxs for molecule idx
                        gridGPULocal.perAtomArray.d_data.data(),  // neighbor counts for molecule idx
                        gridGPULocal.neighborlist.data(),         // neighbor idx for molecule idx
                        gridGPULocal.perBlockArray.d_data.data(), // cumulSumMaxPerBlock
                        warpSize,
                        gpdGlobal.xs(globalActiveIdx),            // as atom idxs 
                        perParticleEng,                           // as atom idxs
                        state->boundsGPU,
                        warpsPerBlock,                            // used to compute molecule idx
                        maxNumNeighbors,                          // defines beginning index in smem
                        evaluator);                               // assumes prepareForRun has been called

    return;
}


void FixE3B::createEvaluator() {
    
    // style defaults to E3B3; otherwise, it can be set to E3B2;
    // there are no other options.
    float kjToKcal = 0.23900573614;
    float rs = 5.0;
    float rf = 5.2;
    float k2 = 4.872;
    float k3 = 1.907;
    // default
    if (style == "E3B3") {
        // as angstroms

        // E2, Ea, Eb, Ec as kJ/mole -> convert to kcal/mole
        float E2 = 453000;
        float Ea = 150.0000;
        float Eb = -1005.0000;
        float Ec = 1880.0000;

        // k2, k3 as angstroms
        
        E2 *= kjToKcal;
        Ea *= kjToKcal;
        Eb *= kjToKcal;
        Ec *= kjToKcal;

        // 0 = REAL, 1 = LJ (see /src/Units.h)
        if (state->units.unitType == 1) {
            mdError("Units for E3B potential are not yet as LJ\n");
        }
            // converting to LJ from kcal/mol

        // instantiate the evaluator
        evaluator = EvaluatorE3B(rs, rf, E2,
                                  Ea, Eb, Ec,
                                  k2, k3);
        
   
    // other option
    } else if (style == "E3B2") {
    
        float E2 = 2349000.0; // kj/mol
        float Ea = 1745.7;
        float Eb = -4565.0;
        float Ec = 7606.8;

        E2 *= kjToKcal;
        Ea *= kjToKcal;
        Eb *= kjToKcal;
        Ec *= kjToKcal;

        // 0 = REAL, 1 = LJ (see /src/Units.h)
        if (state->units.unitType == 1) {
            mdError("Units for E3B potential are not yet as LJ\n");
        }
            // converting to LJ from kcal/mol

        // instantiate the evaluator
        evaluator = EvaluatorE3B(rs, rf, E2,
                                  Ea, Eb, Ec,
                                  k2, k3);
        
    } else {
        mdError("Unknown style in FixE3B.\n");
    }
};

void FixE3B::setStyle(std::string style_) {

    if (style_ == "E3B3") {
        style = style_;
    } else if (style_ == "E3B2") {
        style = style_;
    } else {
        mdError("Unrecognized style in FixE3B; options are E3B2 or E3B3.  Aborting.\n");
    }
}


/* prepareForRun

   */
bool FixE3B::prepareForRun(){
   
    // a style argument should have been passed by now
    // -- if they are not using the default (E3B3), they had to call setStyle('E3B2') in their python script
    nMolecules = waterMolecules.size();
    waterIdsGPU = GPUArrayDeviceGlobal<int4>(nMolecules);
    waterIdsGPU.set(waterIds.data()); // waterIds vector populated as molecs added
    
    createEvaluator();

    std::vector<real4> xs_vec;
    std::vector<uint> ids;

    xs_vec.reserve(nMolecules);
    ids.reserve(nMolecules);

    
    int workingId = 0;
    // so, the mass of the molecule starts at 0.0, until we update this
    // -- note that periodicBoundaryConditions in State::prepareForRun has been called by this point
    //    and so the atom idxs has been set up
    for (auto &molecule: waterMolecules)  {
        molecule.id = workingId;
        Vector this_xs = molecule.COM();
        real4 new_xs = make_real4(this_xs[0], this_xs[1], this_xs[2], 0);
        xs_vec.push_back(new_xs);

        ids.push_back(molecule.id);
        workingId++;
    }

    // note that gpd is the /local/ gpd
    gpdLocal.xs.set(xs_vec); // this is correct
    gpdLocal.ids.set(ids);   // this is correct
   
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

    /* Setting the local gpu data - our molecule information */
    gpdLocal.idToIdxsOnCopy = idToIdxs_vec;
    gpdLocal.idToIdxs.set(idToIdxs_vec);
    gpdLocal.xs.dataToDevice();
    gpdLocal.ids.dataToDevice();
    gpdLocal.idToIdxs.dataToDevice();
    int activeIdx = gpdLocal.activeIdx();
    
    // so, the only buffers that we need are the xs and ids!
    gpdLocal.xsBuffer = GPUArrayGlobal<real4>(nMolecules);
    gpdLocal.idsBuffer = GPUArrayGlobal<uint>(nMolecules);

    /* Setting parameters for E3B kernel; we do one molecule per warp */
    warpsPerBlock = 8;
    numBlocks = (int) ceil(nMolecules / warpsPerBlock);  // at 8 molecules per block, we need 
    // (nMolecules / molecules per block) = nblocks
    // warpSize gives threads per Molecule;
    int warpSize = state->devManager.prop.warpSize;
    // with one molecule per warp, warpSize * warpsPerBlock gives threadsPerBlock
    nThreadsPerMolecule = warpSize;
    threadsPerBlock = warpSize * warpsPerBlock;

    nThreadPerAtom(nThreadsPerMolecule);
    nThreadPerBlock(nThreadsPerMolecule * warpsPerBlock); // XXX this is only correct if its one molecule per warp, as we currently have it

    /* Setting the GridGPU, where we do a molecular neighborlist */
    double maxRCut = rf;// cutoff of our potential (5.2 A)
    double gridDim = maxRCut + padding; // padding is defined in the ctor

    // this number has no meaning whatsoever; it is completely arbitrary;
    // -- we are not using exclusionMode for this grid or set of GPUData
    int exclusionMode = 30;
    // I think this is doubly irrelevant, since we use a doExclusions(false) method later (below)

    gridGPULocal = GridGPU(state, gridDim, gridDim, gridDim, gridDim, exclusionMode, padding, &gpdLocal,state->nPerRingPoly,false); // if we have ringpolymers for atoms, we similarly have ring polymers for molecules..
    // ok; after it gets back from its constructor, manually set our nThreadPerBlock to 
    gridGPULocal.nThreadPerBlock(threadsPerBlock);
    gridGPULocal.nThreadPerAtom(nThreadsPerMolecule); // our
    gridGPULocal.initArraysTune();                    

    // tell gridGPU that the only GPUData we need to sort are positions (and, of course, the molecule/atom id's)
    // XXX This /must/ be set prior to gridGPULocal.periodicBoundaryConditions, else there will 
    // be a segmentation fault (since our gpdLocal does not have vs, fs arrays, and those will 
    // be attempted to be sorted by GridGPU
    gridGPULocal.onlyPositions(true);

    // tell gridGPU not to do any exclusions stuff
    // XXX This /must/ be set prior to gridGPULocal.periodicBoundaryConditions, else there will 
    // be an attempt to do exclusions, and an array with improper size will be sent to the GPU, 
    // causing a segmentation fault.
    gridGPULocal.doExclusions(false);

    
    // makes the neighborlist with the 
    gridGPULocal.periodicBoundaryConditions(-1, true); // MUST update waterIdxsGPU after this 
    // before we compute anything E3B related

    // consider - under what circumstances will we need to compute more than one molecule per warp?

    // we have our nMolecules variable; so,
    // with compute capability > 3.0, our gridDim in the x direction can be like 2.1B...
    // this isn't really something we need to worry about.
    // -- initially, let's try 8 warpsPerBlock (8 molecules per block) -- this can be adjusted later
    
    waterIdxsGPU = GPUArrayDeviceGlobal<int4>(nMolecules);
    waterIdxsGPU.set(waterIds.data()); // initialize as waterIds; this is a map that is updated at...
    
    GPUData &gpdGlobal = state->gpd;
    int globalActiveIdx = gpdGlobal.activeIdx();


    // everything here is prepared; set it true, THEN:  handleLocalData(), which needs to know that 
    // prepared == true; then, return prepared.
    prepared = true;

    handleLocalData(); // on having a changed idToIdx, for either molecules or atoms, this updates the 
    // int4 atomIdxs stored in waterIdxsGPU

    return prepared;
}


/* restart chunk?

// TODO: should load the style, and then also load the int4 waterIds; everything else can be reconstructed
   */



/* postRun
   * nothing to do here

   */


// the atom ids are presented as the input; assembled into a molecule
void FixE3B::addMolecule(int id_O, int id_H1, int id_H2, int id_M) {
    
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

    if (! (ordered)) mdError("Ids in FixE3B::addMolecule must be as O, H1, H2, M");

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

void export_FixE3B() {
  py::class_<FixE3B, boost::shared_ptr<FixE3B>, py::bases<Fix> > 
	("FixE3B",
         py::init<boost::shared_ptr<State>, std::string, std::string> 
	 (py::args("state", "handle", "groupHandle")
	 )
	)
    // methods inherited from Fix are already exposed (see export_Fix())
    .def("addMolecule", &FixE3B::addMolecule,
	     (py::arg("id_O"), 
          py::arg("id_H1"), 
          py::arg("id_H2"),
          py::arg("id_M")
         )
    )
    .def("setStyle", &FixE3B::setStyle,
         py::arg("style")
        )
    // the list of molecules
    .def_readonly("molecules", &FixE3B::waterMolecules)
    .def_readonly("nMolecules", &FixE3B::nMolecules)
    .def_readonly("gridGPU", &FixE3B::gridGPULocal)
    // can i export GPD??  How would I even do this?  Would need to export /[0
    ;
  
}
