#include "FixE3B.h"
#include "DeviceManager.h"

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
                  std::string style_): Fix(state_, handle_, "all", E3BType, true, true, false, 1),style(style_) { 
    // set the cutoffs used in this potential
    rf = 5.2; // far cutoff for threebody interactions (Angstroms)
    rs = 5.0; // short cutoff for threebody interactions (Angstroms)
    rc = 10.0; // cutoff for the neighborlist 

    // ---- dictates that the box must be, at minimum, 2.2nm on a side (rc + padding < half_box_dim)
    // as an aside, we're just going to have the 'positions' of the molecule be the position of the oxygen.
    // -- the GMX implementation always just finds the oxygens on the neighborlist,
    //    so this should yield equivalent results

    padding = 1.0; // 
    requiresForces = false; // does NOT require the forces to prepare itself
    requiresPerAtomVirials = false;
    prepared = false;
    if (style_ != "E3B3" && style_ != "E3B2") {
        std::cout << "FixE3B received the style argument: " << style << std::endl;
        mdError("FixE3B requires the style argument to be either 'E3B3' or 'E3B2'; Aborting.");
    }
};

// needs to be called whenever we form the molecular or atomic neighborlists, as one or the other 
// of the idx maps have changed.
__global__ void updateMoleculeIdxsMap(int nMolecules, int4 *waterIds, 
                                      int4 *waterIdxs, int *mol_idToIdxs, int *idToIdxs) {

    int molId = GETIDX();

    if (molId < nMolecules) {
        // then, get the molecule idx 
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

__global__ void update_xs(int nMolecules, int4 *waterIds, real4 *mol_xs, 
                          real4 *xs, real4 *vs, int *mol_idToIdxs, int *idToIdxs) {
     // --- remember to account for the M-site
    int molId = GETIDX();
    if (molId < nMolecules) {
        int  mol_idx = mol_idToIdxs[molId];
        int4 atomIds = waterIds[molId];
        int  idx_O   = idToIdxs[atomIds.x]; // oxygen atom idx in the global xs aray

        real4 pos_O_whole = xs[idx_O];
        real4 vel_O_whole = vs[idx_O];
        real  invMass     = vel_O_whole.w;
        real4 value_to_store = make_real4(pos_O_whole.x, pos_O_whole.y, pos_O_whole.z, invMass);
        // update the molecule position at idx
        mol_xs[mol_idx] = value_to_store;
    }
}

void FixE3B::compute(int virialMode) {
    
    // get the activeIdx for our local gpdLocal (the molecule-by-molecule stuff);
    int activeIdx = gpdLocal.activeIdx();
    int warpSize = state->devManager.prop.warpSize;
    bool computeVirials = false;
    if (virialMode == 2 or virialMode == 1) computeVirials = true;
    int NTPB = state->nThreadPerBlock;
    int NTPA = state->nThreadPerAtom;
    int nPerRingPoly = state->nPerRingPoly;
    
    // and the global gpd
    // --- IMPORTANT: the virials must be taken from the /global/ gpudata!
    GPUData &gpdGlobal = state->gpd;
    int globalActiveIdx = gpdGlobal.activeIdx();

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

    bool multiThreadPerAtom = state->nThreadPerAtom > 1 ? true : false;
    /*
    std::cout << "in E3B compute().." << std::endl;
    std::cout << "pairPairEnergies has size " << pairPairEnergies.size() << std::endl;
    std::cout << "pairPairForces has size " << pairPairForces.size() << std::endl;
    std::cout << "computeThis has size " << computeThis.size() << std::endl;
    std::cout << "pairPairTotal has size " << pairPairTotal.size() << std::endl;
    */

    size_t smem_twobody_force   = NTPA > 1 ? NTPB * ( 2 * sizeof(real3) + sizeof(Virial)) : 0 ;
    size_t smem_threebody_force = NTPA > 1 ? NTPB * (3 * (sizeof(real3) + sizeof(Virial))): 0 ;
    /*
    std::cout << "smem_twobody_force: " << smem_twobody_force << std::endl;
    std::cout << "smem_threebody_force: " << smem_threebody_force << std::endl;
    */

    if (computeVirials) {
        if (multiThreadPerAtom) {
            compute_E3B_force_twobody<true,true><<<NBLOCKTEAM(nMolecules, NTPB, NTPA),NTPB,smem_twobody_force>>>(nMolecules,
                                    nPerRingPoly,
                                    waterIdxsGPU.data(),                      // atomIdxs for molecule idx
                                    gpdGlobal.xs(globalActiveIdx),            // as atom idxs 
                                    gpdGlobal.fs(globalActiveIdx),            // as atom idxs
                                    gridGPULocal.perAtomArray.d_data.data(),  // neighbor counts for molecule idx
                                    gridGPULocal.neighborlist.data(),         // neighbor idx for molecule idx
                                    gridGPULocal.perBlockArray.d_data.data(), // cumulSumMaxPerBlock
                                    warpSize,
                                    pairPairTotal.data(),
                                    pairPairEnergies.data(),
                                    forces_b2a1.data(),
                                    forces_c2a1.data(),
                                    forces_b1a2.data(),
                                    forces_c1a2.data(),
                                    computeThis.data(),
                                    NTPA,
                                    state->boundsGPU,
                                    gpdGlobal.virials.d_data.data(),          // as atom idxs
                                    evaluator);
            CUT_CHECK_ERROR("compute_E3B_force failed - twobody, call 1.\n");
        } else {
            compute_E3B_force_twobody<true,false><<<NBLOCKTEAM(nMolecules, NTPB, NTPA),
                                                  NTPB,smem_twobody_force>>>(
                                    nMolecules,
                                    nPerRingPoly,
                                    waterIdxsGPU.data(),                      // atomIdxs for molecule idx
                                    gpdGlobal.xs(globalActiveIdx),            // as atom idxs 
                                    gpdGlobal.fs(globalActiveIdx),            // as atom idxs
                                    gridGPULocal.perAtomArray.d_data.data(),  // neighbor counts for molecule idx
                                    gridGPULocal.neighborlist.data(),         // neighbor idx for molecule idx
                                    gridGPULocal.perBlockArray.d_data.data(), // cumulSumMaxPerBlock
                                    warpSize,
                                    pairPairTotal.data(),
                                    pairPairEnergies.data(),
                                    forces_b2a1.data(),
                                    forces_c2a1.data(),
                                    forces_b1a2.data(),
                                    forces_c1a2.data(),
                                    computeThis.data(),
                                    NTPA,
                                    state->boundsGPU,
                                    gpdGlobal.virials.d_data.data(),          // as atom idxs
                                    evaluator);
        CUT_CHECK_ERROR("compute_E3B_force failed - twobody, call 2.\n");
        }

    } else {

            // numBlocks, threadsPerBlock defined in prepareForRun()
        if (multiThreadPerAtom) {
            compute_E3B_force_twobody<false,true><<<NBLOCKTEAM(nMolecules,NTPB, NTPA),NTPB,smem_twobody_force>>>(
                                    nMolecules,
                                    nPerRingPoly,
                                    waterIdxsGPU.data(),                      // atomIdxs for molecule idx
                                    gpdGlobal.xs(globalActiveIdx),            // as atom idxs 
                                    gpdGlobal.fs(globalActiveIdx),            // as atom idxs
                                    gridGPULocal.perAtomArray.d_data.data(),  // neighbor counts for molecule idx
                                    gridGPULocal.neighborlist.data(),         // neighbor idx for molecule idx
                                    gridGPULocal.perBlockArray.d_data.data(), // cumulSumMaxPerBlock
                                    warpSize,
                                    pairPairTotal.data(),
                                    pairPairEnergies.data(),
                                    forces_b2a1.data(),
                                    forces_c2a1.data(),
                                    forces_b1a2.data(),
                                    forces_c1a2.data(),
                                    computeThis.data(),
                                    NTPA,
                                    state->boundsGPU,
                                    gpdGlobal.virials.d_data.data(),          // as atom idxs
                                    evaluator);
            CUT_CHECK_ERROR("compute_E3B_force failed - twobody, call 3.\n");
            } else {
            compute_E3B_force_twobody<false,false><<<NBLOCKTEAM(nMolecules,NTPB,NTPA),NTPB,smem_twobody_force>>>(
                                    nMolecules,
                                    nPerRingPoly,
                                    waterIdxsGPU.data(),                      // atomIdxs for molecule idx
                                    gpdGlobal.xs(globalActiveIdx),            // as atom idxs 
                                    gpdGlobal.fs(globalActiveIdx),            // as atom idxs
                                    gridGPULocal.perAtomArray.d_data.data(),  // neighbor counts for molecule idx
                                    gridGPULocal.neighborlist.data(),         // neighbor idx for molecule idx
                                    gridGPULocal.perBlockArray.d_data.data(), // cumulSumMaxPerBlock
                                    warpSize,
                                    pairPairTotal.data(),
                                    pairPairEnergies.data(),
                                    forces_b2a1.data(),
                                    forces_c2a1.data(),
                                    forces_b1a2.data(),
                                    forces_c1a2.data(),
                                    computeThis.data(),
                                    NTPA,
                                    state->boundsGPU,
                                    gpdGlobal.virials.d_data.data(),          // as atom idxs
                                    evaluator);
            CUT_CHECK_ERROR("compute_E3B_force failed - twobody, call 4.\n");
            }
    }

    if (computeVirials) {
            // numBlocks, threadsPerBlock defined in prepareForRun()
        if (multiThreadPerAtom) {
            compute_E3B_force_threebody<true,true><<<NBLOCKTEAM(nMolecules,NTPB,NTPA),NTPB,smem_threebody_force>>>(
                                    nMolecules,  // nMolecules in E3B potential
                                    nPerRingPoly,
                                    waterIdxsGPU.data(),                      // atomIdxs for molecule idx
                                    gpdGlobal.xs(globalActiveIdx),            // as atom idxs 
                                    gpdGlobal.fs(globalActiveIdx),            // as atom idxs
                                    gridGPULocal.perAtomArray.d_data.data(),  // neighbor counts for molecule idx
                                    gridGPULocal.neighborlist.data(),         // neighbor idx for molecule idx
                                    gridGPULocal.perBlockArray.d_data.data(), // cumulSumMaxPerBlock
                                    warpSize,
                                    state->boundsGPU,
                                    pairPairTotal.data(),
                                    pairPairEnergies.data(),
                                    forces_b2a1.data(),
                                    forces_c2a1.data(),
                                    forces_b1a2.data(),
                                    forces_c1a2.data(),
                                    computeThis.data(),
                                    gpdGlobal.virials.d_data.data(),          // as atom idxs
                                    NTPA,
                                    evaluator);
            CUT_CHECK_ERROR("compute_E3B_force failed - threebody, call 5.\n");
        } else {
            compute_E3B_force_threebody<true,false><<<NBLOCKTEAM(nMolecules,NTPB,NTPA),NTPB,smem_threebody_force>>>(
                                    nMolecules,              // nMolecules in E3B potential
                                    nPerRingPoly,
                                    waterIdxsGPU.data(),                      // atomIdxs for molecule idx
                                    gpdGlobal.xs(globalActiveIdx),            // as atom idxs 
                                    gpdGlobal.fs(globalActiveIdx),            // as atom idxs
                                    gridGPULocal.perAtomArray.d_data.data(),  // neighbor counts for molecule idx
                                    gridGPULocal.neighborlist.data(),         // neighbor idx for molecule idx
                                    gridGPULocal.perBlockArray.d_data.data(), // cumulSumMaxPerBlock
                                    warpSize,
                                    state->boundsGPU,
                                    pairPairTotal.data(),
                                    pairPairEnergies.data(),
                                    forces_b2a1.data(),
                                    forces_c2a1.data(),
                                    forces_b1a2.data(),
                                    forces_c1a2.data(),
                                    computeThis.data(),
                                    gpdGlobal.virials.d_data.data(),          // as atom idxs
                                    NTPA,
                                    evaluator);
            CUT_CHECK_ERROR("compute_E3B_force failed - threebody, call 6.\n");

        }


    } else {

            // numBlocks, threadsPerBlock defined in prepareForRun()
        if (multiThreadPerAtom) {
            compute_E3B_force_threebody<false,true><<<NBLOCKTEAM(nMolecules,NTPB,NTPA),NTPB,smem_threebody_force>>>(
                                    nMolecules,              // nMolecules in E3B potential
                                    nPerRingPoly,
                                    waterIdxsGPU.data(),                      // atomIdxs for molecule idx
                                    gpdGlobal.xs(globalActiveIdx),            // as atom idxs 
                                    gpdGlobal.fs(globalActiveIdx),            // as atom idxs
                                    gridGPULocal.perAtomArray.d_data.data(),  // neighbor counts for molecule idx
                                    gridGPULocal.neighborlist.data(),         // neighbor idx for molecule idx
                                    gridGPULocal.perBlockArray.d_data.data(), // cumulSumMaxPerBlock
                                    warpSize,
                                    state->boundsGPU,
                                    pairPairTotal.data(),
                                    pairPairEnergies.data(),
                                    forces_b2a1.data(),
                                    forces_c2a1.data(),
                                    forces_b1a2.data(),
                                    forces_c1a2.data(),
                                    computeThis.data(),
                                    gpdGlobal.virials.d_data.data(),          // as atom idxs
                                    NTPA,
                                    evaluator);
            CUT_CHECK_ERROR("compute_E3B_force failed - threebody, call 7.\n");
        } else { 
        compute_E3B_force_threebody<false,false><<<NBLOCKTEAM(nMolecules,NTPB,NTPA),NTPB,smem_threebody_force>>>(
                                    nMolecules,              // nMolecules in E3B potential
                                    nPerRingPoly,
                                    waterIdxsGPU.data(),                      // atomIdxs for molecule idx
                                    gpdGlobal.xs(globalActiveIdx),            // as atom idxs 
                                    gpdGlobal.fs(globalActiveIdx),            // as atom idxs
                                    gridGPULocal.perAtomArray.d_data.data(),  // neighbor counts for molecule idx
                                    gridGPULocal.neighborlist.data(),         // neighbor idx for molecule idx
                                    gridGPULocal.perBlockArray.d_data.data(), // cumulSumMaxPerBlock
                                    warpSize,
                                    state->boundsGPU,
                                    pairPairTotal.data(),
                                    pairPairEnergies.data(),
                                    forces_b2a1.data(),
                                    forces_c2a1.data(),
                                    forces_b1a2.data(),
                                    forces_c1a2.data(),
                                    computeThis.data(),
                                    gpdGlobal.virials.d_data.data(),          // as atom idxs
                                    NTPA,
                                    evaluator);
            CUT_CHECK_ERROR("compute_E3B_force failed - threebody, call 8.\n");
        }
    }
    
    cudaDeviceSynchronize();
    
    // memset to zero
    pairPairTotal.memset(0);
    pairPairEnergies.memset(0);
    forces_b2a1.memset(0);
    forces_c2a1.memset(0);
    forces_b1a2.memset(0);
    forces_c1a2.memset(0);
    computeThis.memset(0);
    CUT_CHECK_ERROR("compute_E3B_force failed.\n");
}

void FixE3B::handleLocalData() {

    if (prepared) {
        uint activeIdx = gpdLocal.activeIdx();
        uint globalActiveIdx = state->gpd.activeIdx();
        GPUData &gpdGlobal = state->gpd;
        //BoundsGPU &bounds = state->boundsGPU;

        // update the molecule positions to be consistent with oxygen atom positions
        update_xs<<<NBLOCK(nMolecules), PERBLOCK>>>(nMolecules, 
                                                waterIdsGPU.data(),           
                                                gpdLocal.xs(activeIdx),        // 
                                                gpdGlobal.xs(globalActiveIdx), // 
                                                gpdGlobal.vs(globalActiveIdx), // 
                                                gpdLocal.idToIdxs.d_data.data(),
                                                gpdGlobal.idToIdxs.d_data.data()
                                                );

        CUT_CHECK_ERROR("update_xs failed in handleLocalData!");


        // our grid now operates on the updated molecule xs to get a molecule by molecule neighborlist    
        gridGPULocal.periodicBoundaryConditions(-1,true);

        CUT_CHECK_ERROR("gridGPULocal.pbc() failed in handleLocalData!");
        // calls a kernel that populates our waterIdxsGPU with current data
        // --- this has nothing to do with the positions; this should be called 
        //     every time 
        // grabbing molecule at idx
        updateMoleculeIdxsMap<<<NBLOCK(nMolecules),PERBLOCK>>>(nMolecules,
                                                               waterIdsGPU.data(),
                                                               waterIdxsGPU.data(),
                                                               gpdLocal.idToIdxs.d_data.data(),
                                                               gpdGlobal.idToIdxs.d_data.data());
        CUT_CHECK_ERROR("updateMoleculeIdxsMap failed in handleLocalData!");

        /* resize  arrays as needed */
        pairPairEnergies.resize(gridGPULocal.neighborlist.size());
        forces_b2a1.resize(gridGPULocal.neighborlist.size());
        forces_c2a1.resize(gridGPULocal.neighborlist.size());
        forces_b1a2.resize(gridGPULocal.neighborlist.size());
        forces_c1a2.resize(gridGPULocal.neighborlist.size());
        computeThis.resize(gridGPULocal.neighborlist.size());

        /* memset to zero */
        pairPairEnergies.memset(0);
        forces_b2a1.memset(0);
        forces_c2a1.memset(0);
        forces_b1a2.memset(0);
        forces_c1a2.memset(0);
        pairPairTotal.memset(0);
        computeThis.memset(0);

        // pairPairTotal will have size nMolecules (static through course of simulation)
        // computeThis has size gridGPULocal.neighborlist.size()
        CUT_CHECK_ERROR("reallocation and memset of pairPairEnergies, pairPairForces failed in handleLocalData!");
    }
}

// nothing to do here, actually
bool FixE3B::stepInit(){
    return true;
}

/* Single Point Eng
 *
 *
 *
 */
void FixE3B::singlePointEng(real *perParticleEng) {
    
    int activeIdx = gpdLocal.activeIdx();
    int warpSize = state->devManager.prop.warpSize;
    GPUData &gpdGlobal = state->gpd;
    int globalActiveIdx = gpdGlobal.activeIdx();
    bool multiThreadPerAtom = state->nThreadPerAtom > 1 ? true : false;
    int NTPB = state->nThreadPerBlock;
    int NTPA = state->nThreadPerAtom;
    int nPerRingPoly = state->nPerRingPoly;
    BoundsGPU &bounds = state->boundsGPU;
    size_t smem_twobody_energy    = NTPA > 1 ?  NTPB * (sizeof(real3) + sizeof(real)) : 0 ;
    size_t smem_threebody_energy  = NTPA > 1 ?  NTPB * sizeof(real3) : 0;
    if (multiThreadPerAtom) {
        compute_E3B_energy_twobody<true><<<NBLOCKTEAM(nMolecules,NTPB,NTPA),NTPB,smem_twobody_energy>>>(nMolecules,              // nMolecules in E3B potential
                                                                                                                                           nPerRingPoly,
                            waterIdxsGPU.data(),                      // atomIdxs for molecule idx
                            gpdGlobal.xs(globalActiveIdx),            // as atom idxs 
                            gridGPULocal.perAtomArray.d_data.data(),  // neighbor counts for molecule idx
                            gridGPULocal.neighborlist.data(),         // neighbor idx for molecule idx
                            gridGPULocal.perBlockArray.d_data.data(), // cumulSumMaxPerBlock
                            warpSize,
                            bounds,
                            pairPairTotal.data(),
                            pairPairEnergies.data(),
                            computeThis.data(),
                            perParticleEng,                           // as atom idxs
                            NTPA,
                            evaluator);                               // assumes prepareForRun has been called
        
        compute_E3B_energy_threebody<true><<<NBLOCKTEAM(nMolecules,NTPB,NTPA),NTPB,smem_threebody_energy>>>(nMolecules,              // nMolecules in E3B potential
                                                                                                                                           nPerRingPoly,
                            waterIdxsGPU.data(),                      // atomIdxs for molecule idx
                            gpdGlobal.xs(globalActiveIdx),            // as atom idxs 
                            gridGPULocal.perAtomArray.d_data.data(),  // neighbor counts for molecule idx
                            gridGPULocal.neighborlist.data(),         // neighbor idx for molecule idx
                            gridGPULocal.perBlockArray.d_data.data(), // cumulSumMaxPerBlock
                            warpSize,
                            pairPairTotal.data(),
                            pairPairEnergies.data(),
                            computeThis.data(),
                            perParticleEng,                           // as atom idxs
                            NTPA,
                            evaluator);                               // assumes prepareForRun has been called
    } else {
        compute_E3B_energy_twobody<false><<<NBLOCKTEAM(nMolecules,NTPB,NTPA),NTPB,smem_twobody_energy>>>(nMolecules,              // nMolecules in E3B potential
                                                                                                                                           nPerRingPoly,
                            waterIdxsGPU.data(),                      // atomIdxs for molecule idx
                            gpdGlobal.xs(globalActiveIdx),            // as atom idxs 
                            gridGPULocal.perAtomArray.d_data.data(),  // neighbor counts for molecule idx
                            gridGPULocal.neighborlist.data(),         // neighbor idx for molecule idx
                            gridGPULocal.perBlockArray.d_data.data(), // cumulSumMaxPerBlock
                            warpSize,
                            bounds,
                            pairPairTotal.data(),
                            pairPairEnergies.data(),
                            computeThis.data(),
                            perParticleEng,                           // as atom idxs
                            NTPA,
                            evaluator);                               // assumes prepareForRun has been called
        compute_E3B_energy_threebody<false><<<NBLOCKTEAM(nMolecules,NTPB,NTPA),NTPB,smem_threebody_energy>>>(nMolecules,              // nMolecules in E3B potential
                                                                                                                                           nPerRingPoly,
                            waterIdxsGPU.data(),                      // atomIdxs for molecule idx
                            gpdGlobal.xs(globalActiveIdx),            // as atom idxs 
                            gridGPULocal.perAtomArray.d_data.data(),  // neighbor counts for molecule idx
                            gridGPULocal.neighborlist.data(),         // neighbor idx for molecule idx
                            gridGPULocal.perBlockArray.d_data.data(), // cumulSumMaxPerBlock
                            warpSize,
                            pairPairTotal.data(),
                            pairPairEnergies.data(),
                            computeThis.data(),
                            perParticleEng,                           // as atom idxs
                            NTPA,
                            evaluator);                               // assumes prepareForRun has been called
    }

    CUT_CHECK_ERROR("compute_E3B_energy failed.\n");
    pairPairTotal.memset(0);
    pairPairEnergies.memset(0);
    computeThis.memset(0);
    return;
}

size_t FixE3B::getSmemRequired(int ntpb, int ntpa) {

    // we'll be conservative here and assume we need virials as well.
    //size_t smem_twobody_force   = ntpa > 1 ? ntpb * (2 * sizeof(real3) + sizeof(Virial))  : 0 ;
    
    // so, the kernels are launched separately, and we either need 0 for both, 
    // or smem_threebody_force has an unambiguously larger smem requirement.
    // so, no need to compare the two or even compute the twobody requirement
    size_t smem_threebody_force = ntpa > 1 ? ntpb * (3 * (sizeof(real3) + sizeof(Virial))): 0 ;

    return smem_threebody_force;

}

void FixE3B::createEvaluator() {
    
    // style defaults to E3B3; otherwise, it can be set to E3B2;
    // there are no other options.
    real kjToKcal = 0.23900573614;
    real rs = 5.0;
    real rf = 5.2;
    real k2 = 4.872;
    real k3 = 1.907;
    real E2, Ea, Eb, Ec;
    // default
        // TODO: add check verifying that molecule is attached to FixRigid with style TIP4P/2005!
    if (style == "E3B3") {
        // as angstroms

        // E2, Ea, Eb, Ec as kJ/mole -> convert to kcal/mole
        E2 = 453000.0;
        Ea = 150.0000;
        Eb = -1005.0000;
        Ec = 1880.0000;

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
        // TODO: add check verifying that molecule is attached to FixRigid with style TIP4P!
    } else if (style == "E3B2") {
    
        E2 = 2349000.0; // kj/mol
        Ea = 1745.7;
        Eb = -4565.0;
        Ec = 7606.8;

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

    std::cout << "Created E3B Evaluator:\n" << std::endl;
    std::cout << "Ea: " << Ea << std::endl;
    std::cout << "Eb: " << Eb << std::endl;
    std::cout << "Ec: " << Ec << std::endl;
    std::cout << "E2: " << E2 << std::endl;
    std::cout << "rs: " << rs << std::endl;
    std::cout << "rf: " << rf << std::endl;
    std::cout << "k2: " << k2 << std::endl;
    std::cout << "k3: " << k3 << std::endl;
};


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
    
    /* Setting the GridGPU, where we do a molecular neighborlist */
    double maxRCut = rf;// cutoff of our potential (5.2 A)
    double gridDim = maxRCut + padding; // padding is defined in the ctor

    // this number has no meaning whatsoever; it is completely arbitrary;
    // -- we are not using exclusionMode for this grid or set of GPUData
    int exclusionMode = 30;
    // I think this is doubly irrelevant, since we use a doExclusions(false) method later (below)

    gridGPULocal = GridGPU(state, gridDim, gridDim, gridDim, gridDim, exclusionMode, padding, &gpdLocal,state->nPerRingPoly,false); // if we have ringpolymers for atoms, we similarly have ring polymers for molecules..
    // ok; after it gets back from its constructor, manually set our nThreadPerBlock to 
    gridGPULocal.nThreadPerBlock(state->nThreadPerBlock);
    gridGPULocal.nThreadPerAtom(state->nThreadPerAtom); 
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
    // --- gpd.xs() should be up-to-date when we call gridGPULocal.periodicBoundaryConditions() above
    CUT_CHECK_ERROR("gridGPULocal.pbc() in prepareForRun failed!");

    waterIdxsGPU = GPUArrayDeviceGlobal<int4>(nMolecules);
    waterIdxsGPU.set(waterIds.data()); // initialize as waterIds, then update using idToIdxs maps
    
    pairPairTotal    = GPUArrayDeviceGlobal<real4>(nMolecules);
    pairPairEnergies = GPUArrayDeviceGlobal<real4>(gridGPULocal.neighborlist.size());
    forces_b2a1      = GPUArrayDeviceGlobal<real4>(gridGPULocal.neighborlist.size());
    forces_c2a1      = GPUArrayDeviceGlobal<real4>(gridGPULocal.neighborlist.size());
    forces_b1a2      = GPUArrayDeviceGlobal<real4>(gridGPULocal.neighborlist.size());
    forces_c1a2      = GPUArrayDeviceGlobal<real4>(gridGPULocal.neighborlist.size());
    computeThis      = GPUArrayDeviceGlobal<uint>(gridGPULocal.neighborlist.size());
    pairPairTotal.memset(0);
    pairPairEnergies.memset(0);
    forces_b2a1.memset(0);
    forces_c2a1.memset(0);
    forces_b1a2.memset(0);
    forces_c1a2.memset(0);
    computeThis.memset(0);   
    
    // everything here is prepared; set it true, THEN:  handleLocalData(), which needs to know that 
    // prepared == true; then, return prepared.
    prepared = true;

    CUT_CHECK_ERROR("memset on pairPairEnergies in prepareForRun failed!");
    handleLocalData(); // on having a changed idToIdx, for either molecules or atoms, this updates the 
    // int4 atomIdxs stored in waterIdxsGPU
    CUT_CHECK_ERROR("handleLocalData in prepareForRun failed!");
    return prepared;
}

void FixE3B::takeStateNThreadPerBlock(int NTPB) {
    // E3B maintains a constant value of threadsPerBlock.
    if (prepared) {
        //nThreadPerBlock(NTPB);
        gridGPULocal.nThreadPerBlock(NTPB);
    }
}

void FixE3B::takeStateNThreadPerAtom(int NTPA) {
    // E3B maintains a constant value of threadsPerAtom (molecule, really)
    if (prepared) {
        //nThreadPerAtom(NTPA);
        gridGPULocal.nThreadPerAtom(NTPA);
        // ok; compare oldNThreadPerAtom, oldNThreadPerBlock with nThreadPerBlock;
        // if they changed, call initArraysTune(), and then PBC
        gridGPULocal.initArraysTune();
        gridGPULocal.periodicBoundaryConditions(-1,true);
        handleLocalData();
    }
};


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


void export_FixE3B() {
  py::class_<FixE3B, boost::shared_ptr<FixE3B>, py::bases<Fix> > 
	("FixE3B",
         py::init<boost::shared_ptr<State>, std::string, std::string> 
	 (py::args("state", "handle", "style")
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
    .def("handleLocalData", &FixE3B::handleLocalData)
    // the list of molecules
    .def_readonly("molecules", &FixE3B::waterMolecules)
    .def_readonly("nMolecules", &FixE3B::nMolecules)
    .def_readonly("gridGPU", &FixE3B::gridGPULocal)
    .def_readwrite("prepared", &FixE3B::prepared)
    ;
}
