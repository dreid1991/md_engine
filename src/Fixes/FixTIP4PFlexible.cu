#include "FixTIP4PFlexible.h"
#include "State.h"
#include "VariantPyListInterface.h"
#include "boost_for_export.h"
#include "cutils_math.h"
#include "cutils_func.h"
#include <math.h>
#include "globalDefs.h"
#include "Vector.h"

namespace py = boost::python;

const std::string TIP4PFlexibleType = "TIP4PFlexible";

FixTIP4PFlexible::FixTIP4PFlexible(boost::shared_ptr<State> state_, std::string handle_, std::string groupHandle_) : Fix(state_, handle_, groupHandle_, TIP4PFlexibleType, true, true, false, 1) {

    // set both to false initially; using one of the createRigid functions will flip the pertinent flag to true
    firstPrepare = true;
    style = "DEFAULT";
    
    // set to default values of zero
    rOM = 0.0;
    rHH = 0.0;
    rOH = 0.0;
    theta = 0.0;
}

__global__ void printGPD_Flexible(int4 *waterIds, int* idToIdxs, float4 *xs, float4 *vs, float4 *fs, int nMolecules) {
    int idx = GETIDX();
    
    // print 5 molecules per turn
    if (idx < 5) {
        int4 theseAtoms = waterIds[idx];

        int idO = theseAtoms.x;
        int idH1 = theseAtoms.y;
        int idH2 = theseAtoms.z;
        int idM = theseAtoms.w;

        int idx_O = idToIdxs[theseAtoms.x];
        int idx_H1= idToIdxs[theseAtoms.y];
        int idx_H2= idToIdxs[theseAtoms.z];
        int idx_M = idToIdxs[theseAtoms.w];

        float4 pos_O = xs[idx_O];
        float4 pos_H1= xs[idx_H1];
        float4 pos_H2= xs[idx_H2];
        float4 pos_M = xs[idx_M];

        float4 vel_O = vs[idx_O];
        float4 vel_H1= vs[idx_H1];
        float4 vel_H2= vs[idx_H2];
        float4 vel_M = vs[idx_M];

        float4 force_O = fs[idx_O];
        float4 force_H1= fs[idx_H1];
        float4 force_H2= fs[idx_H2];
        float4 force_M = fs[idx_M];

        printf("\natoms O, H1, H2, M ids %d %d %d %d\n     pos_O %f %f %f\npos_H1 %f %f %f\npos_H2 %f %f %f\npos_M %f %f %f\n",
               idO, idH1, idH2, idM,
               pos_O.x, pos_O.y, pos_O.z,
               pos_H1.x, pos_H1.y, pos_H1.z,
               pos_H2.x, pos_H2.y, pos_H2.z,
               pos_M.x, pos_M.y, pos_M.z);
               
        printf("\n atoms O, H1, H2, M ids %d %d %d %d\nvel_O %f %f %f %f\nvel_H1 %f %f %f %f\nvel_H2 %f %f %f %f \nvel_M %f %f %f %f\n",
               idO, idH1, idH2, idM,
               vel_O.x, vel_O.y, vel_O.z, vel_O.w,
               vel_H1.x, vel_H1.y, vel_H1.z, vel_H1.w,
               vel_H2.x, vel_H2.y, vel_H2.z, vel_H2.w,
               vel_M.x, vel_M.y, vel_M.z, vel_M.w);

        printf("\natoms O, H1, H2, M ids %d %d %d %d\nfs_O %f %f %f %d\nfs_H1 %f %f %f %d\nfs_H2 %f %f %f %d\nfs_M %f %f %f %d\n\n", 
               idO, idH1, idH2, idM,
               force_O.x, force_O.y, force_O.z, (uint) force_O.w,
               force_H1.x, force_H1.y, force_H1.z, (uint) force_H1.w,
               force_H2.x, force_H2.y, force_H2.z, (uint) force_H2.w,
               force_M.x, force_M.y, force_M.z, (uint) force_M.w);

    
    }
}

// distribute the m site forces, and do an unconstrained integration of the velocity component corresponding to this additional force
// -- this is required for 4-site models with a massless particle.
//    see compute_gamma() function for details.
template <bool VIRIALS>
__global__ void distributeMSiteFlexible(int4 *waterIds, float4 *xs, float4 *vs, float4 *fs, 
                                Virial *virials,
                                int nMolecules, float gamma, float dtf, int* idToIdxs, BoundsGPU bounds)

{
    int idx = GETIDX();
    if (idx < nMolecules) {
        
        int id_O  = waterIds[idx].x;
        int id_H1 = waterIds[idx].y;
        int id_H2 = waterIds[idx].z;
        int id_M  = waterIds[idx].w;

        int idx_O = idToIdxs[id_O];
        int idx_H1 = idToIdxs[id_H1];
        int idx_H2 = idToIdxs[id_H2];
        int idx_M = idToIdxs[id_M];

        float4 vel_O = vs[idx_O];
        float4 vel_H1 = vs[idx_H1];
        float4 vel_H2 = vs[idx_H2];

        //printf("In distributeMSite, velocity of Oxygen %d is %f %f %f\n", id_O, vel_O.x, vel_O.y, vel_O.z);
        // need the forces from O, H1, H2, and M
        float4 fs_O  = fs[idx_O];
        float4 fs_H1 = fs[idx_H1];
        float4 fs_H2 = fs[idx_H2];
        float4 fs_M  = fs[idx_M];

        //printf("Force on m site: %f %f %f\n", fs_M.x, fs_M.y, fs_M.z);
        //printf("Force on Oxygen : %f %f %f\n", fs_O.x, fs_M.y, fs_M.z);
        // now, get the partial force contributions from the M-site; prior to adding these to the
        // array of forces for the given atom, integrate the velocity of the atom according to the distributed force contribution

        // this expression derived below in FixTIP4PFlexible::compute_gamma() function
        // -- these are the forces from the M-site partitioned for distribution to the atoms of the water molecule
        float3 fs_O_d = make_float3(fs_M) * (1.0 - (2.0 * gamma));
        float3 fs_H_d = make_float3(fs_M) * gamma;

        // get the inverse masses from velocity variables above
        float invMassO = vel_O.w;

        // if the hydrogens don't have equivalent masses, we have bigger problems
        float invMassH = vel_H1.w;

        // compute the differential addition to the velocities
        float3 dv_O = dtf * invMassO * fs_O_d;
        float3 dv_H = dtf * invMassH * fs_H_d;

        // and add to the velocities of the atoms
        vel_O  += dv_O;
        vel_H1 += dv_H;
        vel_H2 += dv_H;

        // set the velocities to the new velocities in vel_O, vel_H1, vel_H2
        vs[idToIdxs[id_O]] = vel_O; 
        vs[idToIdxs[id_H1]]= vel_H1;
        vs[idToIdxs[id_H2]]= vel_H2;
        
        // is this correct? -- TODO check LAMMPS method of distributing per-atom virials with massless site & constrained dynamics
        if (VIRIALS) {
            Virial virialToDistribute = virials[idx_M];
        
            Virial distribute_O = virialToDistribute * (1.0 - (2.0 * gamma));
            Virial distribute_H = virialToDistribute * gamma;
            
            virials[idx_O]  += distribute_O;
            virials[idx_H1] += distribute_H;
            virials[idx_H2] += distribute_H;

            virials[idx_M] = Virial(0.0, 0.0, 0.0, 
                                    0.0, 0.0, 0.0);
        };
        
        vs[idx_M] = make_float4(0.0, 0.0, 0.0, INVMASSLESS);
        // finally, modify the forces; this way, the distributed force from M-site is incorporated in to nve_v() integration step
        // at beginning of next iteration in IntegratorVerlet.cu
        fs_O += fs_O_d;
        fs_H1 += fs_H_d;
        fs_H2 += fs_H_d;
       
        // set the global variables *fs[idToIdx[id]] to the new values
        fs[idx_O] = fs_O;
        fs[idx_H1]= fs_H1;
        fs[idx_H2]= fs_H2;

        // zero the force on the M-site
        fs[idx_M] = make_float4(0.0, 0.0, 0.0,fs_M.w);
        vs[idx_M] = make_float4(0.0, 0.0, 0.0, INVMASSLESS);
        // this concludes re-distribution of the forces;
        // we assume nothing needs to be done re: virials; this sum is already tabulated at inner force loop computation
        // in the evaluators; for safety, we might just set 

    }
}

__global__ void setMSiteFlexible(int4 *waterIds, int *idToIdxs, float4 *xs, int nMolecules, double rOM, BoundsGPU bounds) {

    int idx = GETIDX();
    if (idx < nMolecules) {
    
        /* What we do here:
         * get the minimum image positions of the O, H, H atoms
         * compute the vector position of the M site
         * apply PBC to this new position (in case the water happens to be on the boundary of the box
         */

        // first, get the ids of the atoms composing this molecule
        int id_O  = waterIds[idx].x;
        int id_H1 = waterIds[idx].y;
        int id_H2 = waterIds[idx].z;
        int id_M  = waterIds[idx].w;

        float4 pos_M_whole = xs[idToIdxs[id_M]];
        // get the positions of said atoms
        float3 pos_O = make_float3(xs[idToIdxs[id_O]]);
        float3 pos_H1= make_float3(xs[idToIdxs[id_H1]]);
        float3 pos_H2= make_float3(xs[idToIdxs[id_H2]]);
        float3 pos_M = make_float3(xs[idToIdxs[id_M]]);

        // compute vectors r_ij and r_ik according to minimum image convention
        // where r_ij = r_j - r_i, r_ik = r_k - r_i,
        // and r_i, r_j, r_k are the 3-component vectors describing the positions of O, H1, H2, respectively
        float3 r_ij = bounds.minImage( (pos_H1 - pos_O) );
        float3 r_ik = bounds.minImage( (pos_H2 - pos_O) );

        // rOM is the O-M bond length
        float3 r_M  = (pos_O) + (rOM * ( (r_ij + r_ik) / ( (length(r_ij + r_ik)))));
        //printf("new position of M-site molecule %d r_M: %f %f %f\n      position of oxygen %d: %f %f %f\n", idx, r_M.x, r_M.y, r_M.z, id_O, pos_O.x, pos_O.y, pos_O.z);
        float4 pos_M_new = make_float4(r_M.x, r_M.y, r_M.z, pos_M_whole.w);
        xs[idToIdxs[id_M]] = pos_M_new;
    }
}

template <bool VIRIALS>
__global__ void initialForcePartitionFlexible(int4 *waterIds, float4 *xs, float4 *fs, 
                                      Virial *virials, int nMolecules, float gamma,
                                      int *idToIdxs, BoundsGPU bounds) {

    // we assume the M-site is located in its proper location at this point (initialization of the system)

    int idx = GETIDX();
    if (idx < nMolecules) {

        // by construction, the id's of the molecules are ordered as follows in waterIds array

        int id_O  = waterIds[idx].x;
        int id_H1 = waterIds[idx].y;
        int id_H2 = waterIds[idx].z;
        int id_M  = waterIds[idx].w;
        
        int idx_O = idToIdxs[id_O];
        int idx_H1= idToIdxs[id_H1];
        int idx_H2= idToIdxs[id_H2];
        int idx_M = idToIdxs[id_M];
        // need the forces from O, H1, H2, and M
        float4 fs_O  = fs[idx_O];
        float4 fs_H1 = fs[idx_H1];
        float4 fs_H2 = fs[idx_H2];
        float4 fs_M  = fs[idx_M];

        //printf("Force on m site: %f %f %f\n", fs_M.x, fs_M.y, fs_M.z);
        //printf("Force on Oxygen : %f %f %f\n", fs_O.x, fs_M.y, fs_M.z);
        // now, get the partial force contributions from the M-site; prior to adding these to the
        // array of forces for the given atom, integrate the velocity of the atom according to the distributed force contribution

        // this expression derived below in FixTIP4PFlexible::compute_gamma() function
        // -- these are the forces from the M-site partitioned for distribution to the atoms of the water molecule
        float3 fs_O_d = (make_float3(fs_M)) * (1.0 - (2.0 * gamma));
        printf("value of fs_O_d from atom M id %d: %f %f %f\n", waterIds[idx].w, fs_O_d.x, fs_O_d.y, fs_O_d.z);
        float3 fs_H_d = (make_float3(fs_M)) * gamma;

        if (VIRIALS) {
            Virial virialToDistribute = virials[idToIdxs[id_M]];
        
            Virial distribute_O = virialToDistribute * (1.0 - (2.0 * gamma));
            Virial distribute_H = virialToDistribute * gamma;
            
            virials[idToIdxs[id_O]] += distribute_O;
            virials[idToIdxs[id_H1]] += distribute_H;
            virials[idToIdxs[id_H2]] += distribute_H;

            virials[idToIdxs[id_M]] = Virial(0., 0., 0., 
                                             0., 0., 0.);
        }
        
        // finally, modify the forces; this way, the distributed force from M-site is incorporated in to nve_v() integration step
        // at beginning of next iteration in IntegratorVerlet.cu
        fs_O += fs_O_d;
        fs_H1 += fs_H_d;
        fs_H2 += fs_H_d;
       
        // set the global variables *fs[idToIdx[id]] to the new values
        fs[idx_O] = fs_O;
        fs[idx_H1]= fs_H1;
        fs[idx_H2]= fs_H2;

        // zero the force on the M-site, just because
        fs[idx_M] = make_float4(0.0, 0.0, 0.0,fs_M.w);
        // this concludes re-distribution of the forces;
    }
}

void FixTIP4PFlexible::setStyle(std::string style_) {
    
    if (style_ == "TIP4P/2005") {
        style = style_;
    } else if (style_ == "q-TIP4P/F") {
        style = style_;
    } else {
        mdError("Unknown argument to FixTIP4PFlexible::style() command\nSupported arguments are \"TIP4P/2005\" and \"q-TIP4P/F\"\n");
    }
    return;
}

void FixTIP4PFlexible::setStyleBondLengths() {
    
    if ( (style == "TIP4P/2005") ) {
        // set to real units, TIP4P/2005 (Angstroms)
        rOM = 0.15460000000;
        rOH = 0.95720000000;
        theta = 1.82421813;
        rHH = 2.0 * rOH * sin(0.5*theta);
    } else if ( (style == "DEFAULT") or (style == "q-TIP4P/F")) {
        rOM = 0.147144032; // calculated separately, as (1.0 - gamma) * ( (h1Pos + h2Pos) / 2), gamma = 0.73612 in the paper
        rOH = 0.9419; // given directly in the paper
        theta = 1.8744836; // given directly in the paper
        rHH = 2.0 * rOH * sin(0.5*theta);
        
    } else {
        // other models here; give appropriate keywords
        if ( (rOM == 0.0) or (rOH == 0.0) or (rHH == 0.0)) {
            mdError("Only DEFAULT and TIP4P/2005 keywords currently supported.\nPlease manually set the rOM, rOH, and rHH values if TIP4P/2005 geometry is not desired.\n");

        }
    }
    return;
}

void FixTIP4PFlexible::handleBoundsChange() {


    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();
    BoundsGPU &bounds = state->boundsGPU;

    // get a few pieces of data as required
    // -- all we're doing here is setting the position of the M-Site prior to computing the forces
    //    within the simulation.  Otherwise, the M-site will be out of its prescribed position.
    // we need to reset the position of the M-Site prior to calculating the forces
    
    setMSiteFlexible<<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.idToIdxs.d_data.data(), gpd.xs(activeIdx), nMolecules, rOM,  bounds);

    return;


}

void FixTIP4PFlexible::updateForPIMD(int nPerRingPoly) {

    // see State.cpp... 
    // at this point, we have all of our molecules added via ::addMolecule(*args)
    nMolecules = waterIds.size() * nPerRingPoly; // this is the total number of beads for PIMD
    
    // note that this is implemented /before/ ::prepareForRun is called!
    


    // we have multiplied all instances of a given atom in the simulation by nPerRingPoly,
    // --- so, now there are nPerRingPoly copies of, e.g., oxygen, before we encounter  the hydrogen, etc.
  
    // so, make a new vector of int4's, and do the same process, and then reset our data structures.
    std::vector<int4> PIMD_waterIds;

    // duplicate each molecule by nPerRingPoly, and stride accordingly
    for (int i = 0; i < waterIds.size(); i++) {
        int4 thisMolecule = waterIds[i];
        int baseIdxO  = thisMolecule.x * nPerRingPoly;
        int baseIdxH1 = thisMolecule.y * nPerRingPoly;
        int baseIdxH2 = thisMolecule.z * nPerRingPoly;
        int baseIdxM  = thisMolecule.w * nPerRingPoly;

        // make nPerRingPoly replicas of this molecule with atom ids
        for (int j = 0; j < nPerRingPoly; j++) {
            int idO_PIMD  = baseIdxO  * nPerRingPoly + j;
            int idH1_PIMD = baseIdxH1 * nPerRingPoly + j;
            int idH2_PIMD = baseIdxH2 * nPerRingPoly + j;
            int idM_PIMD  = baseIdxM  * nPerRingPoly + j;
            int4 newMol = make_int4(idO_PIMD, idH1_PIMD, idH2_PIMD, idM_PIMD);
            PIMD_waterIds.push_back(newMol);
    
        }
    }

    // and now just re-assign waterIds vector; prepareForRun will function as usual
    waterIds = PIMD_waterIds;

    return;

}

void FixTIP4PFlexible::compute_gamma() {

    /*  See Feenstra, Hess, and Berendsen, J. Computational Chemistry, 
     *  Vol. 20, No. 8, 786-798 (1999)
     *
     *  From Appendix A, we see the expression: 
     *  $\mathbf{F}_{ix}^' = \frac{\partial \mathbf{r}_d}{\partial x_i} \cdot \mathbf{F}_d
     *
     *  Moreover, the position of the dummy atom (see, e.g., construction of TIP4P molecule in 
     *  (relative path here) ../../util_py/water.py file) can be written in terms of O,H,H positions
     *
     *  Taking the position of the oxygen as the center, we denote Oxygen as atom 'i',
     *  and the two hydrogens as 'j' and 'k', respectively
     * 
     *  Then, we have the following expression for r_d:
     * 
     *  (Expression 1)
     *  r_d = r_i + 0.1546 * ((r_ij + r_ik) / ( len(r_ij + r_ik)))
     *
     *  Then, rearranging,
     *  
     *  (Expression 2)
     *  r_d = r_i + 0.1546 * ( (r_j + r_k - 2 * r_i) / (len(r_j + r_k - 2 * r_i)))
     * 
     *  So, gamma is then
     *  
     *  (Expression 3)
     *  gamma = 0.1546 / len(r_j + r_k - 2 * r_i)
     *
     *  And force is partitioned according to:
     *
     *  (Expression 4)
     *  F_i^' = (1 - 2.0 * gamma) F_d
     *  F_j^' = F_k^' = gamma * F_d
     * 
     *  which we get from straightforward differentiation of the re-arranged positions in Expression 2 above.
     */

    /*
    // get the bounds
    BoundsGPU &bounds = state->boundsGPU;

    
    // grab any water molecule; the first will do
    // -- at this point, the waterIds array is set up
    int4 waterMolecule = waterIds[0];

    // ids in waterMolecule are (.x, .y, .z, .w) ~ (O, H1, H2, M)
    Vector r_i_v = state->idToAtom(waterMolecule.x).pos;
    Vector r_j_v = state->idToAtom(waterMolecule.y).pos;
    Vector r_k_v = state->idToAtom(waterMolecule.z).pos;

    // cast above vectors as float3 for use in bounds.minImage

    float3 r_i = make_float3(r_i_v[0], r_i_v[1], r_i_v[2]);
    float3 r_j = make_float3(r_j_v[0], r_j_v[1], r_j_v[2]);
    float3 r_k = make_float3(r_k_v[0], r_k_v[1], r_k_v[2]);
    
    // get the minimum image vectors r_ij, r_ikz
    float3 r_ij = bounds.minImage(r_j - r_i);
    float3 r_ik = bounds.minImage(r_k - r_i);

    // denominator of expression 3, written using the minimum image
    float denominator = length( ( r_ij + r_ik) );
    */

    // so take it that a fictitious oxygen is at the origin (0,0,0)
    double phi = (M_PI - theta) * 0.5;
    Vector H1Pos = Vector(cos(phi), sin(phi), 0.);
    phi += theta;
    Vector H2Pos = Vector(cos(phi), sin(phi), 0.);

    double denominator = (H1Pos + H2Pos).len();
    //
    gamma = (float) (rOM / denominator);
    printf("in FixTIP4PFlexible::compute_gamma(): computed a gamma of %f\n", gamma);
    return;
}

// id's must be arranged as O, H1, H2, M
void FixTIP4PFlexible::addMolecule(int id_a, int id_b, int id_c, int id_d) {
    
    int4 waterMol = make_int4(0,0,0,0);

    // grab the positions associated with a given atom
    Vector a = state->idToAtom(id_a).pos;
    Vector b = state->idToAtom(id_b).pos;
    Vector c = state->idToAtom(id_c).pos;
    Vector d = state->idToAtom(id_d).pos;

    // likewise, grab their masses
    double ma = state->idToAtom(id_a).mass;
    double mb = state->idToAtom(id_b).mass;
    double mc = state->idToAtom(id_c).mass;
    double md = state->idToAtom(id_d).mass;

    bool ordered = true;
    if (! (ma > mb && ma > mc)) ordered = false;
    
    if (! (mb == mc) ) ordered = false;
    if (! (mb > md) )  ordered = false;
    
    if (! (ordered)) {
        printf("Found masses O, H, H, M in order: %f %f %f %f\n", ma, mb, mc, md);
    }
    if (! (ordered)) mdError("Ids in FixTIP4PFlexible::createRigid must be as O, H1, H2, M");
    
    waterMol = make_int4(id_a, id_b, id_c, id_d);
    waterIds.push_back(waterMol);

    Bond bondOH1;
    Bond bondOH2;
    Bond bondHH;
    bondOH1.ids = { {waterMol.x,waterMol.y} };
    bondOH2.ids = { {waterMol.x,waterMol.z} };
    bondHH.ids =  { {waterMol.y,waterMol.z} };
    bonds.push_back(bondOH1);
    bonds.push_back(bondOH2);
    bonds.push_back(bondHH);
}

bool FixTIP4PFlexible::prepareForRun() {
    // if this is the first time prepareForRun was called, flip the flag
    // -- we need the forces, which we subsequently partition
    if (firstPrepare) {
        printf("returning from first call to prepareForRun in FixTIP4PFlexible!\n");
        firstPrepare = false;
        return false;
    }

    // then we'll know if we need to partition forces prior to integration
    nMolecules = waterIds.size();
    printf("Found %d molecules in FixTIP4PFlexible::prepareForRun()\n", nMolecules);
    setStyleBondLengths();

    waterIdsGPU = GPUArrayDeviceGlobal<int4>(nMolecules);
    waterIdsGPU.set(waterIds.data());

    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();
    BoundsGPU &bounds = state->boundsGPU;

    // compute the force partition constant
    compute_gamma();
    
    //printf("printing data prior to partitioning in FixTIP4PFlexible::prepareForRun\n");
    //cudaDeviceSynchronize();
    //printGPD_Flexible<<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(),
    //                                                    gpd.idToIdxs.d_data.data(), gpd.xs(activeIdx),
    //                                                    gpd.vs(activeIdx),  gpd.fs(activeIdx),
    //                                                    nMolecules);
    //
    //
    //cudaDeviceSynchronize();
    //printf("END: printing data prior to partitioning in FixTIP4PFlexible::prepareForRun\n");
    

    // partition the intitial forces ( no velocity update )
    initialForcePartitionFlexible<false><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), 
                                                            gpd.xs(activeIdx),
                                                            gpd.fs(activeIdx),
                                                            gpd.virials.d_data.data(),
                                                            nMolecules, gamma, gpd.idToIdxs.d_data.data(), bounds);
    

    //printf("Finished the initial force partition in FixTIP4PFlexible!\n");
    //cudaDeviceSynchronize();
    //printGPD_Flexible<<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(),
    //                                                    gpd.idToIdxs.d_data.data(), gpd.xs(activeIdx),
    //                                                          gpd.vs(activeIdx),  gpd.fs(activeIdx),
    //                                                          nMolecules);
    //cudaDeviceSynchronize();
    return true;
}


// distribute the forces from the M-site, and increment the velocities of the affected particles s.t. the half-step
// velocity update includes the forces from the M-site
bool FixTIP4PFlexible::stepFinal() {
    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();
    BoundsGPU &bounds = state->boundsGPU;
    // first, unconstrained velocity update continues: distribute the force from the M-site
    //        and integrate the velocities accordingly.  Update the forces as well.

    // TODO: correct M-site partitioning of the virials.  likely need to take a look at the change of variables occurring
    bool Virials = false;

    float dtf = 0.5f * state->dt * state->units.ftm_to_v;
    if (Virials) {
        distributeMSiteFlexible<true><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                                     gpd.vs(activeIdx),  gpd.fs(activeIdx),
                                                     gpd.virials.d_data.data(),
                                                     nMolecules, gamma, dtf, gpd.idToIdxs.d_data.data(), bounds);
    } else {
        distributeMSiteFlexible<false><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                                     gpd.vs(activeIdx),  gpd.fs(activeIdx),
                                                     gpd.virials.d_data.data(),
                                                     nMolecules, gamma, dtf, gpd.idToIdxs.d_data.data(), bounds);
    }

    //setMSite<<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.idToIdxs.d_data.data(), gpd.xs(activeIdx), nMolecules, bounds);
    
    //cudaDeviceSynchronize();
    //printGPD_Flexible<<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), 
    //                                                    gpd.idToIdxs.d_data.data(), gpd.xs(activeIdx),
    //                                                          gpd.vs(activeIdx),  gpd.fs(activeIdx),
    //                                                          nMolecules);
    //cudaDeviceSynchronize();
    //printf("Finished TIP4PFlexible::stepFinal at turn %d!\n", (int) state->turn);
    return true;
}

void (FixTIP4PFlexible::*addMolecule_x) (int, int, int, int) = &FixTIP4PFlexible::addMolecule;

void export_FixTIP4PFlexible() 
{
  py::class_<FixTIP4PFlexible, boost::shared_ptr<FixTIP4PFlexible>, py::bases<Fix> > 
      ( 
		"FixTIP4PFlexible",
		py::init<boost::shared_ptr<State>, std::string, std::string>
	    (py::args("state", "handle", "groupHandle")
         )
        )
    .def("addMolecule", addMolecule_x,
        (py::arg("id_a"), 
         py::arg("id_b"),  
         py::arg("id_c"), 
         py::arg("id_d"))
     )
    .def("setStyle", &FixTIP4PFlexible::setStyle,
         py::arg("style")
        )
    .def_readwrite("rOM", &FixTIP4PFlexible::rOM)
    .def_readwrite("rOH", &FixTIP4PFlexible::rOH)
    .def_readwrite("rHH", &FixTIP4PFlexible::rHH)
    ;
}



