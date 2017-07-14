#include "FixRigid.h"

#include "State.h"
#include "VariantPyListInterface.h"
#include "boost_for_export.h"
#include "cutils_math.h"
#include "cutils_func.h"
#include <math.h>
#include "globalDefs.h"
using namespace std;
namespace py = boost::python;
const string rigidType = "Rigid";

FixRigid::FixRigid(boost::shared_ptr<State> state_, string handle_, string groupHandle_) : Fix(state_, handle_, groupHandle_, rigidType, true, true, false, 1) {

    // set both to false initially; using one of the createRigid functions will flip the pertinent flag to true
    firstPrepare = true;
    TIP4P = false;
    TIP3P = false;
    printing = false;
    requiresPostNVE_V = true;
    style = "DEFAULT";
}

__device__ inline float3 positionsToCOM(float3 *pos, float *mass, float ims) {
  return (pos[0]*mass[0] + pos[1]*mass[1] + pos[2]*mass[2])*ims;
}

inline __host__ __device__ float3 rotateCoords(float3 vector, float3 matrix[]) {
  return make_float3(dot(matrix[0],vector),dot(matrix[1],vector),dot(matrix[2],vector));
}

// this function removes any residual velocities along the lengths of the bonds, 
// thus permitting the use of our initializeTemperature() method without violation of the constraints
// on initializing a simulation
__global__ void adjustInitialVelocities(int4* waterIds, int *idToIdxs, float4 *xs, float4 *vs, int nMolecules, BoundsGPU bounds) {

    int idx = GETIDX();
    
    // -- shared variable: COMV for this group! That way, we preserve it on initialization and don't have to worry about it.
    extern __shared__ float3 COMV[];

    // populate the COMV shared memory variable
    if (idx < nMolecules) {
        // get the atom ids
        int id_O = waterIds[idx].x;
        int id_H1 = waterIds[idx].y;
        int id_H2 = waterIds[idx].z;
        
        // get the atom velocities
        float4 vel_O_whole  =  vs[idToIdxs[id_O]];
        float4 vel_H1_whole =  vs[idToIdxs[id_H1]];
        float4 vel_H2_whole =  vs[idToIdxs[id_H2]];

        // get the atom masses
        float mass_O = 1.0f/vel_O_whole.w;
        float mass_H = 1.0f/vel_H1_whole.w;
        
        COMV[idx] = (mass_O * make_float3(vel_O_whole)) + 
                            (mass_H * make_float3(vel_H1_whole)) + 
                            (mass_H * make_float3(vel_H2_whole));
    }
    __syncthreads();

    unsigned int tid = threadIdx.x;
    //unsigned int i = blockIdx.x*(blockSize*2) + tid;
    //unsigned int gridSize = blockSize*2*gridDim.x;
    
    // do an in-place parallel reduction summation of COMV array, and make a local copy of the variable
    float3 total_COMV = make_float3(0.0f, 0.0f, 0.0f);
   
    // TODO:
    //
    // get the total_COMV value via parallel reduction
    //

    if (idx < nMolecules) {

        // get the atom ids
        int id_O = waterIds[idx].x;
        int id_H1 = waterIds[idx].y;
        int id_H2 = waterIds[idx].z;
        
        // get the atom velocities
        float4 vel_O_whole  =  vs[idToIdxs[id_O]];
        float4 vel_H1_whole =  vs[idToIdxs[id_H1]];
        float4 vel_H2_whole =  vs[idToIdxs[id_H2]];

        // get the atom masses
        float mass_O = 1.0f/vel_O_whole.w;
        float mass_H = 1.0f/vel_H1_whole.w;
        // get the kinetic energies of each atom in the molecule - so first get magnitude of velocities
        float v_O =  length(make_float3(vel_O_whole));
        float v_H1 = length(make_float3(vel_H1_whole));
        float v_H2 = length(make_float3(vel_H2_whole));

        // get the kinetic energies
        float ke_O = 0.5 * mass_O * v_O * v_O;
        float ke_H1 = 0.5 * mass_H * v_H1 * v_H1;
        float ke_H2 = 0.5 * mass_H * v_H2 * v_H2;

        float total_ke = ke_O + ke_H1 + ke_H2;

        // preserve the kinetic energy, and point it in the direction of the COMV
    }
}

// called at the end of stepFinal, this will be silent unless something is amiss (if the constraints are not satisifed for every molecule)
// i.e., bond lengths should be fixed, and the dot product of the relative velocities along a bond with the 
// bond vector should be identically zero.
template <class DATA>
__global__ void validateConstraints(int4* waterIds, int *idToIdxs, 
                                    float4 *xs, float4 *vs, 
                                    int nMolecules, BoundsGPU bounds, 
                                    DATA fixRigidData, bool* constraints, int turn) {

    int idx = GETIDX();

    if (idx < nMolecules) {

        // extract the ids
        int id_O =  waterIds[idx].x;
        int id_H1 = waterIds[idx].y;
        int id_H2 = waterIds[idx].z;

        // so these are the actual OH, OH, HH, OM bond lengths, not the sides of the canonical triangle
        // save them as double4, cast them as float for arithmetic, since really all calculations are done in float
        double4 sideLengths = fixRigidData.sideLengths;

        // (re)set the constraint boolean
        constraints[idx] = true;

        // get the positions
        float3 pos_O = make_float3(xs[idToIdxs[id_O]]);
        float3 pos_H1 = make_float3(xs[idToIdxs[id_H1]]);
        float3 pos_H2 = make_float3(xs[idToIdxs[id_H2]]);

        // get the velocities
        float3 vel_O = make_float3(vs[idToIdxs[id_O]]);
        float3 vel_H1= make_float3(vs[idToIdxs[id_H1]]);
        float3 vel_H2= make_float3(vs[idToIdxs[id_H2]]);

        // our constraints are that the 
        // --- OH1, OH2, H1H2 bond lengths are the specified values;
        // --- the dot product of the bond vector with the relative velocity along the bond is zero
        
        // take a look at the bond lengths first
        // i ~ O, j ~ H1, k ~ H2
        float3 r_ij = bounds.minImage(pos_H1 - pos_O);
        float3 r_ik = bounds.minImage(pos_H2 - pos_O);
        float3 r_jk = bounds.minImage(pos_H2 - pos_H1);

        float len_rij = length(r_ij);
        float len_rik = length(r_ik);
        float len_rjk = length(r_jk);
    
        // side length AB (AB = AC) (AB ~ intramolecular OH bond)

        float AB = (float) sideLengths.x;
        float BC = (float) sideLengths.z;

        // and these should be ~0.0f
        float bondLenij = len_rij - AB;
        float bondLenik = len_rik - AB;
        float bondLenjk = len_rjk - BC;

        // these values correspond to the fixed side lengths of the triangle congruent to the specific water model being examined
        // --- or rather, they /should/, if the SETTLE algorithm is implemented correctly

        // now take a look at dot product of relative velocities along the bond vector
        float3 v_ij = vel_H1 - vel_O;
        float3 v_ik = vel_H2 - vel_O;
        float3 v_jk = vel_H2 - vel_H1;

        // this is the "relative velocity along the bond" constraint
        float bond_ij = dot(r_ij, v_ij);
        float bond_ik = dot(r_ik, v_ik);
        float bond_jk = dot(r_jk, v_jk);

        // 1e-5
        float tolerance = 0.00001;
        // note that these values should all be zero
        //printf("molecule id %d bond_ij %f bond_ik %f bond_jk %f\n", idx, bond_ij, bond_ik, bond_jk);
        if ( (fabs(bond_ij) > tolerance) or 
             (fabs(bond_ik) > tolerance) or 
             (fabs(bond_jk) > tolerance)) {
            // then the velocity constraints are not satisfied
            constraints[idx] = false;
            printf("water molecule %d did not have velocity constraints satisfied at turn %d,\ndot(r_ij, v_ij) for ij = {01, 02, 12} was found to be %f, %f, and %f, respectively; tolerance is currently %f\n", idx, (int) turn,
                    bond_ij, bond_ik, bond_jk, tolerance);
        }

        if ( (fabs(bondLenij) > tolerance) or
             (fabs(bondLenik) > tolerance) or 
             (fabs(bondLenjk) > tolerance)) {
            // then the position (bond length) constraints are not satisfied
            constraints[idx] = false;
            printf("water molecule %d did not have position constraints satisfied at turn %d\nExpected bond lengths (OH1, OH2, H1H2) of %f %f %f, got %f %f %f; tolerance is currently %f\n", idx, (int) turn,
                   AB, AB, BC, len_rij, len_rik, len_rjk, tolerance);
        }

        // syncthreads, parallel reduction to get any false evaluations, and mdFatal call if !true


    }
}


// pretty straightforward - printing out the positions, velocities, and forces of 4 atoms                                   
__global__ void printGPD_Rigid(uint* ids, float4 *xs, float4 *vs, float4 *fs, int nAtoms) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        uint id = ids[idx];
        if (id < 4) {
            float4 pos = xs[idx];
            int type = xs[idx].w;
            float4 vel = vs[idx];
            float4 force = fs[idx];
            uint groupTag = force.w;
            printf("atom id %d type %d at coords %f %f %f\n", id, type, pos.x, pos.y, pos.z);
            printf("atom id %d mass %f with vel  %f %f %f\n", id, vel.w, vel.x, vel.y, vel.z);
            printf("atom id %d groupTag %d with force %f %f %f\n", id, groupTag,  force.x, force.y, force.z);
        }
    }
}

//fills r which rotates a to b
__device__ void fillRotMatrix(float3 a, float3 b, float3 r[]) {
  float3 v = cross(a, b);
  float s = length(v);
  float c = dot(b,a);
  float3 vx[3] = {make_float3(0,-v.z,v.y),make_float3(v.z,0,-v.x),make_float3(-v.y,v.x,0)};
  float3 vt[3] = {make_float3(0,v.z,-v.y),make_float3(-v.z,0,v.x),make_float3(v.y,-v.x,0)};
  float3 i[3] = {make_float3(1,0,0),make_float3(0,1,0),make_float3(0,0,1)};
  if (s != 0 and (a.x != b.x or a.y != b.y or a.z != b.z)) {
    for (int row = 0; row < 3; row++) {
      r[row] = rotateCoords(vt[row],vx);
      r[row] *= (1 - c)/(s*s);
      r[row] += vx[row] + i[row];
    }
  } else {
    if (c > -1.0001 and c < -0.9999) {
      for (int row = 0; row < 3; row++) {
	r[row] = -1*(vx[row] + i[row]);
      }
    }
  }
}

// filld r which rotates a to b around the z axis
__device__ void fillRotZMatrix(float3 a, float3 b, float3 r[]){
  float s = length(cross(a, b));
  float c = dot(b,a);
  float3 g[3] = {make_float3(c, s, 0.0f), make_float3(-s, c, 0.0f), make_float3(0.0f, 0.0f, 1.0f)};
  for (int row = 0; row < 3; row++) {
    r[row] = g[row];
  }
}


// distribute the m site forces, and do an unconstrained integration of the velocity component corresponding to this additional force
// -- this is required for 4-site models with a massless particle.
//    see compute_gamma() function for details.
__global__ void distributeMSite(int4 *waterIds, float4 *xs, float4 *vs, float4 *fs, 
                                Virial *virials,
                                int nMolecules, float gamma, float dtf, int* idToIdxs, BoundsGPU bounds)

{
    int idx = GETIDX();
    if (idx < nMolecules) {
        //printf("in distributeMSite idx %d\n", idx);
        // by construction, the id's of the molecules are ordered as follows in waterIds array
        int3 waterMolecule = make_int3(waterIds[idx]);

        int id_O  = waterIds[idx].x;
        int id_H1 = waterIds[idx].y;
        int id_H2 = waterIds[idx].z;
        int id_M  = waterIds[idx].w;

        float4 vel_O = vs[idToIdxs[id_O]];
        float4 vel_H1 = vs[idToIdxs[id_H1]];
        float4 vel_H2 = vs[idToIdxs[id_H2]];

        //printf("In distributeMSite, velocity of Oxygen %d is %f %f %f\n", id_O, vel_O.x, vel_O.y, vel_O.z);
        // need the forces from O, H1, H2, and M
        float4 fs_O  = fs[idToIdxs[id_O]];
        float4 fs_H1 = fs[idToIdxs[id_H1]];
        float4 fs_H2 = fs[idToIdxs[id_H2]];
        float4 fs_M  = fs[idToIdxs[id_M]];

        //printf("Force on m site: %f %f %f\n", fs_M.x, fs_M.y, fs_M.z);
        //printf("Force on Oxygen : %f %f %f\n", fs_O.x, fs_M.y, fs_M.z);
        // now, get the partial force contributions from the M-site; prior to adding these to the
        // array of forces for the given atom, integrate the velocity of the atom according to the distributed force contribution

        // this expression derived below in FixRigid::compute_gamma() function
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
        
        Virial virialToDistribute = virials[idToIdxs[id_M]];
        
        Virial distribute_O = virialToDistribute * (1.0 - (2.0 * gamma));
        Virial distribute_H = virialToDistribute * gamma;
        
        virials[idToIdxs[id_O]] += distribute_O;
        virials[idToIdxs[id_H1]] += distribute_H;
        virials[idToIdxs[id_H2]] += distribute_H;

        vs[idToIdxs[id_M]] = make_float4(0,0,0,INVMASSLESS);
        // finally, modify the forces; this way, the distributed force from M-site is incorporated in to nve_v() integration step
        // at beginning of next iteration in IntegratorVerlet.cu
        fs_O += fs_O_d;
        fs_H1 += fs_H_d;
        fs_H2 += fs_H_d;
       
        // set the global variables *fs[idToIdx[id]] to the new values
        fs[idToIdxs[id_O]] = fs_O;
        fs[idToIdxs[id_H1]]= fs_H1;
        fs[idToIdxs[id_H2]]= fs_H2;

        // zero the force on the M-site, just because
        fs[idToIdxs[id_M]] = make_float4(0.0, 0.0, 0.0,fs_M.w);
        // this concludes re-distribution of the forces;
        // we assume nothing needs to be done re: virials; this sum is already tabulated at inner force loop computation
        // in the evaluators; for safety, we might just set 

    }
}

__global__ void setMSite(int4 *waterIds, int *idToIdxs, float4 *xs, int nMolecules, BoundsGPU bounds) {

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
        float3 r_ij = bounds.minImage( (pos_H1 - pos_O));
        float3 r_ik = bounds.minImage( (pos_H2 - pos_O));

        // 0.1546 is the O-M bond length
        float3 r_M  = (pos_O) + 0.1546 * ( (r_ij + r_ik) / ( (length(r_ij + r_ik))));
    
        float4 pos_M_new = make_float4(r_M.x, r_M.y, r_M.z, pos_M_whole.w);
        xs[idToIdxs[id_M]] = pos_M_new;
    }
}


// computes the center of mass for a given water molecule
template <class DATA>
__global__ void compute_COM(int4 *waterIds, float4 *xs, float4 *vs, int *idToIdxs, int nMolecules, float4 *com, BoundsGPU bounds, DATA fixRigidData) {
  int idx = GETIDX();
  if (idx  < nMolecules) {
    float3 pos[3];

    double4 mass = fixRigidData.weights;
    int ids[3];
    ids[0] = waterIds[idx].x;
    ids[1] = waterIds[idx].y;
    ids[2] = waterIds[idx].z;
    for (int i = 0; i < 3; i++) {
      int myId = ids[i];
      int myIdx = idToIdxs[myId];
      float3 p = make_float3(xs[myIdx]);
      pos[i] = p;
      }
    for (int i=1; i<3; i++) {
      float3 delta = pos[i] - pos[0];
      delta = bounds.minImage(delta);
      pos[i] = pos[0] + delta;
    }
    float ims = com[idx].w;
    com[idx] = make_float4((pos[0] * mass.x) + ( (pos[1] + pos[2]) * mass.y));
    com[idx].w = ims;
  }
}

__global__ void compute_prev_val(int4 *waterIds, float4 *xs, float4 *xs_0, float4 *vs, float4 *vs_0, float4 *fs, float4 *fs_0, int nMolecules, int *idToIdxs) {
  int idx = GETIDX();
  if (idx < nMolecules) {
    int ids[3];

    // get the atom id's associated with this molecule idx

    // id of Oxygen
    ids[0] = waterIds[idx].x;

    // id of H1
    ids[1] = waterIds[idx].y;

    // id of H2
    ids[2] = waterIds[idx].z;

    // for O, H1, H2:
    for (int i = 0; i < 3; i++) {
      // get the idx of this atom id (O, H1, H2)
      int myIdx = idToIdxs[ids[i]];

      // store the xs initial data 
      xs_0[idx*3 + i] = xs[myIdx];
      vs_0[idx*3 + i] = vs[myIdx];
      fs_0[idx*3 + i] = fs[myIdx];
    }
  }
}


// ok, just initializes to zero
// ---- really, do as gromacs does and do a propagation backwards.
//      then call SETTLE with the 'new' coordinates as the 'unconstrained' initial coords - angles should be ~0
__global__ void set_init_vel_correction(int4 *waterIds, float4 *dvs_0, int nMolecules) {
  int idx = GETIDX();
  if (idx < nMolecules) {
    for (int i = 0; i < 3; i++) {
      dvs_0[idx*3 + i] = make_float4(0.0f,0.0f,0.0f,0.0f);
    }
  }
}


// implements the SETTLE algorithm for maintaining a rigid water molecule
template <class DATA>
__global__ void compute_SETTLE(int4 *waterIds, float4 *xs, float4 *xs_0, 
                               float4 *vs, float4 *vs_0, float4 *dvs_0, 
                               float4 *fs, float4 *fs_0, float4 *comOld, 
                               DATA fixRigidData, int nMolecules, 
                               float dt, float dtf, 
                               int *idToIdxs, BoundsGPU bounds) {
    int idx = GETIDX();
    if (idx < nMolecules) {

        float ra = (float) fixRigidData.canonicalTriangle.x;
        float rb = (float) fixRigidData.canonicalTriangle.y;
        float rc = (float) fixRigidData.canonicalTriangle.z;

        double4 weights = fixRigidData.weights;
        double weightO = weights.x;
        double weightH = weights.y;
       
        // so, our initial data from xs_0, accessed via idx
        float3 posO_initial = make_float3(xs_0[idx*3]);
        float3 posH1_initial= make_float3(xs_0[idx*3 + 1]);
        float3 posH2_initial= make_float3(xs_0[idx*3 + 2]);

        // get the molecule at this idx
        int4 atomsFromMolecule = waterIds[idx];

        // and the unconstrained, updated positions
        float3 posO = make_float3(xs[idToIdxs[atomsFromMolecule.x]]);
        float3 posH1= make_float3(xs[idToIdxs[atomsFromMolecule.y]]);
        float3 posH2= make_float3(xs[idToIdxs[atomsFromMolecule.z]]);




    }
}


void FixRigid::handleBoundsChange() {

    if (TIP4P) {
    
        GPUData &gpd = state->gpd;
        int activeIdx = gpd.activeIdx();
        BoundsGPU &bounds = state->boundsGPU;
        
        int nAtoms = state->atoms.size();
        if (printing) {
            printf("Calling printGPD_Rigid at turn %d\n in FixRigid::handleBoundsChange, before doing anything\n", (int) state->turn);
            cudaDeviceSynchronize();
            printGPD_Rigid<<<NBLOCK(nAtoms), PERBLOCK>>>(gpd.ids(activeIdx),
                                                         gpd.xs(activeIdx),
                                                         gpd.vs(activeIdx),
                                                         gpd.fs(activeIdx),nAtoms);
            cudaDeviceSynchronize();
        }
        // get a few pieces of data as required
        // -- all we're doing here is setting the position of the M-Site prior to computing the forces
        //    within the simulation.  Otherwise, the M-Site will likely be far away from where it should be, 
        //    relative to the moleule.  We do not solve the constraints on the rigid body at this time.
        // we need to reset the position of the M-Site prior to calculating the forces
        
        setMSite<<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.idToIdxs.d_data.data(), gpd.xs(activeIdx), nMolecules, bounds);
        if (printing) {
            cudaDeviceSynchronize();

            printf("Calling printGPD_Rigid at turn %d\n in FixRigid::handleBoundsChange, after calling setMSite\n", (int) state->turn);
            printGPD_Rigid<<<NBLOCK(nAtoms), PERBLOCK>>>(gpd.ids(activeIdx),gpd.xs(activeIdx),
                                                         gpd.vs(activeIdx),gpd.fs(activeIdx),nAtoms);
            cudaDeviceSynchronize();
        }
    }

    return;


}




void FixRigid::set_fixed_sides() {
   
    // we already have the bond lengths from 'setStyleBondLengths'...

    // we have not yet set the 'weights' attribute for our fixRigidData instance.
    // do so now.
    double massO = 15.999400000;
    double massH = 1.00800000;
    double massWater = massO + (2.0 * massH);

    fixRigidData.weights.x = massO / massWater;
    fixRigidData.weights.y = massH / massWater;
    fixRigidData.weights.z = massO;
    fixRigidData.weights.w = massH;

    //float4 fixRigidData.sideLengths is as OH, OH, HH [, OM]...
    double rc = 0.5 * fixRigidData.sideLengths.z;

    // OH bond length
    double OH = fixRigidData.sideLengths.x;
    // so, a little bit of trig
    double raPlusRb = sqrt( (OH * OH) - (rc * rc));

    // ra is just the distance along the y-axis from Oxygen to COM..
    // -- compute COM with oxygen at origin, then compute COM and get distance in y-dir (molecule laying in XY plane)
    // -- we know the apex angle HOH due to having the OH, OH, HH bond lengths. 
    //    basic trig to get theta, then compute COM place the two hydrogens at +/- (0.5 theta) along the y axis (3pi/2)
    //    compute COM using weights & double3 position arithmetic average

    double3 pos_O = make_double3(0,0,0);
    
    double halftheta = atan(rc/raPlusRb);

    double h1theta = (3.0 * M_PI / 2.0) + halftheta;
    double h2theta = (3.0 * M_PI / 2.0) - halftheta;
    // and multiply these by the OH bond lengths for this model
    double3 pos_H1 = make_double3( cos(h1theta), sin(h1theta), 0)  * OH;
    double3 pos_H2 = make_double3( cos(h2theta),  sin(h2theta), 0) * OH;

    double3 com = (pos_O * fixRigidData.weights.x) + ( (pos_H1 + pos_H2) * fixRigidData.weights.y);
    
    // take diff.y as ra
    double3 diff = (pos_O - com);

    double ra = diff.y;
    // finally, subtract ra from raPlusRb to get rb.
    double rb = raPlusRb - ra;

    fixRigidData.canonicalTriangle = make_double4(ra,rb,rc,0.0);

}




void FixRigid::setStyleBondLengths() {
    
    if (TIP4P) {
        // 4-site models here

        if (style == "TIP4P/2005") {
            
            sigma_O = 3.15890000;
            r_OH = 0.95720000000;
            r_HH = 1.51390065208;
            r_OM = 0.15460000000;
            
        } else {
            
            mdError("Only TIP3P and TIP4P/2005 are supported in setStyleBondLengths at the moment.\n");

        }

    } else if (TIP3P) {
        // 3-site models here
        // set r_OM to 0.0 for completeness
        r_OM = 0.00000000000;

        if (style == "TIP3P") {

            sigma_O = 3.15890000;
            r_OH = 0.95720000000;
            r_HH = 1.51390065208;
        } else {

            mdError("Only TIP3P and TIP4P/2005 are supported in setStyleBondLengths at the moment.\n");

        }
    } else {

        mdError("Neither 3-site nor 4-site model currently selected in FixRigid.\n");

    }

    if (state->units.unitType == UNITS::LJ) {

        // but how are charges scaled in LJ units? really unclear
        r_OH /= sigma_O;
        r_OM /= sigma_O;
        r_HH /= sigma_O;
        mdError("It is currently not possible to scale the dielectric constant, and so LJ-units simulations are presently incorrect for water.\n");
    }

    fixedSides = make_double4(r_OH, r_OH, r_HH, r_OM);

    // set the sideLengths variable in fixRigidData to the fixedSides values
    fixRigidData.sideLengths = fixedSides;

    return;

}


void FixRigid::setStyle(std::string style_) {
    if (style_ == "TIP4P/LONG") {
        style = style_;
    } else if (style_ == "TIP4P/2005") {
        style = style_;
    } else if (style_ == "TIP4P") {
        style = style_;
    } else if (style_ == "TIP3P") {
        style = style_;
    } else if (style == "TIP3P/LONG") {
        style = style_;
    } else {
        mdError("Supported style arguments are \"TIP4P\", \"TIP4P/2005\", \"TIP4P/LONG\", \"TIP3P\", \"TIP3P/LONG\"");
    };

}


void FixRigid::compute_gamma() {

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

    // our gamma value; 0.1546 is the bond length O-M in Angstroms (which we can 
    // assume is the units being used, because those are only units of distance used in DASH)
    gamma = 0.1546 / denominator;
    //printf("gamma value: %f\n", gamma);

}





void FixRigid::createRigid(int id_a, int id_b, int id_c, int id_d) {
    TIP4P = true;
    int4 waterMol = make_int4(0,0,0,0);
    Vector a = state->idToAtom(id_a).pos;
    Vector b = state->idToAtom(id_b).pos;
    Vector c = state->idToAtom(id_c).pos;
    Vector d = state->idToAtom(id_d).pos;

    double ma = state->idToAtom(id_a).mass;
    double mb = state->idToAtom(id_b).mass;
    double mc = state->idToAtom(id_c).mass;
    double md = state->idToAtom(id_d).mass;


    double ims = 1.0 / (ma + mb + mc + md);
    float4 ims4 = make_float4(0.0f, 0.0f, 0.0f, float(ims));
    invMassSums.push_back(ims4);

    bool ordered = true;
    if (! (ma > mb && ma > mc)) ordered = false;
    
    if (! (mb == mc) ) ordered = false;
    if (! (mb > md) )  ordered = false;
    
    if (! (ordered)) {
        printf("Found masses O, H, H, M in order: %f %f %f %f\n", ma, mb, mc, md);
    }
    if (! (ordered)) mdError("Ids in FixRigid::createRigid must be as O, H1, H2, M");

    /*
    printf("adding ids %d %d %d %d\n", id_a, id_b, id_c, id_d);
    printf("with masses %f %f %f %f\n", ma, mb, mc, md);
    printf("position id_a: %f %f %f\n", a[0], a[1], a[2]);
    printf("position id_b: %f %f %f\n", b[0], b[1], b[2]);
    printf("position id_c: %f %f %f\n", c[0], c[1], c[2]);
    printf("position id_c: %f %f %f\n ", d[0], d[1], d[2]);
    */
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


    // and we need to do something else here as well. what is it tho

}

void FixRigid::createRigid(int id_a, int id_b, int id_c) {
    TIP3P = true;
    int4 waterMol = make_int4(0,0,0,0);
    Vector a = state->idToAtom(id_a).pos;
    Vector b = state->idToAtom(id_b).pos;
    Vector c = state->idToAtom(id_c).pos;

    double ma = state->idToAtom(id_a).mass;
    double mb = state->idToAtom(id_b).mass;
    double mc = state->idToAtom(id_c).mass;
    double ims = 1.0 / (ma + mb + mc);
    float4 ims4 = make_float4(0.0f, 0.0f, 0.0f, float(ims));
    invMassSums.push_back(ims4);

    // this discovers the order of the id's that was passed in, i.e. OHH, HOH, HHO, etc.
    float det = a[0]*b[1]*c[2] - a[0]*c[1]*b[2] - b[0]*a[1]*c[2] + b[0]*c[1]*a[2] + c[0]*a[1]*b[2] - c[0]*b[1]*a[2];
    if (state->idToAtom(id_a).mass == state->idToAtom(id_b).mass) {
    waterMol = make_int4(id_c,id_a,id_b,0);
    if (det < 0) {
      waterMol = make_int4(id_c,id_b,id_a,0);
    }
    }
    else if (state->idToAtom(id_b).mass == state->idToAtom(id_c).mass) {
    waterMol = make_int4(id_a,id_b,id_c,0);
    if (det < 0) {
      waterMol = make_int4(id_a,id_c,id_b,0);
    }
    }
    else if (state->idToAtom(id_c).mass == state->idToAtom(id_a).mass) {
    waterMol = make_int4(id_b,id_c,id_a,0);
    if (det < 0) {
      waterMol = make_int4(id_b,id_a,id_c,0);
    }
    } else {
    assert("waterMol set" == "true");
    }
    waterIds.push_back(waterMol);
    Bond bondOH1;
    Bond bondOH2;
    Bond bondHH;
    bondOH1.ids = { {waterMol.x,waterMol.y} };
    bondOH2.ids = { {waterMol.x,waterMol.z} };
    bondHH.ids = { {waterMol.y,waterMol.z} };
    bonds.push_back(bondOH1);
    bonds.push_back(bondOH2);
    bonds.push_back(bondHH);
}


bool FixRigid::prepareForRun() {
    if (firstPrepare) {
        firstPrepare = false;
        return false;
    }
    nMolecules = waterIds.size();
    // cannot have more than one water model present
    if (TIP3P && TIP4P) {
        mdError("An attempt was made to use both 3-site and 4-site models in a simulation");
    }

    if (!(TIP3P || TIP4P)) {
        mdError("An attempt was made to use neither 3-site nor 4-site water models with FixRigid in simulation.");
    }

    // set the specific style of the fix
    if (style == "DEFAULT") {
        if (TIP3P) {
            style = "TIP3P";
        }
        if (TIP4P) {
            style = "TIP4P/2005";
        }
    }

    // an instance of FixRigidData
    fixRigidData = FixRigidData();
    setStyleBondLengths();

    printf("number of molecules in waterIds: %d\n", nMolecules);
    waterIdsGPU = GPUArrayDeviceGlobal<int4>(nMolecules);
    waterIdsGPU.set(waterIds.data());



    xs_0 = GPUArrayDeviceGlobal<float4>(3*nMolecules);
    vs_0 = GPUArrayDeviceGlobal<float4>(3*nMolecules);
    dvs_0 = GPUArrayDeviceGlobal<float4>(3*nMolecules);
    fs_0 = GPUArrayDeviceGlobal<float4>(3*nMolecules);
    com = GPUArrayDeviceGlobal<float4>(nMolecules);
    constraints = GPUArrayDeviceGlobal<bool>(nMolecules);
    com.set(invMassSums.data());

    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();

    // compute the force partition constant
    if (TIP4P) {
        compute_gamma();
    }

    BoundsGPU &bounds = state->boundsGPU;
    compute_COM<FixRigidData><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), gpd.vs(activeIdx), 
                                         gpd.idToIdxs.d_data.data(), nMolecules, com.data(), bounds, fixRigidData);

    set_fixed_sides();
    
    set_init_vel_correction<<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), dvs_0.data(), nMolecules);

    cudaDeviceSynchronize();
    // adjust the initial velocities to conform to our velocity constraints
    // -- here, we preserve the COMV of the system, while imposing strictly translational motion on the molecules
    //    - we use shared memory to compute center of mass velocity of the group, allowing for one kernel call
    /*
    adjustInitialVelocities<<<NBLOCK(nMolecules), PERBLOCK, nMolecules*sizeof(float3)>>>(
                            waterIdsGPU.data(), gpd.idToIdxs.d_data.data(),
                            gpd.xs(activeIdx), gpd.vs(activeIdx), nMolecules,
                            bounds);
    */
    // make sure all of the initial velocities have been adjusted prior to validating the constraints
    cudaDeviceSynchronize();
    
    // validate that we have good initial conditions
    validateConstraints<FixRigidData> <<<NBLOCK(nMolecules), PERBLOCK>>> (waterIdsGPU.data(), gpd.idToIdxs.d_data.data(), 
                                                           gpd.xs(activeIdx), gpd.vs(activeIdx), 
                                                           nMolecules, bounds, fixRigidData, 
                                                           constraints.data(), state->turn);
  
    return true;
}

bool FixRigid::stepInit() {
    
    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();
    BoundsGPU &bounds = state->boundsGPU;
    //float dtf = 0.5f * state->dt * state->units.ftm_to_v;
    int nAtoms = state->atoms.size();
    float dt = state->dt;
    if (TIP4P) {
        if (printing) {
            cudaDeviceSynchronize();
            printf("Calling printGPD_Rigid at turn %d\n in FixRigid::stepInit, before doing anything\n", (int) state->turn);
            printGPD_Rigid<<<NBLOCK(nAtoms), PERBLOCK>>>(gpd.ids(activeIdx),gpd.xs(activeIdx),gpd.vs(activeIdx),gpd.fs(activeIdx),nAtoms);
            cudaDeviceSynchronize();
        }
    }

    // compute the current center of mass for the solved constraints at the beginning of this turn
    compute_COM<FixRigidData><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                           gpd.vs(activeIdx), gpd.idToIdxs.d_data.data(), 
                                           nMolecules, com.data(), bounds, fixRigidData);

    // save the positions, velocities, forces from the previous, fully updated turn in to our local arrays
    compute_prev_val<<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), xs_0.data(), 
                                                gpd.vs(activeIdx), vs_0.data(), gpd.fs(activeIdx), 
                                                fs_0.data(), nMolecules, gpd.idToIdxs.d_data.data());


    if (TIP4P) {
        if (printing) {
            cudaDeviceSynchronize();
            printf("Calling printGPD_Rigid at turn %d\n in FixRigid::stepInit, after doing compute_COM and compute_prev_val\n", (int) state->turn);
            printGPD_Rigid<<<NBLOCK(nAtoms), PERBLOCK>>>(gpd.ids(activeIdx),gpd.xs(activeIdx),
                                                         gpd.vs(activeIdx),gpd.fs(activeIdx),nAtoms);
            cudaDeviceSynchronize();
        }
    
    }
    //xs_0.get(cpu_com);
    //std::cout << cpu_com[0] << "\n";
    return true;
}

// nope don't need this
bool FixRigid::postNVE_V() {

    /*
       GPUData &gpd = state->gpd;
       int activeIdx = gpd.activeIdx();
       BoundsGPU &bounds = state->boundsGPU;
    //float dtf = 0.5f * state->dt * state->units.ftm_to_v;
    int nAtoms = state->atoms.size();
    float dt = state->dt;
     */
    /*
       updateVs<<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), xs_0.data(),
       gpd.vs(activeIdx), vs_0.data(), dvs_0.data(),  gpd.fs(activeIdx),
       fs_0.data(), com.data(), nMolecules, dt, gpd.idToIdxs.d_data.data(), bounds);

    // and reset the prev vals arrays, since we modified the velocities
    compute_prev_val<<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), xs_0.data(), 
                                                gpd.vs(activeIdx), vs_0.data(), gpd.fs(activeIdx), 
                                                fs_0.data(), nMolecules, gpd.idToIdxs.d_data.data());
    */
    return true;
}

bool FixRigid::stepFinal() {
    float dt = state->dt;
    float dtf = 0.5 * state->dt * state->units.ftm_to_v;
    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();
    BoundsGPU &bounds = state->boundsGPU;
    int nAtoms = state->atoms.size();
    // first, unconstrained velocity update continues: distribute the force from the M-site
    //        and integrate the velocities accordingly.  Update the forces as well.
    // Next,  do compute_SETTLE as usual on the (as-yet) unconstrained positions & velocities

    printf("FixRigid::stepFinal at turn %d\n", (int) state->turn);
    // from IntegratorVerlet
    if (TIP4P) {

        /*
        if (printing) {
            cudaDeviceSynchronize();
            printf("Calling printGPD_Rigid at turn %d\n in FixRigid::stepFinal, before doing anything\n", (int) state->turn);
            printGPD_Rigid<<<NBLOCK(nAtoms), PERBLOCK>>>(gpd.ids(activeIdx),gpd.xs(activeIdx),gpd.vs(activeIdx),gpd.fs(activeIdx),nAtoms);
            cudaDeviceSynchronize();
        }
        */
        float dtf = 0.5f * state->dt * state->units.ftm_to_v;
        distributeMSite<<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                                     gpd.vs(activeIdx),  gpd.fs(activeIdx),
                                                     gpd.virials.d_data.data(),
                                                     nMolecules, gamma, dtf, gpd.idToIdxs.d_data.data(), bounds);

        /*
        if (printing) { 
            cudaDeviceSynchronize();
            printf("Calling printGPD_Rigid at turn %d\n in FixRigid::stepFinal, after calling distributeMSite\n", (int) state->turn);
            printGPD_Rigid<<<NBLOCK(nAtoms), PERBLOCK>>>(gpd.ids(activeIdx),gpd.xs(activeIdx),gpd.vs(activeIdx),gpd.fs(activeIdx),nAtoms);
            cudaDeviceSynchronize();
        }
        */

    }

    cudaDeviceSynchronize();

    SAFECALL((compute_SETTLE<FixRigidData><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                                xs_0.data(), gpd.vs(activeIdx), vs_0.data(), 
                                                dvs_0.data(), gpd.fs(activeIdx), fs_0.data(), 
                                                com.data(), fixRigidData, nMolecules, dt, dtf,  
                                                gpd.idToIdxs.d_data.data(), bounds)));

 
    cudaDeviceSynchronize();
    //printf("did compute_SETTLE!\n");
    // finally, reset the position of the M-site to be consistent with that of the new, constrained water molecule
    if (TIP4P) {
        /*
        if (printing) { 
            printf("Calling printGPD_Rigid at turn %d\n in FixRigid::stepFinal, before calling setMSite \n", (int) state->turn);
            cudaDeviceSynchronize();
            printGPD_Rigid<<<NBLOCK(nAtoms), PERBLOCK>>>(gpd.ids(activeIdx),gpd.xs(activeIdx),gpd.vs(activeIdx),gpd.fs(activeIdx),nAtoms);
        
            cudaDeviceSynchronize();
        }
        */
        setMSite<<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.idToIdxs.d_data.data(), gpd.xs(activeIdx), nMolecules, bounds);
        
        /*
        if (printing) { 
            cudaDeviceSynchronize();
            printf("Calling printGPD_Rigid at turn %d\n in FixRigid::stepFinal, after calling setMSite \n", (int) state->turn);
            printGPD_Rigid<<<NBLOCK(nAtoms), PERBLOCK>>>(gpd.ids(activeIdx),gpd.xs(activeIdx),gpd.vs(activeIdx),gpd.fs(activeIdx),nAtoms);
            cudaDeviceSynchronize();
        }
        */
    }
 
    SAFECALL((validateConstraints<FixRigidData><<<NBLOCK(nMolecules), PERBLOCK>>> (waterIdsGPU.data(), 
                                                                                   gpd.idToIdxs.d_data.data(), 
                                                                                   gpd.xs(activeIdx), 
                                                                                   gpd.vs(activeIdx), 
                                                                                   nMolecules, bounds, 
                                                                                   fixRigidData, 
                                                                                   constraints.data(), 
                                                                                   (int) state->turn)));
    return true;
}

// export the overloaded function
void (FixRigid::*createRigid_x1) (int, int, int)      = &FixRigid::createRigid;
void (FixRigid::*createRigid_x2) (int, int, int, int) = &FixRigid::createRigid;

void export_FixRigid() 
{
  py::class_<FixRigid, boost::shared_ptr<FixRigid>, py::bases<Fix> > 
      ( 
		"FixRigid",
		py::init<boost::shared_ptr<State>, std::string, std::string>
	    (py::args("state", "handle", "groupHandle")
         )
        )
    .def("createRigid", createRigid_x1,
	    (py::arg("id_a"), 
         py::arg("id_b"), 
         py::arg("id_c")
         )
	 )
    .def("createRigid", createRigid_x2,
        (py::arg("id_a"), 
         py::arg("id_b"),  
         py::arg("id_c"), 
         py::arg("id_d"))
     )
    .def("setStyle", &FixRigid::setStyle,
         py::arg("style")
        )
	.def_readwrite("printing", &FixRigid::printing)
    ;
}



