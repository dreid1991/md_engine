#include "FixRigid.h"

#include "State.h"
#include "VariantPyListInterface.h"
#include "boost_for_export.h"
#include "cutils_math.h"
#include "cutils_func.h"
#include <math.h>
#include "globalDefs.h"
#include "xml_func.h"
namespace py = boost::python;
const std::string rigidType = "Rigid";

FixRigid::FixRigid(boost::shared_ptr<State> state_, std::string handle_, std::string groupHandle_) : Fix(state_, handle_, groupHandle_, rigidType, true, true, false, 1) {

    // set both to false initially; using one of the createRigid functions will flip the pertinent flag to true
    TIP4P = false;
    TIP3P = false;
    printing = false;
    //requiresPostNVE_V = true;
    style = "DEFAULT";
    // this fix requires the forces to have already been computed before we can 
    // call prepareForRun()
    requiresForces = true;
    readFromRestart();
}

__device__ inline float3 positionsToCOM(float3 *pos, float *mass, float ims) {
  return (pos[0]*mass[0] + pos[1]*mass[1] + pos[2]*mass[2])*ims;
}

inline __host__ __device__ float3 rotation(float3 vector, float3 X, float3 Y, float3 Z) {
    return make_float3(dot(X,vector), dot(Y,vector), dot(Z, vector));
}

inline __host__ __device__ double3 rotation(double3 vector, double3 X, double3 Y, double3 Z) {
    return make_double3(dot(X,vector), dot(Y,vector), dot(Z, vector));
}


// so this is the exact same function as in FixLinearMomentum. 
__global__ void rigid_remove_COMV(int nAtoms, float4 *vs, float4 *sumData, float3 dims) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        float4 v = vs[idx];
        float4 sum = sumData[0];
        float invMassTotal = 1.0f / sum.w;
        v.x -= sum.x * invMassTotal * dims.x;
        v.y -= sum.y * invMassTotal * dims.y;
        v.z -= sum.z * invMassTotal * dims.z;
        vs[idx] = v;
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

        float mag_v_ij = length(v_ij);
        float mag_v_ik = length(v_ik);
        float mag_v_jk = length(v_jk);

        // this is the "relative velocity along the bond" constraint
        float bond_ij = dot(r_ij, v_ij);
        float bond_ik = dot(r_ik, v_ik);
        float bond_jk = dot(r_jk, v_jk);
        /*
        float constr_relVel_ij = bond_ij / mag_v_ij;
        float constr_relVel_ik = bond_ik / mag_v_ik;
        float constr_relVel_jk = bond_jk / mag_v_jk;
        */
        // 1e-5
        float tolerance = 0.00001;
        // note that these values should all be zero
        //printf("molecule id %d bond_ij %f bond_ik %f bond_jk %f\n", idx, bond_ij, bond_ik, bond_jk);
        if ( ( bond_ij > tolerance) or 
             ( bond_ik > tolerance) or
             ( bond_jk > tolerance) ) {
            constraints[idx] = false;
            printf("water molecule %d unsatisfied velocity constraints at turn %d,\ndot(r_ij, v_ij) for ij = {01, 02, 12} %f, %f, and %f; tolerance %f\nmagnitudes of relative velocities {ij,ik,jk} %f %f %f\n", idx, (int) turn,
                    bond_ij, bond_ik, bond_jk, tolerance,
                    mag_v_ij, mag_v_ik, mag_v_jk);
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
        //if (id < 4) {
        float4 pos = xs[idx];
        int type = xs[idx].w;
        float4 vel = vs[idx];
        float4 force = fs[idx];
        //uint groupTag = force.w;
        printf("atom id %d type %d at coords %f %f %f\n", id, type, pos.x, pos.y, pos.z);
        printf("atom id %d mass %f with vel  %f %f %f\n", id, vel.w, vel.x, vel.y, vel.z);
        printf("atom id %d mass %f with force %f %f %f\n", id, vel.w, force.x, force.y, force.z);
        //}
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

template <class DATA>
__global__ void setMSite(int4 *waterIds, int *idToIdxs, float4 *xs, int nMolecules, BoundsGPU bounds, DATA fixRigidData) {

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

        // fixRigidData.sideLengths.w is the OM vector
        float3 r_M  = (pos_O) + fixRigidData.sideLengths.w * ( (r_ij + r_ik) / ( (length(r_ij + r_ik))));
    
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

template <class DATA, bool HALFSTEP, bool DEBUG_BOOL>
__global__ void settleVelocities(int4 *waterIds, float4 *xs, float4 *xs_0, 
                               float4 *vs, float4 *vs_0,
                               float4 *fs, float4 *fs_0, float4 *comOld, 
                               DATA fixRigidData, int nMolecules, 
                               float dt, float dtf,
                               int *idToIdxs, BoundsGPU bounds, int turn) {
    int idx = GETIDX();
    if (idx < nMolecules) {
        
        // get the molecule at this idx
        int4 atomsFromMolecule = waterIds[idx];

        // get the atom idxs
        int idxO = idToIdxs[atomsFromMolecule.x];
        int idxH1= idToIdxs[atomsFromMolecule.y];
        int idxH2= idToIdxs[atomsFromMolecule.z];
        
        float4 velO_whole = vs[idxO];
        float4 velH1_whole= vs[idxH1];
        float4 velH2_whole= vs[idxH2];

        // grab the global velocities; these are our v0_a, v0_b, v0_c variables
        double3 velO = make_double3(velO_whole);
        double3 velH1= make_double3(velH1_whole);
        double3 velH2= make_double3(velH2_whole);

        if (HALFSTEP) {
            

        } else {
        // extract the whole positions
            float4 posO_whole = xs[idxO];
            float4 posH1_whole= xs[idxH1];
            float4 posH2_whole= xs[idxH2];

            // and the unconstrained, updated positions
            double3 posO = make_double3(posO_whole);
            double3 posH1= make_double3(posH1_whole);
            double3 posH2= make_double3(posH2_whole);

            double3 a_pos = posO;
            double3 b_pos = posO + bounds.minImage( (posH1 - posO) );
            double3 c_pos = posO + bounds.minImage( (posH2 - posO) );

            // note that a_pos, b_pos, c_pos are already minimum image vectors...
            double3 e_AB = b_pos - a_pos;
            double3 e_BC = c_pos - b_pos;
            double3 e_CA = a_pos - c_pos;

            // make them unit vectors by dividing by their length
            double inv_len_AB = 1.0 / (length(e_AB));
            double inv_len_BC = 1.0 / (length(e_BC));
            double inv_len_CA = 1.0 / (length(e_CA));

            e_AB *= inv_len_AB;
            e_BC *= inv_len_BC;
            e_CA *= inv_len_CA;

            /*
            float cosA = dot(-e_AB,e_CA);
            float cosB = dot(-e_BC,e_AB);
            float cosC = dot(-e_CA,e_BC);
            */

            /*
            double cosA = fixRigidData.cosA;
            double cosB = fixRigidData.cosB;
            double cosC = fixRigidData.cosC;
            */

            /*
            if (( (turn % 100) == 0) and idx == 3) {
                printf("cosA, cosB, cosC: %f %f %f\n", cosA, cosB, cosC);
                printf("fixRigidData members: %f %f %f\n", 
                       fixRigidData.cosA,
                       fixRigidData.cosB,
                       fixRigidData.cosC);
            }
            */

            float4 velO_whole = vs[idxO];
            float4 velH1_whole= vs[idxH1];
            float4 velH2_whole= vs[idxH2];

            // grab the global velocities; these are our v0_a, v0_b, v0_c variables
            double3 velO = make_double3(velO_whole);
            double3 velH1= make_double3(velH1_whole);
            double3 velH2= make_double3(velH2_whole);

            // and get the pertinent dot products
            double3 vel0_AB = velH1 - velO;
            double3 vel0_BC = velH2 - velH1;
            double3 vel0_CA = velO - velH2;

            double v0_AB = dot(e_AB, vel0_AB);
            double v0_BC = dot(e_BC, vel0_BC);
            double v0_CA = dot(e_CA, vel0_CA);

            //double4 weights = fixRigidData.weights;
            //double ma =  weights.z;
            //double mb =  weights.w;
            //double mamb = ma + mb;
            //double mambSqr = mamb * mamb;

            // exactly as in Miyamoto
            // --- except, since all three are /divided/ by d, and then later multiplied by dt
            //     for numerical precision, we just don't involve the timestep dt.
            /*
            double d = ( (2.0 * mambSqr) + ( 2.0 * ma * mb * cosA * cosB * cosC) - (2.0 * mb * mb * cosA * cosA) - 
                        ( ( ma * mamb) * ((cosB * cosB) + (cosC * cosC)) ) ) / (2.0 * mb);

            double tau_AB = ma * ( (v0_AB * (2.0*mamb - (ma * cosC * cosC))) +
                                  (v0_BC * ((mb * cosC * cosA) - (mamb * cosB)) ) + 
                                  (v0_CA * (ma * cosB * cosC - (2.0 * mb * cosA) ) ) ) / d;

            double tau_BC = ( (v0_BC * ( ( mambSqr - (mb * mb * cosA * cosA) ) )) + 
                             (v0_CA * ma * ((mb * cosA * cosB) - (mamb * cosC ) ) ) + 
                             (v0_AB * ma * ((mb * cosC * cosA) - (mamb * cosB) ) ) ) / d;

            double tau_CA = ma * ( (v0_CA * ((2.0 * mamb) - (ma * cosB * cosB) )) + 
                                  (v0_AB * ((ma * cosB * cosC) - (2.0 * mb * cosA) )) + 
                                  (v0_BC * ((mb * cosA * cosB) - (mamb * cosC) )) ) / d;

            float3 g_AB = e_AB * tau_AB;
            float3 g_BC = e_BC * tau_BC;
            float3 g_CA = e_CA * tau_CA;
            */

            // all data required to compute tau_AB etc. are in fixRigidData... do the algebra here
           
            double tau_AB = (v0_AB * fixRigidData.tauAB1) + 
                            (v0_BC * fixRigidData.tauAB2) + 
                            (v0_CA * fixRigidData.tauAB3);

            double tau_BC = (v0_BC * fixRigidData.tauBC1) + 
                            (v0_CA * fixRigidData.tauBC2) + 
                            (v0_AB * fixRigidData.tauBC3);

            double tau_CA = (v0_CA * fixRigidData.tauCA1) + 
                            (v0_AB * fixRigidData.tauCA2) + 
                            (v0_BC * fixRigidData.tauCA3);

            double3 g_AB = e_AB * tau_AB;
            double3 g_BC = e_BC * tau_BC;
            double3 g_CA = e_CA * tau_CA;
            
            // we have now computed our lagrange multipliers, and can add these to our velO, 
            // velH1, velH2 vectors.
            double3 constraints_dvO = 0.5 * (velO_whole.w) * ( (g_AB) - (g_CA) );
            double3 constraints_dvH1= 0.5 * (velH1_whole.w)* ( (g_BC) - (g_AB) );
            double3 constraints_dvH2= 0.5 * (velH2_whole.w)* ( (g_CA) - (g_BC) );
        

            velO_whole += constraints_dvO;
            velH1_whole+= constraints_dvH1;
            velH2_whole+= constraints_dvH2;
            // add the float3 differential velocity vectors

            // and assign the new vectors to global memory
            vs[idxO] = velO_whole;
            vs[idxH1]= velH1_whole;
            vs[idxH2]= velH2_whole;
            

            return;
        }
    }
}

// implements the SETTLE algorithm for maintaining a rigid water molecule
template <class DATA, bool DEBUG_BOOL>
__global__ void settlePositions(int4 *waterIds, float4 *xs, float4 *xs_0, 
                               float4 *vs, float4 *vs_0, 
                               float4 *fs, float4 *fs_0, float4 *comOld, 
                               DATA fixRigidData, int nMolecules, 
                               float dt, float dtf,
                               int *idToIdxs, BoundsGPU bounds, int turn) {
    int idx = GETIDX();
    if (idx < nMolecules) {
        
        double ra = fixRigidData.canonicalTriangle.x;
        double rb = fixRigidData.canonicalTriangle.y;
        double rc = fixRigidData.canonicalTriangle.z;

        double4 weights = fixRigidData.weights;
        double weightO = weights.x;
        double weightH = weights.y;
       
        // so, our initial data from xs_0, accessed via idx
        double3 posO_initial = make_double3(xs_0[idx*3]);
        double3 posH1_initial= make_double3(xs_0[idx*3 + 1]);
        double3 posH2_initial= make_double3(xs_0[idx*3 + 2]);

        // get the molecule at this idx
        int4 atomsFromMolecule = waterIds[idx];

        // get the atom idxs
        int idxO = idToIdxs[atomsFromMolecule.x];
        int idxH1= idToIdxs[atomsFromMolecule.y];
        int idxH2= idToIdxs[atomsFromMolecule.z];
        
        // extract the whole positions
        float4 posO_whole = xs[idxO];
        float4 posH1_whole= xs[idxH1];
        float4 posH2_whole= xs[idxH2];

        // and the unconstrained, updated positions
        double3 posO = make_double3(posO_whole);
        double3 posH1= make_double3(posH1_whole);
        double3 posH2= make_double3(posH2_whole);
    
        // get the relative vectors OH1, OH2 for the initial triangle (solution from last step)
        double3 vectorOH1 = bounds.minImage(posH1_initial - posO_initial);
        double3 vectorOH2 = bounds.minImage(posH2_initial - posO_initial);
 
        // get the relative vectors for the unconstrained triangle
        double3 OH1_unconstrained = bounds.minImage(posH1 - posO);
        double3 OH2_unconstrained = bounds.minImage(posH2 - posO);

        // move the hydrogens to their minimum image distance w.r.t. the oxygen
        posH1 = posO + OH1_unconstrained;
        posH2 = posO + OH2_unconstrained;

        // the center of mass 'd1' in the paper
        double3 COM_d1 = (posO * weightO) + ( ( posH1 + posH2 ) * weightH);

        // move the unconstrained atom positions s.t. COM ~ origin
        posO  -= COM_d1;
        posH1 -= COM_d1;
        posH2 -= COM_d1;

        // take the cross product of the two vectors to 
        // get the vector normal to the plane formed by 
        // $\Delta a_0 b_0 c_0$
        // -- note that this is /not/ normalized, and will need to be
        // ------ vector NORMAL to HOH plane ~ Z axis! so, axis3.
        double3 axis3 = cross(vectorOH1, vectorOH2);

        // take the cross product of axis 3 with the vector
        // pointing from the origin to the unconstrained position of the oxygen: pi_0 x A1'
        double3 axis1 = cross(posO, axis3);
        
        // and exhaust our remaining degree of freedom
        // take the cross product of axis 3 with axis 1
        // TODO: confirm, that we take cross(axis3, axis1) and not cross(axis1,axis3)
        double3 axis2 = cross(axis3, axis1);

        // normalize the three to get our X'Y'Z' coordinate system
        axis1 /= length(axis1);
        axis2 /= length(axis2);
        axis3 /= length(axis3);
        
        /* At this point it is useful to make the transpose axes */
        // -- for an orthogonal matrix, R^T ~ R^{-1}.
        //    so we can use this to transform back to our original coordinate system later
        double3 axis1T = make_double3(axis1.x, axis2.x, axis3.x);
        double3 axis2T = make_double3(axis1.y, axis2.y, axis3.y);
        double3 axis3T = make_double3(axis1.z, axis2.z, axis3.z);

        // rotate the hydrogen relative vectors about the axes
        double3 rotated_b0 = rotation(vectorOH1, axis1, axis2, axis3);
        double3 rotated_c0 = rotation(vectorOH2, axis1, axis2, axis3);

        double3 rotated_a1 = rotation(posO,  axis1, axis2, axis3);
        double3 rotated_b1 = rotation(posH1, axis1, axis2, axis3);
        double3 rotated_c1 = rotation(posH2, axis1, axis2, axis3);

        // should check this; if rotated_a1.z > ra, we will have a lot of problems problem
        // equation A8 in Miyamoto
        double sinPhi = rotated_a1.z / ra;

        //printf("molecule %d atoms O H H %d %d %d sinPhi value %f\n", idx, atomsFromMolecule.x, atomsFromMolecule.y, atomsFromMolecule.z, sinPhi);

        // this should be greater than zero
        double cosPhiSqr = 1.0 - (sinPhi * sinPhi);

        // cosPhiSqr must be strictly /greater/ than zero, 
        // since cosPhi shows up in a denominator (eq. A9)
        double cosPhi = sqrt(cosPhiSqr);

        // we now have cosPhi, the first relation in equation A10
        // --- calculate sinPsi; trig to get cosPsiSqr, then sqrtf()

        double sinPsi = (rotated_b1.z - rotated_c1.z) / (2.0 * rc * cosPhi);
        double cosPsiSqr = 1.0 - (sinPsi * sinPsi);
        //TODO: sqrtf --> sqrt?
        double cosPsi = sqrt(cosPsiSqr);

        // these assertions should likely be removed in the production release
        if ( (cosPhiSqr <= 0.0) or (cosPsiSqr <= 0.0)) {
            printf("failed at turn %d", turn);
            printf("Molecule %d computed cosPhiSqr %f cosPsiSqr value %f\nAborting.\n", idx, cosPhiSqr, cosPsiSqr);
            printf("Molecule %d rotated_a1.z, ra values of %f %f\n", idx, rotated_a1.z, ra);
            printf("Molecule %d posO values of : %f %f %f\n", idx, posO.x, posO.y, posO.z);
            printf("Molecule %d COM_d1 values of : %f %f %f\n", idx, COM_d1.x, COM_d1.y, COM_d1.z);
            printf("Molecule %d rotated_b1.z, rotated_c1.z, cosPhi values of %f %f %f\n", 
                   idx, rotated_b1.z, rotated_c1.z, cosPhi);
            printf("Molecule %d new O Pos: %f %f %f\n", idx, posO_whole.x, posO_whole.y, posO_whole.z);
            printf("Molecule %d new H1Pos: %f %f %f\n", idx, posH1_whole.x,posH1_whole.y,posH1_whole.z);
            printf("Molecule %d new H2Pos: %f %f %f\n", idx, posH2_whole.x,posH2_whole.y,posH2_whole.z);
            printf("Molecule %d expr A9 numerator, denominator, rc: %f %f %f\n",
                   idx, rotated_b1.z - rotated_c1.z, 2.0 * rc * cosPhi, rc);
            printf("Molecule %d O Pos turn init: %f %f %f\n", idx, posO_initial.x,
                   posO_initial.y, posO_initial.z);
            printf("Molecule %d H1Pos turn init: %f %f %f\n", idx, posH1_initial.x,
                   posH1_initial.y, posH1_initial.z);
            printf("Molecule %d H2Pos turn init: %f %f %f\n", idx, posH2_initial.x,
                   posH2_initial.y, posH2_initial.z);

            printf("Aborting\n");
            __syncthreads();
            assert(cosPhiSqr > 0.0);
            assert(cosPsiSqr > 0.0);
        }
        
        // -- all that remains is calculating $\theta$
        // first, do the displacements of the canonical triangle.
        double3 aPrime0 = make_double3(0.0, ra, 0.0);
        double3 bPrime0 = make_double3(-rc, -rb, 0.0);
        double3 cPrime0 = make_double3(rc, -rb, 0.0);

        // aPrime1 == aPrime0
        double3 bPrime1 = make_double3(-rc*cosPsi, -rb, rc*sinPsi);
        double3 cPrime1 = make_double3(rc*cosPsi, -rb, -rc*sinPsi);

        // skip to *Prime2.. (expressions A3 in Miyamoto)
        // --- note: I think there is an error in the paper here? why would it be rc? 
        //           we'll go with ra.
        double3 aPrime2 = make_double3(0.0,
                                     ra*cosPhi,
                                     ra*sinPhi);
        double3 bPrime2 = make_double3(-rc*cosPsi,
                                     -rb*cosPhi - rc*sinPsi*sinPhi,
                                     -rb*sinPhi + rc*sinPsi*cosPhi);
        double3 cPrime2 = make_double3(rc*cosPsi,
                                     -rb*cosPhi + rc*sinPsi*sinPhi,
                                     -rb*sinPhi - rc*sinPsi*cosPhi);

        // computing theta.. first compute alpha, beta, gamma as in Miyamoto
        double alpha = bPrime2.x * (rotated_b0.x - rotated_c0.x) +
                       bPrime2.y * (rotated_b0.y ) + 
                       cPrime2.y * (rotated_c0.y );
        double beta =  bPrime2.x * (rotated_c0.y - rotated_b0.y) + 
                       bPrime2.y * (rotated_b0.x) + 
                       cPrime2.y * (rotated_c0.x);
        double gamma = (rotated_b0.x * rotated_b1.y) - 
                       (rotated_b1.x * rotated_b0.y) + 
                       (rotated_c0.x * rotated_c1.y) - 
                       (rotated_c1.x * rotated_c0.y);

        // sin(theta) = ( alpha * gamma - beta * sqrt(alpha^2 + beta^2 - gamma^2)) / (alpha^2 + beta^2)
        double alphaSqrBetaSqr = (alpha * alpha) + (beta * beta);
        double sinTheta = (alpha * gamma - (beta * sqrt(alphaSqrBetaSqr - (gamma*gamma)))) / (alphaSqrBetaSqr);
    
        double cosThetaSqr = 1.0 - (sinTheta * sinTheta);
        double cosTheta = sqrt(cosThetaSqr);
        if (cosThetaSqr <= 0.0) {
            printf("Error settling molecule %d; cosThetaSqr value %f\nAborting.", idx, cosThetaSqr);
            assert(cosThetaSqr > 0.0);
        }

        // so, make aPrime3, bPrime3, cPrime3 coordinates

        // note that aPrime2.x is zero... compare with Miyamoto eqns A4
        double3 aPrime3 = make_double3(-aPrime2.y*sinTheta,
                                      aPrime2.y*cosTheta,
                                      aPrime2.z);

        double3 bPrime3 = make_double3(bPrime2.x * cosTheta - bPrime2.y * sinTheta,
                                     bPrime2.x * sinTheta + bPrime2.y * cosTheta,
                                     bPrime2.z);
        double3 cPrime3 = make_double3(cPrime2.x * cosTheta - cPrime2.y * sinTheta,
                                     cPrime2.x * sinTheta + cPrime2.y * cosTheta,
                                     cPrime2.z);

        // and put back in to our original coordinate system - use /transposed/ axes
        double3 a_pos = rotation(aPrime3, axis1T, axis2T, axis3T);
        double3 b_pos = rotation(bPrime3, axis1T, axis2T, axis3T);
        double3 c_pos = rotation(cPrime3, axis1T, axis2T, axis3T);

        // while everything is reduced (COM @ origin)... get the differential contribution to the velocity
        double3 dvO = (a_pos - posO) / dt;
        double3 dvH1= (b_pos - posH1)/ dt;
        double3 dvH2= (c_pos - posH2)/ dt;
        
        // add back the COM
        a_pos += COM_d1;
        b_pos += COM_d1;
        c_pos += COM_d1;

        // and now make float4
        float4 posO_new = make_float4(a_pos.x, a_pos.y, a_pos.z,posO_whole.w);
        float4 posH1_new= make_float4(b_pos.x, b_pos.y, b_pos.z,posH1_whole.w);
        float4 posH2_new= make_float4(c_pos.x, c_pos.y, c_pos.z,posH2_whole.w);

        // set the global variables xs to the new values
        xs[idxO] = posO_new;
        xs[idxH1]= posH1_new;
        xs[idxH2]= posH2_new;

        // add a velocity component corresponding to the displacements of the vertices
        // TODO: this contributes to the virial!
        //       ---- algebraic equivalence ??!! We need a per-atom sum here...
        
        // NOTE XXX: order is important here, do not do this after moving a_pos /up/
        // by += COM_d1 while posO (& posH1 & posH2) are still at the origin!!!!
        //  ---- if you do, life will be unpleasant
        //float3 dvO = (a_pos - posO) / dt;
        //float3 dvH1= (b_pos - posH1)/ dt;
        //float3 dvH2= (c_pos - posH2)/ dt;
        
        
        float4 velO_whole = vs[idxO];
        float4 velH1_whole= vs[idxH1];
        float4 velH2_whole= vs[idxH2];

        // add the float3 differential velocity vectors
        
        velO_whole += dvO;
        velH1_whole += dvH1;
        velH2_whole += dvH2;
        
        // and assign the new vectors to global memory
        vs[idxO] = velO_whole;
        vs[idxH1]= velH1_whole;
        vs[idxH2]= velH2_whole;
    }
}

// so, removeNDF is called by instances of DataComputerTemperature on instantiation,
// which occurs (see IntegratorVerlet.cu, Integrator.cu) after 
int FixRigid::removeNDF() {
    int ndf = 0;

    if (TIP4P) {
    // so, for each molecule, we have three constraints on the positions of the real atoms:
    // -- two OH bond lengths, and the angle between them (+3)
    // we have three constraints on the velocities along the length of the bonds:
    // -- the relative velocities along the length of the bonds must be zero (+3)
    // and we have three constraints on the virtual site (its position is completely defined) (+3)
    // 
    // in total, we have a reduction in DOF of 9 per molecule if this is a 4-site model
        ndf = 6 * nMolecules;
        
    } else {
    
    // so, for each molecule, we have three constraints on the positions of the real atoms:
    // -- two OH bond lengths, and the angle between them (+3)
    // we have three constraints on the velocities along the length of the bonds:
    // -- the relative velocities along the length of the bonds must be zero (+3)
    // 
    // in total, we have a reduction in DOF of 6 per molecule
        ndf = 3 * nMolecules;

    }

    
    return ndf;
}

// 
void FixRigid::populateRigidData() {
    // so at this point we have populated canonicalTriangle and sideLengths, which hold
    // the XYZ coordinate axes of the canonical triangle (see ref: Miyamoto) and the actual 
    // side lengths of the water molecule (OH, OH, HH), respectively

    // canonicalTriangle = double4(ra, rb, rc, 0.0)
    // sideLengths = double4(OH, OH, HH, OM)
    
    // some trigonometry here
    double cosC = fixRigidData.canonicalTriangle.z / fixRigidData.sideLengths.y;
    
    // isosceles triangles
    double cosB = cosC;
   
    // algebraic rearrangement of Law of Cosines, and solving for cosA:

    // a is the HH bond length, b is OH bond length, c is OH bond length (b == c)
    double a = fixRigidData.sideLengths.z;
    double b = fixRigidData.sideLengths.x;

    // b == c..
    double cosA = ((-1.0 * a * a ) + (b*b + b*b) ) / (2.0 * b * b);
    
    /*
     * Set the members of fixRigidData to the computed values
     */
    
    fixRigidData.cosA = cosA;
    fixRigidData.cosC = cosC;
    fixRigidData.cosB = cosB;

    // sum of the angles should be pi radians - if not, exit the simulation.
    double sumAngles = acos(cosA) + acos(cosB) + acos(cosC);
    if ( fabs(sumAngles - M_PI) > 0.000001) {
        printf("The sum of the angles for the triangle in SETTLE was found to be %3.10f radians, rather than pi radians. Aborting.\n",sumAngles);
        printf("cosA: %3.6f\ncosB: %3.6f\ncosC: %3.6f\nwith theta values of %f, %f %f\n", 
               cosA, cosB, cosC, acos(cosA), acos(cosB), acos(cosC));
        mdError("Sum of the interior angles of the triangle do not add up to pi radians in FixRigid.\n");
    }

    // also, get local variables of denominator, ma, mb, mc so that we can write the expressions concisely.

    // mass of Oxygen and Hydrogen respectively
    // --- these are initialized in ::set_fixed_sides()
    double ma = fixRigidData.weights.z;
    double mb = fixRigidData.weights.w;
    //double mc = mb;

    // first, we need to calculate the denominator expression, d
    double d = (1.0 / (2.0 * mb) ) * ( (2.0 * ( (ma+mb) * (ma+mb) ) ) + 
                                               (2.0 * ma * mb * cosA * cosB * cosC) - 
                                               (2.0 * mb * mb * cosA * cosA) - 
                                               (ma * (ma + mb) * ( (cosB * cosB) + (cosC * cosC) ) ) ) ;

    //printf("denominator expression evaluates to: %3.10f\n", d);

    // filling in the values for the tau expressions, so that we can do the velocity constraints
    // --- see Miyamoto, equations B2.
    //     Note that these are correction velocities w.r.t. the /solved/ positions at the end of the timestep,
    //     and so cosA, cosB, cosC etc. are all constants.  
    //     we are re-distributing the momenta s.t. the molecule is 
    //     only undergoing translational and rotational motion, and thus the COM has 6 DOF; the atoms do not.

    // these are just the factors, in order of listing, in equations B2 of Miyamoto
    double tauAB1 = (ma / d) * ( (2.0 * (ma + mb)) - (ma * cosC * cosC));
    double tauAB2 = (ma / d) * ( (mb * cosC * cosA) - ( (ma + mb) * cosB));
    double tauAB3 = (ma / d) * ( (ma * cosB * cosC) - (2.0 * mb * cosA));

    double tauBC1 = ( ((ma + mb) * (ma + mb)) - (mb * mb * cosA * cosA) ) / d;
    double tauBC2 = (ma / d) * ( (mb * cosA * cosB ) - ( (ma + mb) * cosC) );
    double tauBC3 = (ma / d) * ( (mb * cosC * cosA ) - ( (ma + mb) * cosB) );

    double tauCA1 = (ma / d) * ( (2.0 * (ma + mb) ) - ( ma * cosB * cosB) );
    double tauCA2 = (ma / d) * ( (ma * cosB * cosC) - (2.0 * mb * cosA));
    double tauCA3 = (ma / d) * ( (mb * cosA * cosB) - ( (ma + mb) * cosC));

    // and set the values to fixRigidData!! v. important...

    fixRigidData.tauAB1 = tauAB1;
    fixRigidData.tauAB2 = tauAB2;
    fixRigidData.tauAB3 = tauAB3;

    fixRigidData.tauBC1 = tauBC1;
    fixRigidData.tauBC2 = tauBC2;
    fixRigidData.tauBC3 = tauBC3;

    fixRigidData.tauCA1 = tauCA1;
    fixRigidData.tauCA2 = tauCA2;
    fixRigidData.tauCA3 = tauCA3;

    fixRigidData.denominator = d;

    /*
    printf("values for tau constants: \n%f %f %f\n%f %f %f\n%f %f %f\n",
           tauAB1, tauAB2, tauAB3,
           tauBC1, tauBC2, tauBC3, 
           tauCA1, tauCA2, tauCA3);
    */

    return;

}

/*
bool FixRigid::postNVE_V() {

    float dt = state->dt;
    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();
    BoundsGPU &bounds = state->boundsGPU;
    int nAtoms = state->atoms.size();
    // first, unconstrained velocity update continues: distribute the force from the M-site
    //        and integrate the velocities accordingly.  Update the forces as well.
    // Next,  do compute_SETTLE as usual on the (as-yet) unconstrained positions & velocities

    // from IntegratorVerlet
    float dtf = 0.5f * state->dt * state->units.ftm_to_v;
    // need to fix the velocities before position integration
    settleVelocities<FixRigidData, true><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                                xs_0.data(), gpd.vs(activeIdx), vs_0.data(), 
                                                dvs_0.data(), gpd.fs(activeIdx), fs_0.data(), 
                                                com.data(), fixRigidData, nMolecules, dt, dtf, 
                                                gpd.idToIdxs.d_data.data(), bounds);
    return true;
}
*/
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
                                                         gpd.fs(activeIdx),
                                                         nAtoms);
            cudaDeviceSynchronize();
        }
        // get a few pieces of data as required
        // -- all we're doing here is setting the position of the M-Site prior to computing the forces
        //    within the simulation.  Otherwise, the M-Site will likely be far away from where it should be, 
        //    relative to the molecule.  We do not solve the constraints on the rigid body at this time.
        // we need to reset the position of the M-Site prior to calculating the forces
        
        setMSite<FixRigidData><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.idToIdxs.d_data.data(), gpd.xs(activeIdx), nMolecules, bounds, fixRigidData);
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

        // how do we enforce that consistent scaling is achieved here? 
        // where consistent is understood to be w.r.t. how the user scales the LJ interactions, charge-charge interactions, etc.
        // since this is modular - i.e., FixRigid just maintains the geometry and partitions M-site forces (if necessary);
        // ---- needs to be given more thought
        r_OH /= sigma_O;
        r_OM /= sigma_O;
        r_HH /= sigma_O;
        mdError("Currently, real units must be used for simulations of rigid water models.\n");
    }

    fixedSides = make_double4(r_OH, r_OH, r_HH, r_OM);

    // set the sideLengths variable in fixRigidData to the fixedSides values
    fixRigidData.sideLengths = fixedSides;
    return;

}

std::string FixRigid::restartChunk(std::string format) {
    std::stringstream ss;
    
    ss << "<atomsInMolecule n=\"" << waterIds.size() << "\">\n";
    for (int4 &atomIds : waterIds) {
        ss << atomIds.x << " " << atomIds.y << " " << atomIds.z << " " << atomIds.w << "\n";
    }
    ss << "</atomsInMolecule>\n";

    ss << "<style type=\'" << style << "\'>\n";
    ss << "</style>\n";
    if (TIP4P) {
        ss << "<model fourSite=\'true\'/>\n";
    } else {
        ss << "<model fourSite=\'false\'/>\n";
    }

    return ss.str();
}

bool FixRigid::readFromRestart() {
    auto restData = getRestartNode();

    if (restData) {

        auto curr_param = restData.first_child();
        while (curr_param) {
            std::string tag = curr_param.name();
            if (tag == "atomsInMolecule") {
                int n = boost::lexical_cast<int>(curr_param.attribute("n").value());
                std::vector<int4> atomsInMolecule(n);
                // we need to pass this data to atomsInMolecule, which will then be assigned to waterIds;
                // and while we are here, we may as well make the bonds
                // -- note: even in a 3-site model (e.g. TIP3P), we use int4's. so, there shouldn't be an issue here, although some xml file size bloat occurs. Something to address later.
                xml_assignValues<int, 4>(curr_param, [&] (int i, int *vals) {
                                            //int id = vals[3];
                                            atomsInMolecule[i] = make_int4(vals[0], vals[1], vals[2], vals[3]);
                                            });

                // and assign our class member 'waterIds' the data denoted by atomsInMolecule
                waterIds = atomsInMolecule;
            } else if (tag == "style") {
                std::string thisStyle = boost::lexical_cast<std::string>(curr_param.attribute("type").value());
                style = thisStyle;
                printf("In FixRigid::readFromRestart(), found style %s\n", style.c_str());
            } else if (tag == "model") {
                bool fourSite = boost::lexical_cast<bool>(curr_param.attribute("fourSite").value());
                // set our class member boolean flags TIP3P & TIP4P
                TIP4P = fourSite;
                TIP3P = !fourSite;
            }
            curr_param = curr_param.next_sibling();
        }

    }

    
    for (int i = 0; i < waterIds.size(); i++) {
        Bond bondOH1;
        Bond bondOH2;
        Bond bondHH;
        bondOH1.ids = { {waterIds[i].x,waterIds[i].y} };
        bondOH2.ids = { {waterIds[i].x,waterIds[i].z} };
        bondHH.ids =  { {waterIds[i].y,waterIds[i].z} };

        bonds.push_back(bondOH1);
        bonds.push_back(bondOH2);
        bonds.push_back(bondHH);
        if (TIP4P) {
            Bond bondOM;
            bondOM.ids = { {waterIds[i].x, waterIds[i].w } };
            bonds.push_back(bondOM);
        }
    }

    std::cout << "There are " << waterIds.size() << " molecules read in from the restart file and " << bonds.size() << " bonds were made in FixRigid.\n";
    return true;
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
    Bond bondOM;

    bondOH1.ids = { {waterMol.x,waterMol.y} };
    bondOH2.ids = { {waterMol.x,waterMol.z} };
    bondHH.ids =  { {waterMol.y,waterMol.z} };
    bondOM.ids =  { {waterMol.x,waterMol.w} };

    bonds.push_back(bondOH1);
    bonds.push_back(bondOH2);
    bonds.push_back(bondHH);
    bonds.push_back(bondOM);

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
    set_fixed_sides();
    populateRigidData();

    printf("number of molecules in waterIds: %d\n", nMolecules);
    waterIdsGPU = GPUArrayDeviceGlobal<int4>(nMolecules);
    waterIdsGPU.set(waterIds.data());



    xs_0 = GPUArrayDeviceGlobal<float4>(3*nMolecules);
    vs_0 = GPUArrayDeviceGlobal<float4>(3*nMolecules);
    fs_0 = GPUArrayDeviceGlobal<float4>(3*nMolecules);
    com = GPUArrayDeviceGlobal<float4>(nMolecules);
    constraints = GPUArrayDeviceGlobal<bool>(nMolecules);
    com.set(invMassSums.data());

    float dt = state->dt;
    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();
    BoundsGPU &bounds = state->boundsGPU;
    int nAtoms = state->atoms.size();
    float dtf = 0.5f * state->dt * state->units.ftm_to_v;
    
    //SAFECALL((printGPD_Rigid<<<NBLOCK(nAtoms), PERBLOCK>>>(gpd.ids(activeIdx),gpd.xs(activeIdx),gpd.vs(activeIdx),gpd.fs(activeIdx),nAtoms)));

    //printf("Completed printing GPD data in FixRigid::prepareForRun, before doing anything.\n\n");
    // compute the force partition constant
    if (TIP4P) {
        compute_gamma();
    }


    compute_COM<FixRigidData><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), 
                                                                gpd.xs(activeIdx), 
                                                                gpd.vs(activeIdx), 
                                                                gpd.idToIdxs.d_data.data(), 
                                                                nMolecules, com.data(), 
                                                                bounds, fixRigidData);

    //set_fixed_sides();
    
    compute_prev_val<<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), xs_0.data(), 
                                                gpd.vs(activeIdx), vs_0.data(), gpd.fs(activeIdx), 
                                                fs_0.data(), nMolecules, gpd.idToIdxs.d_data.data());


    settlePositions<FixRigidData, true><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                                xs_0.data(), gpd.vs(activeIdx), vs_0.data(), 
                                                gpd.fs(activeIdx), fs_0.data(), 
                                                com.data(), fixRigidData, nMolecules, dt, dtf, 
                                                gpd.idToIdxs.d_data.data(), bounds, (int) state->turn);
   

    SAFECALL((settleVelocities<FixRigidData,false, true><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                                xs_0.data(), gpd.vs(activeIdx), vs_0.data(), 
                                                gpd.fs(activeIdx), fs_0.data(), 
                                                com.data(), fixRigidData, nMolecules, dt, dtf, 
                                                gpd.idToIdxs.d_data.data(), bounds, (int) state->turn)));

    // and remove the COMV we may have acquired from settling the velocity constraints just now.
    GPUArrayDeviceGlobal<float4> sumMomentum = GPUArrayDeviceGlobal<float4>(2);

    CUT_CHECK_ERROR("Error occurred during solution of velocity constraints.");
    sumMomentum.memset(0);
    int warpSize = state->devManager.prop.warpSize;

    float3 dimsFloat3 = make_float3(1.0, 1.0, 1.0);
    accumulate_gpu<float4, float4, SumVectorXYZOverW, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(float4)>>>
            (
             sumMomentum.data(),
             gpd.vs(activeIdx),
             nAtoms,
             warpSize,
             SumVectorXYZOverW()
            );

    rigid_remove_COMV<<<NBLOCK(nAtoms), PERBLOCK>>>(nAtoms, gpd.vs(activeIdx), sumMomentum.data(), dimsFloat3);

    CUT_CHECK_ERROR("Removal of COMV in FixRigid failed.");
    
    // adjust the initial velocities to conform to our velocity constraints
    // -- here, we preserve the COMV of the system, while imposing strictly translational motion on the molecules
    //    - we use shared memory to compute center of mass velocity of the group, allowing for one kernel call
    /*
       // note that this kernel declaration above is commented out as well
    adjustInitialVelocities<<<NBLOCK(nMolecules), PERBLOCK, nMolecules*sizeof(float3)>>>(
                            waterIdsGPU.data(), gpd.idToIdxs.d_data.data(),
                            gpd.xs(activeIdx), gpd.vs(activeIdx), nMolecules,
                            bounds);
    */
    // make sure all of the initial velocities have been adjusted prior to validating the constraints
    cudaDeviceSynchronize();
    
    // validate that we have good initial conditions
    SAFECALL((validateConstraints<FixRigidData> <<<NBLOCK(nMolecules), PERBLOCK>>> (waterIdsGPU.data(), gpd.idToIdxs.d_data.data(), 
                                                           gpd.xs(activeIdx), gpd.vs(activeIdx), 
                                                           nMolecules, bounds, fixRigidData, 
                                                           constraints.data(), state->turn)));
    cudaDeviceSynchronize();

    CUT_CHECK_ERROR("Validation of constraints failed in FixRigid.");
    
    //SAFECALL((printGPD_Rigid<<<NBLOCK(nAtoms), PERBLOCK>>>(gpd.ids(activeIdx),gpd.xs(activeIdx),gpd.vs(activeIdx),gpd.fs(activeIdx),nAtoms)));

    //cudaDeviceSynchronize();
    prepared = true;
    return prepared;
}

bool FixRigid::stepInit() {
    
    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();
    BoundsGPU &bounds = state->boundsGPU;
    //float dtf = 0.5f * state->dt * state->units.ftm_to_v;
    int nAtoms = state->atoms.size();
    //float dt = state->dt;
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

bool FixRigid::stepFinal() {
    float dt = state->dt;
    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();
    BoundsGPU &bounds = state->boundsGPU;
    int nAtoms = state->atoms.size();
    // first, unconstrained velocity update continues: distribute the force from the M-site
    //        and integrate the velocities accordingly.  Update the forces as well.
    // Next,  do compute_SETTLE as usual on the (as-yet) unconstrained positions & velocities

    // from IntegratorVerlet
    float dtf = 0.5f * state->dt * state->units.ftm_to_v;
    if (TIP4P) {

        /*
        if (printing) {
            cudaDeviceSynchronize();
            printf("Calling printGPD_Rigid at turn %d\n in FixRigid::stepFinal, before doing anything\n", (int) state->turn);
            printGPD_Rigid<<<NBLOCK(nAtoms), PERBLOCK>>>(gpd.ids(activeIdx),gpd.xs(activeIdx),gpd.vs(activeIdx),gpd.fs(activeIdx),nAtoms);
            cudaDeviceSynchronize();
        }
        */
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

    settlePositions<FixRigidData, true><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                                xs_0.data(), gpd.vs(activeIdx), vs_0.data(), 
                                                gpd.fs(activeIdx), fs_0.data(), 
                                                com.data(), fixRigidData, nMolecules, dt, dtf, 
                                                gpd.idToIdxs.d_data.data(), bounds, (int) state->turn);
    
    
    settleVelocities<FixRigidData,false, true><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                                xs_0.data(), gpd.vs(activeIdx), vs_0.data(), 
                                                gpd.fs(activeIdx), fs_0.data(), 
                                                com.data(), fixRigidData, nMolecules, dt, dtf, 
                                                gpd.idToIdxs.d_data.data(), bounds, (int) state->turn);
    
    if (TIP4P) {
        /*
        if (printing) { 
            printf("Calling printGPD_Rigid at turn %d\n in FixRigid::stepFinal, before calling setMSite \n", (int) state->turn);
            cudaDeviceSynchronize();
            printGPD_Rigid<<<NBLOCK(nAtoms), PERBLOCK>>>(gpd.ids(activeIdx),gpd.xs(activeIdx),gpd.vs(activeIdx),gpd.fs(activeIdx),nAtoms);
        
            cudaDeviceSynchronize();
        }
        */
        setMSite<FixRigidData><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.idToIdxs.d_data.data(), gpd.xs(activeIdx), nMolecules, bounds,fixRigidData);
        
        /*
        if (printing) { 
            cudaDeviceSynchronize();
            printf("Calling printGPD_Rigid at turn %d\n in FixRigid::stepFinal, after calling setMSite \n", (int) state->turn);
            printGPD_Rigid<<<NBLOCK(nAtoms), PERBLOCK>>>(gpd.ids(activeIdx),gpd.xs(activeIdx),gpd.vs(activeIdx),gpd.fs(activeIdx),nAtoms);
            cudaDeviceSynchronize();
        }
        */
    }
 
    validateConstraints<FixRigidData><<<NBLOCK(nMolecules), PERBLOCK>>> (waterIdsGPU.data(), 
                                                                                   gpd.idToIdxs.d_data.data(), 
                                                                                   gpd.xs(activeIdx), 
                                                                                   gpd.vs(activeIdx), 
                                                                                   nMolecules, bounds, 
                                                                                   fixRigidData, 
                                                                                   constraints.data(), 
                                                                                   (int) state->turn);
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



