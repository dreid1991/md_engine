#include "FixRigid.h"

#include "State.h"
#include "Atom.h"
#include "VariantPyListInterface.h"
#include "boost_for_export.h"
#include "cutils_math.h"
#include "cutils_func.h"
#include <math.h>
#include "globalDefs.h"
#include "xml_func.h"
#include "helpers.h"


namespace py = boost::python;
const std::string rigidType = "Rigid";

FixRigid::FixRigid(boost::shared_ptr<State> state_, std::string handle_, std::string style_) : Fix(state_, handle_, "all", rigidType, true, true, false, 1), style(style_) {

    // set both to false initially; using one of the createRigid functions will flip the pertinent flag to true
    FOURSITE = false;
    THREESITE = false;
    printing = false;
    // set flag 'requiresPostNVE_V' to true
    requiresPostNVE_V = true;
    // this fix requires the forces to have already been computed before we can 
    // call prepareForRun()
    requiresForces = true;
    solveInitialConstraints = true;
    // this is the list of style arguments currently supported.
    if ( style != "TIP4P/2005" && 
         style != "TIP3P"      &&
         style != "TIP4P"      &&
         style != "SPC"        && 
         style != "SPC/E"        ) {
        mdError("Unsupported style argument passed to the Rigid fix; current, style options are TIP4P, SPC, SPC/E, TIP4P/2005 and TIP3P; Please adjust accordingly.");
    }

    readFromRestart();

}


inline __host__ __device__ real3 rotation(real3 vector, real3 X, real3 Y, real3 Z) {
    return make_real3(dot(X,vector), dot(Y,vector), dot(Z, vector));
}

#ifndef DASH_DOUBLE
inline __host__ __device__ double3 rotation(double3 vector, double3 X, double3 Y, double3 Z) {
    return make_double3(dot(X,vector), dot(Y,vector), dot(Z, vector));
}
#endif

inline __host__ __device__ void computeVirial_double(Virial &v, double3 force, double3 dr) {
    v[0] += force.x * dr.x;
    v[1] += force.y * dr.y;
    v[2] += force.z * dr.z;
    v[3] += force.x * dr.y;
    v[4] += force.x * dr.z;
    v[5] += force.y * dr.z;
}


// verified to be correct
inline __host__ __device__ double matrixDet(double3 ROW1, double3 ROW2, double3 ROW3) 
{
    // [a   b    c]
    // [d   e    f]  = A
    // [g   h    i]
    //
    // det(A) = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
    //
    //
    double det;
    // so, get the first value..
    //         a   * (  e    *   i    -   f    *   h   )
    det =   ROW1.x * (ROW2.y * ROW3.z - ROW2.z * ROW3.y);
    //  -      b   * (  d    *   i    -   f    *   g   )
    det -= (ROW1.y * (ROW2.x * ROW3.z - ROW2.z * ROW3.x));
    //  +      c   * (  d    *   h    -   e    *   g   )
    det += (ROW1.z * (ROW2.x * ROW3.y - ROW2.y * ROW3.x));
    return det;
}


// kind of just need this for validation that M*M^-1 results in the identity
// ---- and it does! May as well keep it, though.
// verified to be correct
inline __host__ __device__ void matrixMultiplication(double3 A1, double3 A2, double3 A3,
                                                     double3 B1, double3 B2, double3 B3,
                                                     double3 &C1, double3 &C2, double3 &C3) {


    C1.x = A1.x * B1.x + A1.y * B2.x + A1.z * B3.x;
    C2.x = A2.x * B1.x + A2.y * B2.x + A2.z * B3.x;
    C3.x = A3.x * B1.x + A3.y * B2.x + A3.z * B3.x;

    C1.y = A1.x * B1.y + A1.y * B2.y + A1.z * B3.y;
    C2.y = A2.x * B1.y + A2.y * B2.y + A2.z * B3.y;
    C3.y = A3.x * B1.y + A3.y * B2.y + A3.z * B3.y;

    C1.z = A1.x * B1.z + A1.y * B2.z + A1.z * B3.z;
    C2.z = A2.x * B1.z + A2.y * B2.z + A2.z * B3.z;
    C3.z = A3.x * B1.z + A3.y * B2.z + A3.z * B3.z;

}

inline __host__ __device__ double3 matrixVectorMultiply(double3 A1, double3 A2, double3 A3, double3 V)
{
    return make_double3( A1.x * V.x + A1.y * V.y + A1.z * V.z,
                         A2.x * V.x + A2.y * V.y + A2.z * V.z,
                         A3.x * V.x + A3.y * V.y + A3.z * V.z);


}

// verified to be correct
inline __host__ __device__ void invertMatrix(double3 ROW1, double3 ROW2, double3 ROW3, double3 &invROW1, double3 &invROW2, double3 &invROW3)
{
    // get the inverse determinant, and then compute the individual elements of the rows as required for the 
    // inverse 'matrix': three rows of double3's
    double det = matrixDet(ROW1, ROW2, ROW3);
    double invDet = 1.0 / det;

    invROW1.x =        invDet * (ROW2.y * ROW3.z - ROW3.y * ROW2.z);
    invROW1.y = -1.0 * invDet * (ROW1.y * ROW3.z - ROW3.y * ROW1.z);
    invROW1.z =        invDet * (ROW1.y * ROW2.z - ROW2.y * ROW1.z);

    invROW2.x = -1.0 * invDet * (ROW2.x * ROW3.z - ROW3.x * ROW2.z);
    invROW2.y =        invDet * (ROW1.x * ROW3.z - ROW3.x * ROW1.z);
    invROW2.z = -1.0 * invDet * (ROW1.x * ROW2.z - ROW2.x * ROW1.z);
    
    invROW3.x =        invDet * (ROW2.x * ROW3.y - ROW3.x * ROW2.y);
    invROW3.y = -1.0 * invDet * (ROW1.x * ROW3.y - ROW3.x * ROW1.y);
    invROW3.z =        invDet * (ROW1.x * ROW2.y - ROW2.x * ROW1.y);

    // and done

}


namespace {
static inline double series_sinhx(double x) {
    double xSqr = x*x;
    return (1.0 + (xSqr/6.0)*(1.0 + (xSqr/20.0)*(1.0 + (xSqr/42.0)*(1.0 + (xSqr/72.0)*(1.0 + (xSqr/110.0))))));
}


// so this is the exact same function as in FixLinearMomentum. 
__global__ void rigid_remove_COMV(int nAtoms, real4 *vs, real4 *sumData, real3 dims) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        real4 v = vs[idx];
        real4 sum = sumData[0];
        real invMassTotal = 1.0f / sum.w;
        v.x -= sum.x * invMassTotal * dims.x;
        v.y -= sum.y * invMassTotal * dims.y;
        v.z -= sum.z * invMassTotal * dims.z;
        vs[idx] = v;
    }
}

template <class DATA, bool FOURSITE>
__global__ void rigid_scaleSystem_cu(int4* waterIds, real4* xs, int* idToIdxs,
                                     real3 lo, real3 rectLen, BoundsGPU bounds, real3 scaleBy,
                                     DATA fixRigidData, int nMolecules) {

    int idx = GETIDX();
    if (idx < nMolecules) {
        // compute the COM; 
        // perform the displacement;
        // compute the difference in the positions
        // translate all the individual atoms xs's accordingly, and exit
        // --- we do not need to check groupTags
        // --- if (FOURSITE) { translate M-site position as well }
        double weightH = fixRigidData.weights.y;
       
        // get the molecule at this idx
        int4 atomsFromMolecule = waterIds[idx];

        // get the atom idxs
        int idxO = idToIdxs[atomsFromMolecule.x];
        int idxH1= idToIdxs[atomsFromMolecule.y];
        int idxH2= idToIdxs[atomsFromMolecule.z];
        
        // extract the whole positions
        real4 posO_whole = xs[idxO];
        real4 posH1_whole= xs[idxH1];
        real4 posH2_whole= xs[idxH2];

        // and the unconstrained, updated positions
        double3 posO = make_double3(posO_whole);
        double3 posH1= make_double3(posH1_whole);
        double3 posH2= make_double3(posH2_whole);
    
        // get the relative vectors for the unconstrained triangle
        // r_ij = r_j - r_i?
        double3 OH1_unconstrained = bounds.minImage(posH1 - posO);
        double3 OH2_unconstrained = bounds.minImage(posH2 - posO);

        // move the hydrogens to their minimum image distance w.r.t. the oxygen
        posH1 = posO + OH1_unconstrained;
        posH2 = posO + OH2_unconstrained;

        // the center of mass 'd1' in the paper
        double3 COM_d1 = (posO) + ( ( OH1_unconstrained + OH2_unconstrained ) * weightH);
        
        // displacements from the center of mass, per particle
        double3 dx_O   = bounds.minImage(posO  - COM_d1);
        double3 dx_H1  = bounds.minImage(posH1 - COM_d1);
        double3 dx_H2  = bounds.minImage(posH2 - COM_d1);

        //do as Mod::scale on COM_d1
#ifdef DASH_DOUBLE
        double3 center = lo + rectLen * 0.5;
        double3 newRel = (COM_d1 - center) * (scaleBy);
#else
        double3 center = make_double3(lo) + make_double3(rectLen) * 0.5;
        double3 newRel = (COM_d1 - center) * make_double3(scaleBy);
#endif
        double3 newCOM = center + newRel;
        posO   += (newCOM - COM_d1); 
        posH1  += (newCOM - COM_d1); 
        posH2  += (newCOM - COM_d1);

#ifdef DASH_DOUBLE
        real3 newPosO = posO;
        real3 newPosH1= posH1;
        real3 newPosH2= posH2;
#else
        real3 newPosO = make_real3(posO);
        real3 newPosH1= make_real3(posH1);
        real3 newPosH2= make_real3(posH2);
#endif
        xs[idxO] = make_real4(newPosO, posO_whole.w);
        xs[idxH1]= make_real4(newPosH1,posH1_whole.w);
        xs[idxH2]= make_real4(newPosH2,posH2_whole.w);
    }
}



template <class DATA, bool FOURSITE>
__global__ void rigid_scaleSystemGroup_cu(int4* waterIds, real4* xs, int* idToIdxs,
                                          real3 lo, real3 rectLen, BoundsGPU bounds, real3 scaleBy,
                                          DATA fixRigidData, int nMolecules, uint32_t groupTag,
                                          real4* fs) {

    int idx = GETIDX();
    if (idx < nMolecules) {
        // get the molecule at this idx
        int4 atomsFromMolecule = waterIds[idx];
        int idxO = idToIdxs[atomsFromMolecule.x];
        uint32_t tag = * (uint32_t *) &(fs[idxO].w);
        if (tag & groupTag) {
            // compute the COM; 
            // perform the displacement;
            // compute the difference in the positions
            // translate all the individual atoms xs's accordingly, and exit
            // ---- check the groupTag of just the oxygen; if 
            //      the oxygen atom is in the group, we will assume that the other atoms are as well
            //      mostly because if they aren't, then something is being done incorrectly by the user
            // --- if (FOURSITE) { translate M-site position as well }
            //double weightO = fixRigidData.weights.x;
            double weightH = fixRigidData.weights.y;
           

            // get the atom idxs
            int idxH1= idToIdxs[atomsFromMolecule.y];
            int idxH2= idToIdxs[atomsFromMolecule.z];
            
            // extract the whole positions
            real4 posO_whole = xs[idxO];
            real4 posH1_whole= xs[idxH1];
            real4 posH2_whole= xs[idxH2];

            // and the unconstrained, updated positions
            double3 posO = make_double3(posO_whole);
            double3 posH1= make_double3(posH1_whole);
            double3 posH2= make_double3(posH2_whole);
        
            // get the relative vectors for the unconstrained triangle
            double3 OH1_unconstrained = bounds.minImage(posH1 - posO);
            double3 OH2_unconstrained = bounds.minImage(posH2 - posO);

            // move the hydrogens to their minimum image distance w.r.t. the oxygen
            posH1 = posO + OH1_unconstrained;
            posH2 = posO + OH2_unconstrained;

            // the center of mass 'd1' in the paper
            double3 COM_d1 = (posO) + ( ( OH1_unconstrained + OH2_unconstrained ) * weightH);
            // do as Mod::scale on COM_d1
            // displacements from the center of mass, per particle
            double3 dx_O   = posO  - COM_d1 ;
            double3 dx_H1  = posH1 - COM_d1;
            double3 dx_H2  = posH2 - COM_d1;

        //do as Mod::scale on COM_d1
#ifdef DASH_DOUBLE
            double3 center = lo + rectLen * 0.5;
            double3 newRel = (COM_d1 - center) * (scaleBy);
#else
            double3 center = make_double3(lo) + make_double3(rectLen) * 0.5;
            double3 newRel = (COM_d1 - center) * make_double3(scaleBy);
#endif

            if (FOURSITE) {
                int idxM = idToIdxs[atomsFromMolecule.w];
                real4 posM_whole = xs[idxM];
                double3 posM = make_double3(posM_whole);
                double3 posM_relO = bounds.minImage(posM - posO);
                double3 dx_M = bounds.minImage(posM_relO - COM_d1);
                posM = dx_M + center + newRel;
                //posM += diff;
#ifdef DASH_DOUBLE
                real3 newPosM = posM;
#else 
                real3 newPosM = make_real3(posM);
#endif
                xs[idxM] = make_real4(newPosM, posM_whole.w);
            }

            posO   = dx_O  + center + newRel;
            posH1  = dx_H1 + center + newRel;
            posH2  = dx_H2 + center + newRel;

#ifdef DASH_DOUBLE
            real3 newPosO = posO;
            real3 newPosH1= posH1;
            real3 newPosH2= posH2;
#else
            real3 newPosO = make_real3(posO);
            real3 newPosH1= make_real3(posH1);
            real3 newPosH2= make_real3(posH2);
#endif 
            xs[idxO] = make_real4(newPosO, posO_whole.w);
            xs[idxH1]= make_real4(newPosH1,posH1_whole.w);
            xs[idxH2]= make_real4(newPosH2,posH2_whole.w);

        }


    }

}

// called at the end of stepFinal, this will be silent unless something is amiss (if the constraints are not satisifed for every molecule)
// i.e., bond lengths should be fixed, and the dot product of the relative velocities along a bond with the 
// bond vector should be identically zero.
template <class DATA>
__global__ void validateConstraints(int4* waterIds, int *idToIdxs, 
                                    real4 *xs, real4 *vs, 
                                    int nMolecules, BoundsGPU bounds, 
                                    DATA fixRigidData, bool* constraints, int turn) {

    int idx = GETIDX();

    if (idx < nMolecules) {

        // extract the ids
        int id_O =  waterIds[idx].x;
        int id_H1 = waterIds[idx].y;
        int id_H2 = waterIds[idx].z;

        // so these are the actual OH, OH, HH, OM bond lengths, not the sides of the canonical triangle
        // save them as double4, cast them as real for arithmetic, since really all calculations are done in real
        double4 sideLengths = fixRigidData.sideLengths;

        // (re)set the constraint boolean
        constraints[idx] = true;

        // get the positions
        real3 pos_O = make_real3(xs[idToIdxs[id_O]]);
        real3 pos_H1 = make_real3(xs[idToIdxs[id_H1]]);
        real3 pos_H2 = make_real3(xs[idToIdxs[id_H2]]);

        // get the velocities
        real3 vel_O = make_real3(vs[idToIdxs[id_O]]);
        real3 vel_H1= make_real3(vs[idToIdxs[id_H1]]);
        real3 vel_H2= make_real3(vs[idToIdxs[id_H2]]);

        // our constraints are that the 
        // --- OH1, OH2, H1H2 bond lengths are the specified values;
        // --- the dot product of the bond vector with the relative velocity along the bond is zero
        
        // take a look at the bond lengths first
        // i ~ O, j ~ H1, k ~ H2
        real3 r_ij = bounds.minImage(pos_H1 - pos_O);
        real3 r_ik = bounds.minImage(pos_H2 - pos_O);
        real3 r_jk = bounds.minImage(pos_H2 - pos_H1);

        real len_rij = length(r_ij);
        real len_rik = length(r_ik);
        real len_rjk = length(r_jk);
    
        // side length AB (AB = AC) (AB ~ intramolecular OH bond)

        real AB = (real) sideLengths.x;
        real BC = (real) sideLengths.z;

        // and these should be ~0.0f
        real bondLenij = len_rij - AB;
        real bondLenik = len_rik - AB;
        real bondLenjk = len_rjk - BC;

        // these values correspond to the fixed side lengths of the triangle congruent to the specific water model being examined

        // now take a look at dot product of relative velocities along the bond vector
        real3 v_ij = vel_H1 - vel_O;
        real3 v_ik = vel_H2 - vel_O;
        real3 v_jk = vel_H2 - vel_H1;

        real mag_v_ij = length(v_ij);
        real mag_v_ik = length(v_ik);
        real mag_v_jk = length(v_jk);

        // this is the "relative velocity along the bond" constraint
        real bond_ij = dot(r_ij, v_ij);
        real bond_ik = dot(r_ik, v_ik);
        real bond_jk = dot(r_jk, v_jk);
        /*
        real constr_relVel_ij = bond_ij / mag_v_ij;
        real constr_relVel_ik = bond_ik / mag_v_ik;
        real constr_relVel_jk = bond_jk / mag_v_jk;
        */
        // 1e-3;
        // so, we only ever have ~7 digits of precision (log10(2^24) = 7.22....) due to the global variables
        real tolerance = 0.001;
        // note that these values should all be zero
        if ( ( bond_ij > tolerance) or 
             ( bond_ik > tolerance) or
             ( bond_jk > tolerance) ) {
            constraints[idx] = false;
            printf("water molecule %d unsatisfied velocity constraints at turn %d,\ndot(r_ij, v_ij) for ij = {01, 02, 12} %f, %f, and %f; tolerance %f\n", idx, (int) turn,
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
    }
}


// distribution of m-site potential energy to appropriate atom (oxygen)
__global__ void distributeMSite_Energy(int nMolecules,
                                       int4 *waterIds, 
                                       int* idToIdxs,
                                       real *perParticleEng) {

    int idx = GETIDX();
    if (idx < nMolecules) {
        // by construction, the id's of the molecules are ordered as follows in waterIds array
        // waterIds contains id's; we need idxs
        int idx_O  = idToIdxs[waterIds[idx].x];
        int idx_M  = idToIdxs[waterIds[idx].w];
        
        real ppe_M = perParticleEng[idx_M];
        real ppe_O = perParticleEng[idx_O];
        ppe_O += ppe_M;
        
        perParticleEng[idx_O] = ppe_O;
        perParticleEng[idx_M] = 0.0;
    }
}


// distribute the m site forces, and do an unconstrained integration of the velocity component corresponding to this additional force
// -- this is required for 4-site models with a massless site.
template <bool VIRIALS,bool FORCES>
__global__ void distributeMSite(int4 *waterIds, real4 *xs, real4 *vs, real4 *fs, 
                                Virial *virials,
                                int nMolecules, real gamma, real dtf, int* idToIdxs, BoundsGPU bounds)

{
    int idx = GETIDX();
    if (idx < nMolecules) {
        // by construction, the id's of the molecules are ordered as follows in waterIds array

        // waterIdxs contains id's; we need idxs
        int idx_O  = idToIdxs[waterIds[idx].x];
        int idx_H1 = idToIdxs[waterIds[idx].y];
        int idx_H2 = idToIdxs[waterIds[idx].z];
        int idx_M  = idToIdxs[waterIds[idx].w];

        if (FORCES) {
            real4 vel_O = vs[idx_O];
            real4 vel_H1 = vs[idx_H1];
            real4 vel_H2 = vs[idx_H2];

            // need the forces from O, H1, H2, and M
            real4 fs_O  = fs[idx_O];
            real4 fs_H1 = fs[idx_H1];
            real4 fs_H2 = fs[idx_H2];
            real4 fs_M  = fs[idx_M];

            // -- these are the forces from the M-site partitioned for distribution to the atoms of the water molecule
            real3 fs_O_d = make_real3(fs_M) * gamma;
            real3 fs_H_d = make_real3(fs_M) * 0.5 * (1.0 - gamma);

            // get the inverse masses from velocity variables above
            real invMassO = vel_O.w;

            // if the hydrogens don't have equivalent masses, we have bigger problems
            real invMassH = vel_H1.w;

            // compute the differential addition to the velocities
            real3 dv_O = dtf * invMassO * fs_O_d;
            real3 dv_H = dtf * invMassH * fs_H_d;

            // and add to the velocities of the atoms
            vel_O  += dv_O;
            vel_H1 += dv_H;
            vel_H2 += dv_H;

            // set the velocities to the new velocities in vel_O, vel_H1, vel_H2
            vs[idx_O] = vel_O; 
            vs[idx_H1]= vel_H1;
            vs[idx_H2]= vel_H2;
           
            vs[idx_M] = make_real4(0,0,0,INVMASSLESS);
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
            fs[idx_M] = make_real4(0.0, 0.0, 0.0,fs_M.w);
        } 


        if (VIRIALS) {
            Virial virialToDistribute = virials[idx_M];
            
            Virial distribute_O = virialToDistribute * (gamma);
            Virial distribute_H = virialToDistribute * 0.5 * (1.0 -  gamma);
            
            virials[idx_O]  += distribute_O;
            virials[idx_H1] += distribute_H;
            virials[idx_H2] += distribute_H;
            virials[idx_M] = Virial(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }
        // this concludes re-distribution of the forces;
        // we assume nothing needs to be done re: virials; this sum is already tabulated at inner force loop computation

    }
}

template <class DATA>
__global__ void setMSite(int4 *waterIds, int *idToIdxs, real4 *xs, int nMolecules, BoundsGPU bounds, DATA fixRigidData) {

    int idx = GETIDX();
    if (idx < nMolecules) {
    
        /* What we do here:
         * get the minimum image positions of the O, H, H atoms
         * compute the vector position of the M site
         * apply PBC to this new position (in case the water happens to be on the boundary of the box
         */

        // first, get the ids of the atoms composing this molecule
        int idx_O  = idToIdxs[waterIds[idx].x];
        int idx_H1 = idToIdxs[waterIds[idx].y];
        int idx_H2 = idToIdxs[waterIds[idx].z];
        int idx_M  = idToIdxs[waterIds[idx].w];

        real4 pos_M_whole = xs[idx_M];
        
        // get the positions of said atoms

        real3 pos_O = make_real3(xs[idx_O]);
        real3 pos_H1= make_real3(xs[idx_H1]);
        real3 pos_H2= make_real3(xs[idx_H2]);
        real3 pos_M = make_real3(xs[idx_M]);

        // compute vectors r_ij and r_ik according to minimum image convention
        // where r_ij = r_j - r_i, r_ik = r_k - r_i,
        // and r_i, r_j, r_k are the 3-component vectors describing the positions of O, H1, H2, respectively
        real3 r_ij = bounds.minImage( (pos_H1 - pos_O));
        real3 r_ik = bounds.minImage( (pos_H2 - pos_O));

        // fixRigidData.sideLengths.w is the OM vector
        real3 r_M  = (pos_O) + fixRigidData.sideLengths.w * ( (r_ij + r_ik) / ( (length(r_ij + r_ik))));
    
        real4 pos_M_new = make_real4(r_M.x, r_M.y, r_M.z, pos_M_whole.w);
        xs[idx_M] = pos_M_new;
    }
}


// computes the center of mass for a given water molecule
template <class DATA>
__global__ void compute_COM(int4 *waterIds, real4 *xs, real4 *vs, int *idToIdxs, int nMolecules, real4 *com, BoundsGPU bounds, DATA fixRigidData) {
  int idx = GETIDX();
  if (idx  < nMolecules) {
    real3 pos[3];

    double4 mass = fixRigidData.weights;
    int ids[3];
    ids[0] = waterIds[idx].x;
    ids[1] = waterIds[idx].y;
    ids[2] = waterIds[idx].z;
    for (int i = 0; i < 3; i++) {
      int myId = ids[i];
      int myIdx = idToIdxs[myId];
      real3 p = make_real3(xs[myIdx]);
      pos[i] = p;
      }
    for (int i=1; i<3; i++) {
      real3 delta = pos[i] - pos[0];
      delta = bounds.minImage(delta);
      pos[i] = pos[0] + delta;
    }
    real ims = com[idx].w;
    com[idx] = make_real4((pos[0] * mass.x) + ( (pos[1] + pos[2]) * mass.y));
    com[idx].w = ims;
  }
}

// save the previous positions of the constraints to use as the basis for our next constraint solution
// --- note that we do not care about the M-site in this instance;
__global__ void save_prev_val(int4 *waterIds, real4 *xs, real4 *xs_0, int nMolecules, int *idToIdxs) {
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
    }
  }
}


template <class DATA, bool VIRIALS>
__global__ void settleVelocities(int4 *waterIds, real4 *xs, real4 *xs_0, 
                               real4 *vs,
                               real4 *fs, Virial *virials,  
                               real4 *comOld, 
                               DATA fixRigidData, int nMolecules, 
                               real dt, real dtf,
                               int *idToIdxs, BoundsGPU bounds, int turn) {
    int idx = GETIDX();
    if (idx < nMolecules) {
        

        // get the molecule at this idx
        int4 atomsFromMolecule = waterIds[idx];

        // get the atom idxs
        int idxO = idToIdxs[atomsFromMolecule.x];
        int idxH1= idToIdxs[atomsFromMolecule.y];
        int idxH2= idToIdxs[atomsFromMolecule.z];
        
        // extract the whole velocities
        real4 velO_whole = vs[idxO];
        real4 velH1_whole= vs[idxH1];
        real4 velH2_whole= vs[idxH2];

        // convert to double3 (drop the inv mass)
        double3 velO = make_double3(velO_whole);
        double3 velH1= make_double3(velH1_whole);
        double3 velH2= make_double3(velH2_whole);


        double3 O_corr, H1_corr, H2_corr;
        real3 velO_tmp, velH1_tmp, velH2_tmp;

        // and our positions - get the current sidelengths
        real4 posO_whole  = xs[idxO];
        real4 posH1_whole = xs[idxH1];
        real4 posH2_whole = xs[idxH2];
        
        // and cast as double
        double3 xO = make_double3(posO_whole);
        double3 xH1= make_double3(posH1_whole);
        double3 xH2= make_double3(posH2_whole);
     
        //double imO = fixRigidData.invMasses.z;
        //double imH = fixRigidData.invMasses.w;
        //double dOH = fixRigidData.sideLengths.x;
        //double dHH = fixRigidData.sideLengths.z;
        //double invdOH = fixRigidData.invSideLengths.x;
        //double invdHH = fixRigidData.invSideLengths.z;
       
        double3 rOH1 = bounds.minImage(xO-xH1);
        double3 rOH2 = bounds.minImage(xO-xH2);
        double3 rH1H2  = bounds.minImage(xH1-xH2);

        // inverse bond lengths as stipulated by the fixed geometry
        double inverseROH = fixRigidData.invSideLengths.x;
        double inverseRHH = fixRigidData.invSideLengths.z;

        // these are the vectors along which the forces are applied 
        // --- keep this in mind for the Virials!
        rOH1   *= inverseROH;
        rOH2   *= inverseROH;
        rH1H2  *= inverseRHH;
        
        // OKAY, so, everything up to here is confirmed correct..
        double3 relativeVelocity;

        // set the x, y, z components as required.  Keep orientation O<-->H consistent with the force projection 
        // matrix or you will be unhappy (and so will your water molecules!)
        double3 relVelOH1 = velO - velH1;
        double3 relVelOH2 = velO - velH2;
        double3 relVelH1H2= velH1 - velH2;
        relativeVelocity.x = dot(relVelOH1,rOH1);
        relativeVelocity.y = dot(relVelOH2,rOH2);
        relativeVelocity.z = dot(relVelH1H2,rH1H2);


        double3 vc = matrixVectorMultiply(fixRigidData.M1_inv,
                                                     fixRigidData.M2_inv,
                                                     fixRigidData.M3_inv,
                                                     relativeVelocity);
       
        // TODO velocity scale factor from NHC...??

        // velocity corrections to apply to the atoms in the molecule 
        O_corr = (rOH1 * vc.x + vc.y * rOH2) * (-1.0 * fixRigidData.invMasses.z);
        H1_corr= (rOH1 * (-1.0) * vc.x + rH1H2 * vc.z) * (-1.0 * fixRigidData.invMasses.w);
        H2_corr= (rOH2 * (-1.0) * vc.y - rH1H2 * vc.z) * (-1.0 * fixRigidData.invMasses.w);
        
        velO  += O_corr;
        velH1 += H1_corr;
        velH2 += H2_corr;

#ifdef DASH_DOUBLE
        velO_tmp = velO;
        velH1_tmp= velH1;
        velH2_tmp= velH2;
#else
        velO_tmp = make_real3(velO);
        velH1_tmp= make_real3(velH1);
        velH2_tmp= make_real3(velH2);
#endif

        vs[idxO] = make_real4(velO_tmp.x, velO_tmp.y, velO_tmp.z, velO_whole.w);
        vs[idxH1]= make_real4(velH1_tmp.x,velH1_tmp.y,velH1_tmp.z,velH1_whole.w);
        vs[idxH2]= make_real4(velH2_tmp.x,velH2_tmp.y,velH2_tmp.z,velH2_whole.w);



        /*
        if (VIRIALS) {
            // ok, do these:

            Virial virialO = Virial(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            // fcv is vc..
            // roh2, roh3, rhh are : rOH1, rOH2, rH1H2, respectively
            // and dOH, dHH are.. dOH, dHH. as above.

            // you know what.. we just aggregate these anyways, never use per-atom stresses.
            // so, just throw them all on to oxygen
            virialO[0] += dOH*rOH1.x*rOH1.x*vc.x + 
                          dOH*rOH2.x*rOH2.x*vc.y + 
                          dHH*rH1H2.x*rH1H2.x*vc.z;
            virialO[1] += dOH*rOH1.y*rOH1.y*vc.x + 
                          dOH*rOH2.y*rOH2.y*vc.y + 
                          dHH*rH1H2.y*rH1H2.y*vc.z;
            virialO[2] += dOH*rOH1.z*rOH1.z*vc.x + 
                          dOH*rOH2.z*rOH2.z*vc.y + 
                          dHH*rH1H2.z*rH1H2.z*vc.z;
            virialO[3] += dOH*rOH1.x*rOH1.y*vc.x + 
                          dOH*rOH2.x*rOH2.y*vc.y + 
                          dHH*rH1H2.x*rH1H2.y*vc.z;
            virialO[4] += dOH*rOH1.x*rOH1.z*vc.x + 
                          dOH*rOH2.x*rOH2.z*vc.y + 
                          dHH*rH1H2.x*rH1H2.z*vc.z;
            virialO[5] += dOH*rOH1.y*rOH1.z*vc.x + 
                          dOH*rOH2.y*rOH2.z*vc.y + 
                          dHH*rH1H2.y*rH1H2.z*vc.z;
            
            
            //printf("Virials idx %d: %f %f %f %f %f %f\n", idxO, virialO[0], virialO[1],virialO[2],
            //       virialO[3],virialO[4],virialO[5]);
            virials[idxO] += virialO;
            */
            /*
            rmmder[1][2] += dOH*roh2[1]*roh2[2]*fcv[0] + 
                            dOH*roh3[1]*roh3[2]*fcv[1] + 
                            dHH*rhh [1]*rhh [2]*fcv[2];

            rmmder[0][0] += dOH*roh2[0]*roh2[0]*fcv[0] + 
                            dOH*roh3[0]*roh3[0]*fcv[1] + 
                            dHH*rhh [0]*rhh [0]*fcv[2];
            rmmder[1][1] += dOH*roh2[1]*roh2[1]*fcv[0] + 
                            dOH*roh3[1]*roh3[1]*fcv[1] + 
                            dHH*rhh [1]*rhh [1]*fcv[2];
            rmmder[2][2] += dOH*roh2[2]*roh2[2]*fcv[0] + 
                            dOH*roh3[2]*roh3[2]*fcv[1] + 
                            dHH*rhh [2]*rhh [2]*fcv[2];
            rmmder[0][1] += dOH*roh2[0]*roh2[1]*fcv[0] + 
                            dOH*roh3[0]*roh3[1]*fcv[1] + 
                            dHH*rhh [0]*rhh [1]*fcv[2];
            rmmder[0][2] += dOH*roh2[0]*roh2[2]*fcv[0] + 
                            dOH*roh3[0]*roh3[2]*fcv[1] + 
                            dHH*rhh [0]*rhh [2]*fcv[2];
            for(m=0; m<DIM; m++)
            {
                for(m2=0; m2<DIM; m2++)
                {
                    rmdder[m][m2] +=
                        dOH*roh2[m]*roh2[m2]*fcv[0] +
                        dOH*roh3[m]*roh3[m2]*fcv[1] +
                        dHH*rhh [m]*rhh [m2]*fcv[2]; 
                }
            }
            */
            /*
            vals[0] = xx;
            vals[1] = yy;
            vals[2] = zz;
            vals[3] = xy;
            vals[4] = xz;
            vals[5] = yz;
            */


        //}
    }

}

// implements the SETTLE algorithm for maintaining a rigid water molecule
template <class DATA, bool VIRIALS>
__global__ void settlePositions(int4 *waterIds, real4 *xs, real4 *xs_0, 
                               real4 *vs, 
                               real4 *fs, Virial *constr_virials, real4 *comOld, 
                               DATA fixRigidData, int nMolecules, 
                               real dt, real dtf,
                               int *idToIdxs, BoundsGPU bounds, int turn, double invdt,
                               double rvscale) {
    int idx = GETIDX();
    if (idx < nMolecules) {

        // grab some data from our FixRigidData instance
        double inv2Rc = fixRigidData.canonicalTriangle.w; 
        double ra = fixRigidData.canonicalTriangle.x;
        double rb = fixRigidData.canonicalTriangle.y;
        double rc = fixRigidData.canonicalTriangle.z;
        double weightH = fixRigidData.weights.y;
        double mO = fixRigidData.weights.z;
        double mH = fixRigidData.weights.w;
        
        // so, our initial data from xs_0, accessed via idx;
        // -- this will form the basis of our X'Y'Z' system, the previous solution to the constraints
        double3 posO_initial = make_double3(xs_0[idx*3]);
        double3 posH1_initial= make_double3(xs_0[idx*3 + 1]);
        double3 posH2_initial= make_double3(xs_0[idx*3 + 2]);

        // get the molecule at this idx
        int4 atomsFromMolecule = waterIds[idx];

        // get the atom idxs
        // --- we used the variable 'idx' as an id above, in e.g. posO_initial
        //     and so we must treat it as the id here as well to access data 
        //     in a consistent manner
        int idxO = idToIdxs[atomsFromMolecule.x];
        int idxH1= idToIdxs[atomsFromMolecule.y];
        int idxH2= idToIdxs[atomsFromMolecule.z];
        
        // extract the whole positions
        real4 posO_whole = xs[idxO];
        real4 posH1_whole= xs[idxH1];
        real4 posH2_whole= xs[idxH2];

        // and the unconstrained, updated positions
        double3 posO = make_double3(posO_whole);
        double3 posH1= make_double3(posH1_whole);
        double3 posH2= make_double3(posH2_whole);
        
        // get the velocities from vs[] array
        real4 velO_whole = vs[idxO];
        real4 velH1_whole= vs[idxH1];
        real4 velH2_whole= vs[idxH2];
        
        // cast as double
        double3 velO = make_double3(velO_whole);
        double3 velH1= make_double3(velH1_whole);
        double3 velH2= make_double3(velH2_whole);
        
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
        double3 COM_d1 = (posO) + ( ( OH1_unconstrained + OH2_unconstrained ) * weightH);

        double3 posA1 = (OH1_unconstrained + OH2_unconstrained) * (-1.0 * weightH);
        double3 posB1 = posH1 - COM_d1;
        double3 posC1 = posH2 - COM_d1;

        // get our X'Y'Z' coordinate system
        double3 axis3 = cross(vectorOH1, vectorOH2);
        double3 axis1 = cross(posA1,axis3);
        double3 axis2 = cross(axis3,axis1);
       
        // normalize so that we have unit length basis vectors
        axis1 /= length(axis1);
        axis2 /= length(axis2);
        axis3 /= length(axis3);

        // rotate the relative vectors of the solved triangle
        double3 rotated_b0 = rotation(vectorOH1,axis1,axis2,axis3);
        double3 rotated_c0 = rotation(vectorOH2,axis1,axis2,axis3);
        
        // rotate our unconstrained update about the axes
        double3 rotated_a1 = rotation(posA1,axis1,axis2,axis3);
        double3 rotated_b1 = rotation(posB1,axis1,axis2,axis3);
        double3 rotated_c1 = rotation(posC1,axis1,axis2,axis3);

        double sinPhi = rotated_a1.z / ra; 
        double cosPhiSqr = 1.0 - (sinPhi * sinPhi);
        double cosPhi, cosPsiSqr, cosPsi, sinPsi;


        if (cosPhiSqr <= 0)
        {
            printf("cosPhiSqr <= 0 in settlePositions!\n");
        }
        else
        {
            cosPhi = sqrt(cosPhiSqr);
            sinPsi = (rotated_b1.z - rotated_c1.z) *  (inv2Rc / cosPhi);
            cosPsiSqr = 1.0 - (sinPsi * sinPsi);
            if (cosPsiSqr <= 0)
            {
                printf("cosPsiSqr <= 0 in settlePositions!\n");
            }
            else
            {
                cosPsi = sqrt(cosPsiSqr);
            }
        }

        double3 aPrime2 = make_double3(0.0, 
                                       ra * cosPhi,
                                       rb * sinPhi);

        double3 bPrime2 = make_double3(-1.0 * rc * cosPsi,
                                       -rb * cosPhi - rc * sinPsi * sinPhi,
                                       -rb * sinPhi + rc * sinPsi * cosPhi);

        double3 cPrime2 = make_double3(rc * cosPsi,
                                       -1.0 * rb * cosPhi + rc * sinPsi * sinPhi,
                                       -1.0 * rb * sinPhi - rc * sinPsi * cosPhi);

        
        double alpha = bPrime2.x * (rotated_b0.x - rotated_c0.x) +
                           bPrime2.y * (rotated_b0.y) + 
                           cPrime2.y * (rotated_c0.y);
    
        double beta  = bPrime2.x * (rotated_c0.y - rotated_b0.y) + 
                           bPrime2.y * (rotated_b0.x) + 
                           cPrime2.y * (rotated_c0.x);

        double gamma = (rotated_b0.x * rotated_b1.y) - 
                           (rotated_b1.x * rotated_b0.y) +
                           (rotated_c0.x * rotated_c1.y) - 
                           (rotated_c1.x * rotated_c0.y);

        double a2b2 = (alpha * alpha) + (beta * beta);
        double sinTheta = (alpha * gamma - (beta * sqrt(a2b2 - (gamma * gamma)))) / a2b2;

        double cosThetaSqr = 1.0 - (sinTheta * sinTheta);
        double cosTheta = sqrt(cosThetaSqr);

        double3 aPrime3 = make_double3(-aPrime2.y * sinTheta,
                                        aPrime2.y * cosTheta,
                                        rotated_a1.z);
    
        double3 bPrime3 = make_double3(bPrime2.x * cosTheta - bPrime2.y * sinTheta,
                                       bPrime2.x * sinTheta + bPrime2.y * cosTheta,
                                       bPrime2.z);

        double3 cPrime3 = make_double3(cPrime2.x * cosTheta - cPrime2.y * sinTheta,
                                       cPrime2.x * sinTheta + cPrime2.y * cosTheta,
                                       cPrime2.z);

        // get the transpose (inverse) of our rotation matrix from above
        double3 axis1T = make_double3(axis1.x, axis2.x, axis3.x);
        double3 axis2T = make_double3(axis1.y, axis2.y, axis3.y);
        double3 axis3T = make_double3(axis1.z, axis2.z, axis3.z);

        // rotate the solutions back to the original coordinate system
        double3 a_pos = rotation(aPrime3, axis1T, axis2T, axis3T);
        double3 b_pos = rotation(bPrime3, axis1T, axis2T, axis3T);
        double3 c_pos = rotation(cPrime3, axis1T, axis2T, axis3T);

        // add back COM to get final position of the atoms
        double3 oPosFinal = a_pos + COM_d1;
        double3 H1PosFinal= b_pos + COM_d1;
        double3 H2PosFinal= c_pos + COM_d1;

        // differential contributions to the velocities
        double3 dx_a = a_pos - posA1;
        double3 dx_b = b_pos - posB1;
        double3 dx_c = c_pos - posC1;

#ifdef DASH_DOUBLE
        // if DASH_DOUBLE, then just set equal - do not call make_real3
        real3 oPosNew = oPosFinal;
        real3 H1PosNew= H1PosFinal;
        real3 H2PosNew= H2PosFinal;
#else
        // else, we still did the intermediate calculations in double, and must now send to real
        // -- as an aside, this is only be done so that the code compiles.  If you're simulating rigid water, you should 
        //    be using double precision
        real3 oPosNew = make_real3(oPosFinal);
        real3 H1PosNew= make_real3(H1PosFinal);
        real3 H2PosNew= make_real3(H2PosFinal);
#endif
        // set the positions in the global arrays as the new, solved positions
        xs[idxO] = make_real4(oPosNew.x, oPosNew.y, oPosNew.z,posO_whole.w);
        xs[idxH1]= make_real4(H1PosNew.x,H1PosNew.y,H1PosNew.z,posH1_whole.w);
        xs[idxH2]= make_real4(H2PosNew.x,H2PosNew.y,H2PosNew.z,posH2_whole.w);

        // add the differential contributions to the velocity from settling the positions
        velO  += (dx_a * invdt);
        velH1 += (dx_b * invdt);
        velH2 += (dx_c * invdt);

        // cast as real
#ifdef DASH_DOUBLE
        // just set equal
        real3 newVelO = velO;
        real3 newVelH1= velH1;
        real3 newVelH2= velH2;
#else 
        real3 newVelO = make_real3(velO);
        real3 newVelH1= make_real3(velH1);
        real3 newVelH2= make_real3(velH2);
#endif
        // set the velocities in global arrays as the new, solved velocities
        vs[idxO] = make_real4(newVelO,velO_whole.w);
        vs[idxH1]= make_real4(newVelH1,velH1_whole.w);
        vs[idxH2]= make_real4(newVelH2,velH2_whole.w);

        if (VIRIALS) {

            // ok, dax, day, daz --> dx_a
            //     dbx, dby, dbz --> dx_b
            //     dcx, dcy, dcz --> dx_c
            // what about mOs? -->  mass oxygen, with scale factor from Nose thermostat.
            // --- note that we do not use MTK barostat with this; don't worry about it.
            double mOs = mO / rvscale;
            double mHs = mH / rvscale;
            double dt2 = invdt * invdt;
            double3 mda_O = mOs * dx_a / dt2;
            double3 mda_H1= mHs * dx_b / dt2;
            double3 mda_H2= mHs * dx_c / dt2;
            // ok; so, for one, they divide by dt^2 here, so we should do that as well
            
            Virial virialO  = Virial(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            Virial virialH1 = Virial(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            Virial virialH2 = Virial(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

            // displace xs_0 relative to oPosFinal using PBC
            double3 init_O = oPosFinal + bounds.minImage(posO_initial - oPosFinal);
            double3 init_H1= init_O + bounds.minImage(posH1_initial - init_O);
            double3 init_H2= init_O + bounds.minImage(posH2_initial - init_O);
            
            computeVirial_double(virialO ,init_O,-mda_O);
            computeVirial_double(virialH1,init_H1,-mda_H1);
            computeVirial_double(virialH2,init_H2,-mda_H2);
            
            constr_virials[idxO]  += virialO;
            constr_virials[idxH1] += virialH1;
            constr_virials[idxH2] += virialH2;

          /*
              mdax = mOs*dax;
              mday = mOs*day;
              mdaz = mOs*daz;
              mdbx = mHs*dbx;
              mdby = mHs*dby;
              mdbz = mHs*dbz;
              mdcx = mHs*dcx;
              mdcy = mHs*dcy;
              mdcz = mHs*dcz;
          */

            // the ones we actually compute
            /*
            rmdr[XX][XX] -= b4[ow1]*mdax + b4[hw2]*mdbx + b4[hw3]*mdcx;
            rmdr[YY][YY] -= b4[ow1+1]*mday + b4[hw2+1]*mdby + b4[hw3+1]*mdcy;
            rmdr[ZZ][ZZ] -= b4[ow1+2]*mdaz + b4[hw2+2]*mdbz + b4[hw3+2]*mdcz;
            rmdr[XX][YY] -= b4[ow1]*mday + b4[hw2]*mdby + b4[hw3]*mdcy;
            rmdr[XX][ZZ] -= b4[ow1]*mdaz + b4[hw2]*mdbz + b4[hw3]*mdcz;
            rmdr[YY][ZZ] -= b4[ow1+1]*mdaz + b4[hw2+1]*mdbz + b4[hw3+1]*mdcz;
            */
            
            /*
            rmdr[YY][XX] -= b4[ow1+1]*mdax + b4[hw2+1]*mdbx + b4[hw3+1]*mdcx;
            rmdr[ZZ][XX] -= b4[ow1+2]*mdax + b4[hw2+2]*mdbx + b4[hw3+2]*mdcx;
            rmdr[ZZ][YY] -= b4[ow1+2]*mday + b4[hw2+2]*mdby + b4[hw3+2]*mdcy;
            */
        }
    }
}

} // namespace

// 
void FixRigid::populateRigidData() {
    
    // some trigonometry here
    double cosC = fixRigidData.canonicalTriangle.z / fixRigidData.sideLengths.y;
    
    // isosceles triangles
    double cosB = cosC;
   
    // a is the HH bond length, b is OH bond length, c is OH bond length (b == c)
    double a = fixRigidData.sideLengths.z;
    double b = fixRigidData.sideLengths.x;

    // b == c..
    double cosA = ((-1.0 * a * a ) + (b*b + b*b) ) / (2.0 * b * b);

    // sum of the angles should be pi radians - if not, exit the simulation.
    double sumAngles = acos(cosA) + acos(cosB) + acos(cosC);
    if ( fabs(sumAngles - M_PI) > 0.000001) {
        printf("The sum of the angles for the triangle in SETTLE was found to be %3.10f radians, rather than pi radians. Aborting.\n",sumAngles);
        printf("cosA: %3.6f\ncosB: %3.6f\ncosC: %3.6f\nwith theta values of %f, %f %f\n", 
               cosA, cosB, cosC, acos(cosA), acos(cosB), acos(cosC));
        mdError("Sum of the interior angles of the triangle do not add up to pi radians in FixRigid.\n");
    }

    double invMH_normalized = fixRigidData.invMasses.w / fixRigidData.invMasses.z;

    double3 M1_tmp = make_double3(0.0, 0.0, 0.0);

    double HH = fixRigidData.sideLengths.z;
    //std::cout << "Using HH length of " << HH << std::endl;
    //printf("dHH:  %18.14f\n",HH);
    double OH = fixRigidData.sideLengths.x;
    //std::cout << "Using OH length of " << OH << std::endl;
    // [0,0]
    M1_tmp.x = 1.0 + ( invMH_normalized );
    // [0,1]
    M1_tmp.y = ( 1.0 - 0.5 * HH * HH  / (OH * OH) ) ;
    // [0,2]
    M1_tmp.z = invMH_normalized * 0.5 * HH / OH;

    double3 M2_tmp = make_double3(0.0, 0.0, 0.0);

    // [1,0]
    M2_tmp.x = M1_tmp.y;
    // [1,1]
    M2_tmp.y = M1_tmp.x;
    // [1,2]
    M2_tmp.z = M1_tmp.z;


    double3 M3_tmp = make_double3(0.0, 0.0, 0.0);
    // [2,0]
    M3_tmp.x = M1_tmp.z;
    // [2,1]
    M3_tmp.y = M2_tmp.z;
    // [2,2]
    M3_tmp.z = 2.0 * invMH_normalized;

    fixRigidData.M1 = M1_tmp;
    fixRigidData.M2 = M2_tmp;
    fixRigidData.M3 = M3_tmp;

    double3 M1_inv, M2_inv, M3_inv;

    // computes the inverse matrix of {M1;M2;M3} and stores in M1_inv, M2_inv, M3_inv;
    invertMatrix(M1_tmp,M2_tmp,M3_tmp,M1_inv,M2_inv,M3_inv);

    M1_inv *= fixRigidData.weights.z;
    M2_inv *= fixRigidData.weights.z;
    M3_inv *= fixRigidData.weights.z;

    fixRigidData.M1_inv = M1_inv;
    fixRigidData.M2_inv = M2_inv;
    fixRigidData.M3_inv = M3_inv;

    return;

}

void FixRigid::singlePointEng_massless(real *perParticleEng) {


    // we want a kernel that 
    // (1) reads the PPE assigned to the m-site, if there is m sites present..
    // (2) adds said PPE to the oxygen
    // (3) sets PPE of m-site to zero
    if (FOURSITE) {
        distributeMSite_Energy<<<NBLOCK(nMolecules), PERBLOCK>>>(nMolecules,
                                                                 waterIdsGPU.data(),
                                                                 state->gpd.idToIdxs.d_data.data(),
                                                                 perParticleEng);
    }

    // if 3-site model, do nothing.
    return;
}

bool FixRigid::postNVE_X() {

    real dt = state->dt;
    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();
    BoundsGPU &bounds = state->boundsGPU;
    int nAtoms = state->atoms.size();
    // first, unconstrained velocity update continues: distribute the force from the M-site
    //        and integrate the velocities accordingly.  Update the forces as well.
    // Next,  do compute_SETTLE as usual on the (as-yet) unconstrained positions & velocities

    // from IntegratorVerlet
    real dtf = 0.5f * state->dt * state->units.ftm_to_v;

    double invdt = (double) 1.0 /  (double(state->dt));
    
    double rvscale = 1.0; // TODO
    int virialMode = state->dataManager.getVirialModeForTurn(state->turn);
    bool virials = ((virialMode == 1) or (virialMode == 2));

    if (virials) {
        settlePositions<FixRigidData,true><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                                    xs_0.data(), gpd.vs(activeIdx), 
                                                    gpd.fs(activeIdx), 
                                                    gpd.virials.d_data.data(),
                                                    com.data(), fixRigidData, nMolecules, dt, dtf, 
                                                    gpd.idToIdxs.d_data.data(), bounds, (int) state->turn, invdt,rvscale);

    } else {
        settlePositions<FixRigidData,false><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                                    xs_0.data(), gpd.vs(activeIdx), 
                                                    gpd.fs(activeIdx), 
                                                    gpd.virials.d_data.data(),
                                                    com.data(), fixRigidData, nMolecules, dt, dtf, 
                                                    gpd.idToIdxs.d_data.data(), bounds, (int) state->turn, invdt, rvscale);

    }

    if (FOURSITE) {
    
        setMSite<FixRigidData><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.idToIdxs.d_data.data(), gpd.xs(activeIdx), nMolecules, bounds, fixRigidData);
    
    }

    return true;
}




void FixRigid::set_fixed_sides() {
   
    // we already have the bond lengths from 'setStyleBondLengths'...

    // we have not yet set the 'weights' attribute for our fixRigidData instance.
    // do so now.

    mdAssert(waterIds.size() > 0,"There needs to be at least one water in FixRigid for this fix to be activated!");
    int4 firstMolecule = waterIds[0];
    int id_O1 = firstMolecule.x;
    int id_H1 = firstMolecule.y;
    //double massO = 15.999400000;
    double massO = state->atoms[id_O1].mass;
    //std::cout << "massO found to be " << massO << std::endl;
    //double massH = 1.00800000;
    double massH = state->atoms[id_H1].mass;
    //std::cout << "massH1 found to be " << massH << std::endl;
    double massWater = massO + (2.0 * massH);

    //std::cout << "massWater found to be " << massWater << std::endl;


    fixRigidData.weights.x = massO / massWater;
    fixRigidData.weights.y = massH / massWater;
    fixRigidData.weights.z = massO;
    fixRigidData.weights.w = massH;

    double4 invWeights = make_double4( 1.0 / (massO / massWater),
                                       1.0 / (massH / massWater),
                                       1.0 / massO,
                                       1.0 / massH);

    fixRigidData.invMasses = invWeights;

    //real4 fixRigidData.sideLengths is as OH, OH, HH [, OM]...
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

    //std::cout << "ra: " << ra << "; rb: " << rb << "; rc: " << rc << std::endl;


    double raVal = 2.0 * massH * sqrt(OH * OH - (0.5 * fixRigidData.sideLengths.z * 0.5 * fixRigidData.sideLengths.z) ) / (massWater);
    double rbVal = sqrt(OH * OH - (0.5 * fixRigidData.sideLengths.z * 0.5 * fixRigidData.sideLengths.z)) - raVal;
    double inv2Rc = 1.0 / fixRigidData.sideLengths.z;

    //std::cout << "raVal: " << raVal << "; rbVal: " << rbVal << "; inv2Rc: " << inv2Rc << std::endl;
    fixRigidData.canonicalTriangle = make_double4(ra,rb,rc,inv2Rc);

}




void FixRigid::setStyleBondLengths() {
    
    if (FOURSITE) {
        // 4-site models here

        if (style == "TIP4P/2005") {
            std::cout << "SETTLE algorithm is maintaining a TIP4P/2005 geometry!" << std::endl;
            r_OH = 0.95720000000;
            r_HH = 1.51390000000;
            r_OM = 0.15460000000;
            gamma = 0.73612409;
            // see q-TIP4P/f paper by Manolopoulos et. al., eq. 2; this quantity can be computed from an arbitrary
            // TIP4P/2005 molecule fairly easily and is consistent with the above r_OH, r_HH, r_OM values
            gamma = 0.73612446364836; 
        } else if (style == "TIP4P") {
            std::cout << "SETTLE algorithm is maintaining a TIP4P geometry!" << std::endl;
            r_OH  = 0.9572000000;
            r_HH  = 1.51390000000;
            r_OM  = 0.15000000;
            gamma = 0.74397533;
        } else {
            // I don't think that this would ever get triggered, since we check in the constructor that a valid style argument was selected..
            mdError("Only TIP4P and TIP4P/2005 geometries supported for four-site models!");
        }

    } else if (THREESITE) {
        // 3-site models here
        
        // set r_OM to 0.0 for completeness
        r_OM  = 0.0;
        gamma = 0.0;

        if (style == "TIP3P") {
            std::cout << "SETTLE algorithm is maintaining a TIP3P geometry!" << std::endl;
            r_OH = 0.95720000000;
            r_HH = 1.51390000000;
        } else if (style == "SPC") {
            std::cout << "SETTLE algorithm is maintaining a SPC geometry!" << std::endl;
            r_OH = 1.00000000000;
            r_HH = 1.63300000000;
        } else if (style == "SPC/E") {
            std::cout << "SETTLE algorithm is maintaining a SPC/E geometry!" << std::endl;
            r_OH = 1.00000000000;
            r_HH = 1.63300000000;
        } else {
            mdError("Only TIP3P, SPC, and SPC/E geometries supported for three-site models!");
        }

    } else {

        mdError("Neither 3-site nor 4-site model currently selected in FixRigid.\n");

    }

    if (state->units.unitType == UNITS::LJ) {
        mdError("Currently, real units must be used for simulations of rigid water models.  \nSet state.units.setReal() in your simulation script and adjust all units accordingly.\n");
    }

    fixedSides = make_double4(r_OH, r_OH, r_HH, r_OM);

    // set the sideLengths variable in fixRigidData to the fixedSides values
    fixRigidData.sideLengths = fixedSides;
    fixRigidData.gamma = gamma;
    double invOM = 0.0;
    if (FOURSITE) invOM = 1.0 / r_OM;

    double4 invSideLengths = make_double4(1.0/r_OH, 1.0 / r_OH, 1.0 / r_HH, invOM);
    fixRigidData.invSideLengths = invSideLengths;
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
    if (FOURSITE) {
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
                // set our class member boolean flags THREESITE & FOURSITE
                FOURSITE = fourSite;
                THREESITE = !fourSite;
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
        if (FOURSITE) {
            Bond bondOM;
            bondOM.ids = { {waterIds[i].x, waterIds[i].w } };
            bonds.push_back(bondOM);
        }
    }

    //std::cout << "There are " << waterIds.size() << " molecules read in from the restart file and " << bonds.size() << " bonds were made in FixRigid.\n";
    return true;
}

void FixRigid::assignNDF() {
   
    // if this is THREESITE, assign NDF of 2 to each atom in a given molecule (distribute the reduction equally)
    if (THREESITE) {
        for (int i = 0; i < waterIds.size(); i++) {
            int4 thisMolIds = waterIds[i];
            int idO = thisMolIds.x;
            int idH1= thisMolIds.y;
            int idH2= thisMolIds.z;
            Atom &O = state->atoms[state->idToIdx[idO]];
            Atom &H1= state->atoms[state->idToIdx[idH1]];
            Atom &H2= state->atoms[state->idToIdx[idH2]];
            O.setNDF(2);
            H1.setNDF(2);
            H2.setNDF(2);
        }
    }
    // if this is 4 site model, assign NDF of 2 to each real atom, and 0 to the M-site
    if (FOURSITE) {
        
        for (int i = 0; i < waterIds.size(); i++) {
            int4 thisMolIds = waterIds[i];
            int idO = thisMolIds.x;
            int idH1= thisMolIds.y;
            int idH2= thisMolIds.z;
            int idM = thisMolIds.w;
            Atom &O = state->atoms[state->idToIdx[idO]];
            Atom &H1= state->atoms[state->idToIdx[idH1]];
            Atom &H2= state->atoms[state->idToIdx[idH2]];
            Atom &Msite = state->atoms[state->idToIdx[idM]];
            O.setNDF(2);
            H1.setNDF(2);
            H2.setNDF(2);
            Msite.setNDF(0);
        }
    }
}

//returns ids of atoms that belong to a constrained group
std::vector<int> FixRigid::getRigidAtoms() {

    std::vector<int> atomsToReturn;
    // by now, this is prepared. so, we know if it is a 3-site or 4-site model.
    if (THREESITE) {
        atomsToReturn.reserve(3 * nMolecules);
    } else {
        atomsToReturn.reserve(4 * nMolecules);
    }

    if (THREESITE) {
        for (int i = 0; i < waterIds.size(); i++ ) {
            int4 theseAtomIds = waterIds[i];
            int idO = theseAtomIds.x;
            int idH1= theseAtomIds.y;
            int idH2= theseAtomIds.z;
            atomsToReturn.push_back(idO);
            atomsToReturn.push_back(idH1);
            atomsToReturn.push_back(idH2);
        }
    } else {
        for (int i = 0; i < waterIds.size(); i++ ) {
            int4 theseAtomIds = waterIds[i];
            int idO = theseAtomIds.x;
            int idH1= theseAtomIds.y;
            int idH2= theseAtomIds.z;
            int idM = theseAtomIds.w;
            atomsToReturn.push_back(idO);
            atomsToReturn.push_back(idH1);
            atomsToReturn.push_back(idH2);
            atomsToReturn.push_back(idM);
        }
    }

    return atomsToReturn;
}

void FixRigid::scaleRigidBodies(real3 scaleBy, uint32_t groupTag) {

    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();
    BoundsGPU &bounds = state->boundsGPU;
    int nAtoms = state->atoms.size();
    if (groupTag == 1) {
            rigid_scaleSystem_cu<FixRigidData, false><<<NBLOCK(nMolecules),PERBLOCK>>>(waterIdsGPU.data(),
                                                                  gpd.xs(activeIdx),
                                                                  gpd.idToIdxs.d_data.data(),
                                                                  bounds.lo,
                                                                  bounds.rectComponents,
                                                                  bounds, 
                                                                  scaleBy,
                                                                  fixRigidData,
                                                                  nMolecules);

    } else {
            rigid_scaleSystemGroup_cu<FixRigidData,false><<<NBLOCK(nMolecules),PERBLOCK>>>(waterIdsGPU.data(),
                                                                  gpd.xs(activeIdx),
                                                                  gpd.idToIdxs.d_data.data(),
                                                                  bounds.lo,
                                                                  bounds.rectComponents,
                                                                  bounds, 
                                                                  scaleBy,
                                                                  fixRigidData,
                                                                  nMolecules, 
                                                                  groupTag,
                                                                  gpd.fs(activeIdx)
                                                                  );
    }


    if (FOURSITE) {
            setMSite<FixRigidData><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.idToIdxs.d_data.data(), gpd.xs(activeIdx), nMolecules, bounds,fixRigidData);
    }

}

void FixRigid::createRigid(int id_a, int id_b, int id_c, int id_d) {
    FOURSITE = true;
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
    real4 ims4 = make_real4(0.0f, 0.0f, 0.0f, real(ims));
    invMassSums.push_back(ims4);

    bool ordered = true;
    if (! (ma > mb && ma > mc)) ordered = false;
    
    if (! (mb == mc) ) ordered = false;
    if (! (mb > md) )  ordered = false;
    
    if (! (ordered)) {
        printf("Found masses O, H, H, M in order: %f %f %f %f\n", ma, mb, mc, md);
    }
    if (! (ordered)) mdError("Ids in FixRigid::createRigid must be as O, H1, H2, M");

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

    // finally, if the state->rigidBodies flag is not yet tripped, set it to true
    state->rigidBodies = true;

}

void FixRigid::createRigid(int id_a, int id_b, int id_c) {
    THREESITE = true;
    int4 waterMol = make_int4(0,0,0,0);
    Vector a = state->idToAtom(id_a).pos;
    Vector b = state->idToAtom(id_b).pos;
    Vector c = state->idToAtom(id_c).pos;

    double ma = state->idToAtom(id_a).mass;
    double mb = state->idToAtom(id_b).mass;
    double mc = state->idToAtom(id_c).mass;
    double ims = 1.0 / (ma + mb + mc);
    real4 ims4 = make_real4(0.0f, 0.0f, 0.0f, real(ims));
    invMassSums.push_back(ims4);

    // this discovers the order of the id's that was passed in, i.e. OHH, HOH, HHO, etc.
    real det = a[0]*b[1]*c[2] - a[0]*c[1]*b[2] - b[0]*a[1]*c[2] + b[0]*c[1]*a[2] + c[0]*a[1]*b[2] - c[0]*b[1]*a[2];
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
    
    state->rigidBodies = true;
}

bool FixRigid::postNVE_V() {


    return true;
}

bool FixRigid::prepareForRun() {
    
    nMolecules = waterIds.size();
    // consider: in postRun(), we set rigidBodies to false;
    // if another run command is issued without adding another water molecule, it will remain false, 
    // and that will cause problems.  So, set it to true here as well.  Is this too late though? 
    // ---- I think it is not too late to trip this flag here.
    if (nMolecules > 0) {
        state->rigidBodies = true;
    }
    // cannot have more than one water model present
    if (THREESITE && FOURSITE) {
        mdError("An attempt was made to use both 3-site and 4-site models in a simulation");
    }

    if (!(THREESITE || FOURSITE)) {
        mdError("An attempt was made to use neither 3-site nor 4-site water models with FixRigid in simulation.");
    }

    if (FOURSITE) state->masslessSites = true;

    // an instance of FixRigidData
    fixRigidData = FixRigidData();
    setStyleBondLengths();
    set_fixed_sides();
    populateRigidData();

    //printf("number of molecules in waterIds: %d\n", nMolecules);
    waterIdsGPU = GPUArrayDeviceGlobal<int4>(nMolecules);
    waterIdsGPU.set(waterIds.data());
    
    virials_local = GPUArrayDeviceGlobal<Virial>(state->gpd.virials.size());
    virials_local.memset(0);
    gpuBuffer = GPUArrayGlobal<real>(state->atoms.size() * 6);

    xs_0 = GPUArrayDeviceGlobal<real4>(3*nMolecules);
    com = GPUArrayDeviceGlobal<real4>(nMolecules);
    constraints = GPUArrayDeviceGlobal<bool>(nMolecules);
    com.set(invMassSums.data());

    real dt = state->dt;
    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();
    BoundsGPU &bounds = state->boundsGPU;
    int nAtoms = state->atoms.size();
    real dtf = 0.5f * state->dt * state->units.ftm_to_v;
    
    int virialMode = state->dataManager.getVirialModeForTurn(state->turn);
    bool virials = ((virialMode == 1) or (virialMode == 2));
    
    compute_COM<FixRigidData><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), 
                                                                gpd.xs(activeIdx), 
                                                                gpd.vs(activeIdx), 
                                                                gpd.idToIdxs.d_data.data(), 
                                                                nMolecules, com.data(), 
                                                                bounds, fixRigidData);

    
    save_prev_val<<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), xs_0.data(), 
                                                nMolecules, gpd.idToIdxs.d_data.data());

    double invdt = (double) 1.0 /  (double(state->dt));

    // iterate over all atoms and reduce their configurational DOF accordingly as the atom struct
    assignNDF();
    // so; solve the positions (if we have good initial conditions, nothing happens here)

    // get our pointer to the NoseHoover thermostat, if it exists; else, nullptr


    double rvscale = 1.0; // TODO
    // this prepare is called before NoseHoover's prepare is called; so, 
    // we'll need to prepare NoseHoover
    if (solveInitialConstraints) {
        if (virials) {
            settlePositions<FixRigidData,true><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                                        xs_0.data(), gpd.vs(activeIdx),  
                                                        gpd.fs(activeIdx), 
                                                        gpd.virials.d_data.data(),
                                                        com.data(), fixRigidData, nMolecules, dt, dtf, 
                                                        gpd.idToIdxs.d_data.data(), bounds, (int) state->turn, invdt,rvscale);
        } else {

            settlePositions<FixRigidData,false><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                                        xs_0.data(), gpd.vs(activeIdx), 
                                                        gpd.fs(activeIdx), 
                                                        gpd.virials.d_data.data(),
                                                        com.data(), fixRigidData, nMolecules, dt, dtf, 
                                                        gpd.idToIdxs.d_data.data(), bounds, (int) state->turn, invdt,rvscale);
        }

        if (FOURSITE) {
            // if we need the virials initially, distribute them; do not distribute forces.
            if (virials) {
                distributeMSite<true,false><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), 
                                                                              gpd.xs(activeIdx), 
                                                             gpd.vs(activeIdx),  gpd.fs(activeIdx),
                                                             gpd.virials.d_data.data(),
                                                             nMolecules, gamma, dtf, gpd.idToIdxs.d_data.data(), bounds);
                
            }  
            // just to be redundant - set the M-site at its assigned position
            setMSite<FixRigidData><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.idToIdxs.d_data.data(), gpd.xs(activeIdx), nMolecules, bounds,fixRigidData);
        }
        settleVelocities<FixRigidData,false><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                                    xs_0.data(), gpd.vs(activeIdx), 
                                                    gpd.fs(activeIdx), 
                                                    gpd.virials.d_data.data(),
                                                    com.data(), fixRigidData, nMolecules, dt, dtf, 
                                                    gpd.idToIdxs.d_data.data(), bounds, (int) state->turn);
        // and remove the COMV we may have acquired from settling the velocity constraints just now.
        GPUArrayDeviceGlobal<real4> sumMomentum = GPUArrayDeviceGlobal<real4>(2);

        // should probably make removal of COMV optional
        CUT_CHECK_ERROR("Error occurred during solution of velocity constraints.");
        sumMomentum.memset(0);
        int warpSize = state->devManager.prop.warpSize;

        real3 dimsreal3 = make_real3(1.0, 1.0, 1.0);
        accumulate_gpu<real4, real4, SumVectorXYZOverW, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(real4)>>>
                (
                 sumMomentum.data(),
                 gpd.vs(activeIdx),
                 nAtoms,
                 warpSize,
                 SumVectorXYZOverW()
                );
        rigid_remove_COMV<<<NBLOCK(nAtoms), PERBLOCK>>>(nAtoms, gpd.vs(activeIdx), sumMomentum.data(), dimsreal3);
        CUT_CHECK_ERROR("Removal of COMV in FixRigid failed.");
        
        // validate that we have good initial conditions
        SAFECALL((validateConstraints<FixRigidData> <<<NBLOCK(nMolecules), PERBLOCK>>> (waterIdsGPU.data(), gpd.idToIdxs.d_data.data(), 
                                                               gpd.xs(activeIdx), gpd.vs(activeIdx), 
                                                               nMolecules, bounds, fixRigidData, 
                                                               constraints.data(), state->turn)));
        CUT_CHECK_ERROR("Validation of constraints failed in FixRigid.");
    }

    prepared = true;
    return prepared;
}

bool FixRigid::stepInit() {
    
    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();
    
    // save the positions, velocities, forces from the previous, fully updated turn in to our local arrays
    // ---- we really only need the positions here
    save_prev_val<<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), xs_0.data(), 
                                                nMolecules, gpd.idToIdxs.d_data.data());


    return true;
}

void FixRigid::updateScaleVariables() {
    // if we have an instance of FixNoseHoover, get the scale variable values;
    // else, assign s.t. nothing material happens
    bool noseHoover = false; // ok, TODO, this line is currently nonsense - tbd for npt water
    if (noseHoover) {
        // ok, so these should have values by now..
        /*
        alpha  = noseHoover->alpha;
        vscale = noseHoover->vscale;
        rscale = noseHoover->rscale;
        rvscale = vscale * rscale;
        */
    } else {
        alpha  = 1.0;
        vscale = 1.0;
        rscale = 1.0;
        rvscale = vscale * rscale;
    }
}

bool FixRigid::preStepFinal() {
    
    real dt = state->dt;
    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();
    BoundsGPU &bounds = state->boundsGPU;
    // first, unconstrained velocity update continues: distribute the force from the M-site
    //        and integrate the velocities accordingly.  Update the forces as well.
    // finally, solve the velocity constraints on the velocities.
    // from IntegratorVerlet
    real dtf = 0.5f * state->dt * state->units.ftm_to_v;
    
    int virialMode = state->dataManager.getVirialModeForTurn(state->turn);
    bool virials = ((virialMode == 1) or (virialMode == 2));


    if (FOURSITE) {
        if (virials) {
            distributeMSite<true,true><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                                         gpd.vs(activeIdx),  gpd.fs(activeIdx),
                                                         gpd.virials.d_data.data(),
                                                         nMolecules, gamma, dtf, gpd.idToIdxs.d_data.data(), bounds);
            
        } else { 
        
            distributeMSite<false,true><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                                         gpd.vs(activeIdx),  gpd.fs(activeIdx),
                                                         gpd.virials.d_data.data(),
                                                         nMolecules, gamma, dtf, gpd.idToIdxs.d_data.data(), bounds);
        }
    }

    if (virials) {
        settleVelocities<FixRigidData,true><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                                    xs_0.data(), gpd.vs(activeIdx),
                                                    gpd.fs(activeIdx), 
                                                    gpd.virials.d_data.data(), 
                                                    com.data(), fixRigidData, nMolecules, dt, dtf, 
                                                    gpd.idToIdxs.d_data.data(), bounds, (int) state->turn);
    } else {

        settleVelocities<FixRigidData,false><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                                    xs_0.data(), gpd.vs(activeIdx),
                                                    gpd.fs(activeIdx),
                                                    gpd.virials.d_data.data(),
                                                    com.data(), fixRigidData, nMolecules, dt, dtf, 
                                                    gpd.idToIdxs.d_data.data(), bounds, (int) state->turn);

    }
    // set the MSite; note that this is not required for proper integration/dynamics; rather,
    // we do this extraneous call so that, in the even the configuration is printed out, the Msite is in the 
    // correct position.
    
    if (FOURSITE) {
        setMSite<FixRigidData><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.idToIdxs.d_data.data(), gpd.xs(activeIdx), nMolecules, bounds,fixRigidData);
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


Virial FixRigid::velocity_virials(double alpha, double veta, bool returnFromStep) {

    // this is called after 
    double dt = (double) state->dt;
    double g = 0.5*veta*dt;
    rscale = std::exp(g)*series_sinhx(g);
    g = -0.25 * alpha * veta * dt;
    vscale = std::exp(g)*series_sinhx(g);
    rvscale = vscale*rscale;
    
    // reset virials_local, which we are about to write to (since idxs switch around, can't assume all old data would be overwritten)
    virials_local.memset(0);
    gpuBuffer.d_data.memset(0); 

    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();
    BoundsGPU &bounds = state->boundsGPU;
    int nAtoms = state->atoms.size();
    real dtf = 0.5 * state->dt * state->units.ftm_to_v;

    // here, do the thing. get the shake virials, then reduce and return an aggregated quantity.
    // do not adjust for units - that will be handled elsewhere

    // -- note that we do not want an accumulation class as implemented..
    settleVelocities<FixRigidData,true><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), 
                                                gpd.xs(activeIdx), 
                                                xs_0.data(), gpd.vs(activeIdx),
                                                gpd.fs(activeIdx), 
                                                virials_local.data(), 
                                                com.data(), fixRigidData, 
                                                nMolecules,(real) dt, dtf, 
                                                gpd.idToIdxs.d_data.data(), 
                                                bounds, (int) state->turn);
    
    // ok, we've written to virials_local.data...
    /*
    accumulate_gpu<Virial, Virial, SumVirial, N_DATA_PER_THREAD>  <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(Virial)>>>
    (
         (Virial *) gpuBuffer.getDevData(), virials_local.data(), 
          nAtoms, state->devManager.prop.warpSize, SumVirial()
    );    
     
    gpuBuffer.dataToHost();

    cudaDeviceSynchronize();

    sumVirial = * (Virial *) gpuBuffer.h_data.data();
    */
    sumVirial = Virial(0.0, 0.0, 0.0, 0.0, 0.0,0.0);
    return sumVirial;
}

bool FixRigid::stepFinal() {

    return true;

}


// postRun is primarily for re-setting local and global flags;
// in this case, tell state that there are no longer rigid bodies
bool FixRigid::postRun() {
    prepared = false;
    solveInitialConstraints = false;
    state->rigidBodies = false;
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
	    (py::args("state", "handle", "style")
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
	.def_readwrite("printing", &FixRigid::printing)
    .def_readwrite("solveInitialConstraints", &FixRigid::solveInitialConstraints)
    ;
}





