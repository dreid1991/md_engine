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
}

__device__ inline float3 positionsToCOM(float3 *pos, float *mass, float ims) {
  return (pos[0]*mass[0] + pos[1]*mass[1] + pos[2]*mass[2])*ims;
}

inline __host__ __device__ float3 rotateCoords(float3 vector, float3 matrix[]) {
  return make_float3(dot(matrix[0],vector),dot(matrix[1],vector),dot(matrix[2],vector));
}



// this function removes any residual velocities along the lengths of the bonds, 
// thus permitting the use of our initializeTemperature() module without violation of the constraints
// on initializing a simulation
__global__ void adjustInitialVelocities(int4* waterIds, int *idToIdxs, float4 *xs, float4 *vs, int nMolecules, BoundsGPU bounds) {

    int idx = GETIDX();

    if (idx < nMolecules) {
        // get the atom ids
        int id_O = waterIds[idx].x;
        int id_H1 = waterIds[idx].y;
        int id_H2 = waterIds[idx].z;

        
    }

}

// called at the end of stepFinal, this should print out values expected for the constrained water molecules
// i.e., bond lengths should be fixed, and the dot product of the relative velocities along a bond with the 
// bond vector should be identically zero.
__global__ void validateConstraints(int4* waterIds, int *idToIdxs, float4 *xs, float4 *vs, int nMolecules, BoundsGPU bounds) {

    int idx = GETIDX();

    // just 'tag' the first 5 molecules
    if (idx < 5) {

        // extract the ids
        int id_O =  waterIds[idx].x;
        int id_H1 = waterIds[idx].y;
        int id_H2 = waterIds[idx].z;

        // get the positions
        float3 pos_O = make_float3(xs[idToIdxs[id_O]]);
        float3 pos_H1 = make_float3(xs[idToIdxs[id_H1]]);
        float3 pos_H2 = make_float3(xs[idToIdxs[id_H2]]);

        // get the velocities
        float3 vel_O = make_float3(vs[idToIdxs[id_O]]);
        float3 vel_H1= make_float3(vs[idToIdxs[id_H1]]);
        float3 vel_H2= make_float3(vs[idToIdxs[id_H2]]);

        float O_vel = length(vel_O);
        float H1_vel= length(vel_H1);
        float H2_vel= length(vel_H2);

        // our constraints are that the 
        // --- OH1, OH2, H1H2 bond lengths are the same;
        // --- the dot product of the bond vector with the relative velocity along the bond is zero
        
        // take a look at the bond lengths first
        // i ~ O, j ~ H1, k ~ H2
        float3 r_ij = bounds.minImage(pos_H1 - pos_O);
        float3 r_ik = bounds.minImage(pos_H2 - pos_O);
        float3 r_jk = bounds.minImage(pos_H2 - pos_H1);

        float len_rij = length(r_ij);
        float len_rik = length(r_ik);
        float len_rjk = length(r_jk);

        // these values correspond to the fixed side lengths of the triangle congruent to the specific water model being examined
        // --- or rather, they /should/, if the SETTLE algorithm is implemented correctly
        printf("molecule id %d OH1 %f OH2 %f H1H2 %f\n", idx, len_rij, len_rik, len_rjk);
 
        printf("molecule id %d vel O %f vel H1 %f vel H2 %f\n", idx, O_vel, H1_vel, H2_vel);

        // now take a look at dot product of relative velocities along the bond vector
        float3 v_ij = vel_H1 - vel_O;
        float3 v_ik = vel_H2 - vel_O;
        float3 v_jk = vel_H2 - vel_H1;

        float bond_ij = dot(r_ij, v_ij);
        float bond_ik = dot(r_ik, v_ik);
        float bond_jk = dot(r_jk, v_jk);

        // note that these values should all be zero
        printf("molecule id %d bond_ij %f bond_ik %f bond_jk %f\n", idx, bond_ij, bond_ik, bond_jk);
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
__global__ void compute_COM(int4 *waterIds, float4 *xs, float4 *vs, int *idToIdxs, int nMolecules, float4 *com, BoundsGPU bounds) {
  int idx = GETIDX();
  if (idx  < nMolecules) {
    float3 pos[3];
    float mass[3];
    int ids[3];
    ids[0] = waterIds[idx].x;
    ids[1] = waterIds[idx].y;
    ids[2] = waterIds[idx].z;
    for (int i = 0; i < 3; i++) {
      int myId = ids[i];
      int myIdx = idToIdxs[myId];
      float3 p = make_float3(xs[myIdx]);
      pos[i] = p;
      mass[i] = 1.0f / vs[myIdx].w;
      }
    for (int i=1; i<3; i++) {
      float3 delta = pos[i] - pos[0];
      delta = bounds.minImage(delta);
      pos[i] = pos[0] + delta;
    }
    float ims = com[idx].w;
    com[idx] = make_float4(positionsToCOM(pos, mass, ims));
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




// so, given that this is only to be used with water, we can 
// actually make it so that specific bond lengths are taken
// rather than having this operate on the geometries of the input
__global__ void set_fixed_sides(int4 *waterIds, float4 *xs, float4 *com, float4 *fix_len, int nMolecules, int *idToIdxs) {
  int idx = GETIDX();
  if (idx < nMolecules) {
    int ids[3];
    ids[0] = waterIds[idx].x;
    ids[1] = waterIds[idx].y;
    ids[2] = waterIds[idx].z;
    float4 pts[3];
    
    for (int i = 0; i < 3; i++) {
      int myIdx = idToIdxs[ids[i]];
      pts[i] = xs[myIdx];
    }
    float4 comCut = com[idx];
    comCut.w = 0.0f;
    float ra = fabs(length(comCut - pts[0]));
    float rc = fabs(length(pts[2] - pts[1])*0.5);
    float rb = sqrtf(length(pts[0]-pts[2])*length(pts[0]-pts[2]) - (rc*rc)) - ra;
    fix_len[idx] = make_float4(ra, rb, rc, 0.0f);
  }
} 

// ok, just initializes to zero
__global__ void set_init_vel_correction(int4 *waterIds, float4 *dvs_0, int nMolecules) {
  int idx = GETIDX();
  if (idx < nMolecules) {
    for (int i = 0; i < 3; i++) {
      dvs_0[idx*3 + i] = make_float4(0.0f,0.0f,0.0f,0.0f);
    }
  }
}

/* ------- Based off of the SETTLE Algorithm outlined in ------------
   ------ Miyamoto et al., J Comput Chem. 13 (8): 952–962 (1992). ------ */

__device__ void settle_xs(float timestep, float3 com, float3 com1, float3 *xs_0, float3 *xs, float3 fix_len) {
  
    // the positions of the initial, constrained triangle (ABC)_0
    // --- note that these are minimum image vector coordinates (see compute_SETTLE routine)
    float3 A0 = xs_0[0];
    float3 B0 = xs_0[1];
    float3 C0 = xs_0[2];
    
    // the positions after an unconstrained update - in their notation, (ABC)_1
    float3 a1 = xs[0];
    float3 b1 = xs[1];
    float3 c1 = xs[2];
    
    // the fixed lengths (see figure 2 of paper referenced above) for a canonical triangle
    // --- small note: we might just make this a single float3 type, since it will be the same for 
    //                 all water molecules in a given simulation
    float ra = fix_len.x;
    float rc = fix_len.z;
    float rb = fix_len.y;

    // the position of the canonical triangle in terms of the fixed lengths
    float3 ap0 = make_float3(0,ra,0);
    float3 bp0 = make_float3(-rc,-rb,0);
    float3 cp0 = make_float3(rc,-rb,0);

    // move the perturbed triangle to the origin by subtracting its center of mass..
    // -- com1 is the center of mass of the unconstrained triangle: so, this is D1 = D3 in Miyamoto
    float3 ap1 = a1 - com1;
    float3 bp1 = b1 - com1;
    float3 cp1 = c1 - com1;

    // see if this corrects things
    float3 Ap0 = A0 - com;
    float3 Bp0 = B0 - com;
    float3 Cp0 = C0 - com;

    //float3 normal = cross(B0-A0,C0-A0);
    //normal = normalize(normal);

    // this is a vector normal to the plane formed by triangle (ABC)_0 with its COM at the origin
    // composed by the cross product of (AB,AC)
    float3 normal = cross(Bp0 - Ap0, Cp0 - Ap0);
    normal = normalize(normal);

    float3 zaxis = make_float3(0,0,1);
    float3 r[3] = {make_float3(1,0,0),make_float3(0,1,0),make_float3(0,0,1)};
    fillRotMatrix(normal,zaxis,r);

    float3 tr[3] = {make_float3(1,0,0),make_float3(0,1,0),make_float3(0,0,1)};
    fillRotMatrix(zaxis,normal,tr);

    A0 -= com;
    B0 -= com;
    C0 -= com;
    
    A0 = rotateCoords(A0,r);
    B0 = rotateCoords(B0,r);
    C0 = rotateCoords(C0,r);

    float3 rt[3] = {make_float3(r[0].x,r[1].x,r[2].x),make_float3(r[0].y,r[1].y,r[2].y),make_float3(r[0].z,r[1].z,r[2].z)};
    float3 xaxis = make_float3(1,0,0);
    float3 yaxis = make_float3(0,1,0);
    
    zaxis = rotateCoords(zaxis,tr);
    yaxis = rotateCoords(yaxis,tr);
    xaxis = rotateCoords(xaxis,tr);
    
    float3 rt0 = normal;
    float3 rt1 = cross(ap1, rt0);
    float3 rt2 = cross(rt0, rt1);

    rt0 = normalize(rt0);
    rt1 = normalize(rt1);
    rt2 = normalize(rt2); 

    ap1 = rotateCoords(ap1,r);
    bp1 = rotateCoords(bp1,r);
    cp1 = rotateCoords(cp1,r);

    float3 a_unit = normalize(ap1);
    float a_py = sqrt(1 - a_unit.z*a_unit.z);
    float3 a_plane = make_float3(0, a_py, a_unit.z);
    
    a_unit = make_float3(a_unit.x,a_unit.y,0);
    a_unit = normalize(a_unit);
    a_plane = make_float3(0,1,0);
    float3 rotz[3] = {make_float3(1,0,0),make_float3(0,1,0),make_float3(0,0,1)};
    fillRotMatrix(a_unit,a_plane,rotz); 
    float3 trot[3] = {make_float3(rotz[0].x, rotz[1].x, rotz[2].x), make_float3(rotz[0].y, rotz[1].y, rotz[2].y), make_float3(rotz[0].z, rotz[1].z, rotz[2].z)};
    
    ap1 = rotateCoords(ap1,rotz);
    bp1 = rotateCoords(bp1,rotz);
    cp1 = rotateCoords(cp1,rotz);
  
    float sin_phi = ap1.z/ra;
    if (sin_phi >= 1.0 and sin_phi < 1.0001) {
        sin_phi = 1.0;
    }
    
    float cos_phi = 0;
    if (sin_phi > -1.0001 and sin_phi <= -1.0) {
        sin_phi = -1.0;
        cos_phi = 0;
    } else {
        cos_phi = sqrtf(1-(sin_phi*sin_phi));
    }
    
    float sin_psi = (bp1.z-cp1.z)/(2*rc*cos_phi);
    if (sin_psi >= 1.0 and sin_psi < 1.0001) {
        sin_psi = 1.0;
    }
    
    float cos_psi = 0;
    if (sin_psi > -1.0000 and sin_psi <= -1.0) {
        sin_psi = -1.0;
        cos_psi = 0;
    } else {
        cos_psi = sqrtf(1-(sin_psi*sin_psi));
    }
    
    float3 a2 = make_float3(0,ra*cos_phi,ra*sin_phi);
    float3 b2 = make_float3(-rc*cos_psi,-rb*cos_phi-rc*sin_psi*sin_phi,-rb*sin_phi+rc*sin_psi*cos_phi);
    float3 c2 = make_float3(rc*cos_psi,-rb*cos_phi+rc*sin_psi*sin_phi,-rb*sin_phi-rc*sin_psi*cos_phi);

    float alpha = b2.x*(bp0.x - cp0.x) + (bp0.y - ap0.y)*b2.y + (cp0.y - ap0.y)*c2.y;
    float beta = b2.x*(cp0.y - bp0.y) + (bp0.x - ap0.x)*b2.y + (cp0.x - ap0.x)*c2.y;
    float gamma = (bp0.x - ap0.x)*bp1.y - bp1.x*(bp0.y - ap0.y) + (cp0.x - ap0.x)*cp1.y - cp1.x*(cp0.y - ap0.y);
    float under_sqrt = alpha*alpha + beta*beta - gamma*gamma;
    float sin_theta = 0;
    if (under_sqrt > -0.0001 and under_sqrt < 0.0001) {
        sin_theta = (alpha*gamma)/(alpha*alpha + beta*beta);
    } else {
        sin_theta = (alpha*gamma - beta*sqrtf(alpha*alpha + beta*beta - gamma*gamma))/(alpha*alpha + beta*beta);
    } 
    if (sin_theta >= 1.0 and sin_theta < 1.0001) {
        sin_theta = 1.0;
    }
    float cos_theta = 0;
    if (sin_theta > -1.0001 and sin_theta <= -1.0) {
        sin_theta = -1.0;
        cos_theta = 0;
    } else {
        cos_theta = sqrtf(1 - sin_theta*sin_theta);
    }
    if((sin_theta*alpha + cos_theta*beta > (gamma + 0.001)) or ((sin_theta*alpha + cos_theta*beta) < (gamma - 0.001))) {
        sin_theta = (alpha*gamma + beta*sqrtf(alpha*alpha + beta*beta - gamma*gamma))/(alpha*alpha + beta*beta);
        if (sin_theta >= 1.0 and sin_theta < 1.0001) {
            sin_theta = 1.0;
            cos_theta = 0;
        }
        if (sin_theta > -1.0001 and sin_theta <= -1.0) {
            sin_theta = -1.0;
            cos_theta = 0;
        } else {
            cos_theta = sqrtf(1 - sin_theta*sin_theta);
        }
    }
  //printf("sin_theta = %f  cos_theta = %f  alpha = %f  beta = %f  gamma = %f\n", sin_theta, cos_theta, alpha, beta, gamma);
    if (!(fabs(sin_phi) <= 1.00001) or (!((fabs(sin_psi) <= 1.00001))) or (!(fabs(sin_theta) <= 1.00001))) {
        printf("fabs(sin_phi) : %f\n", fabs(sin_phi));
        printf("fabs(sin_psi) : %f\n", fabs(sin_psi));
        printf("fabs(sin_theta): %f\n", fabs(sin_theta));
        assert(fabs(sin_phi) <= 1.00001);
        assert(fabs(sin_psi) <= 1.00001);
        assert(fabs(sin_theta) <= 1.00001);
    }

    float3 a3 = make_float3(a2.x*cos_theta-a2.y*sin_theta,a2.x*sin_theta+a2.y*cos_theta,a2.z);
    float3 b3 = make_float3(b2.x*cos_theta-b2.y*sin_theta,b2.x*sin_theta+b2.y*cos_theta,b2.z);
    float3 c3 = make_float3(c2.x*cos_theta-c2.y*sin_theta,c2.x*sin_theta+c2.y*cos_theta,c2.z);

    // rotate back to original coordinate system
    a3 = rotateCoords(a3, trot);
    b3 = rotateCoords(b3, trot);
    c3 = rotateCoords(c3, trot);
    

    a3 = rotateCoords(a3,tr);
    b3 = rotateCoords(b3,tr);
    c3 = rotateCoords(c3,tr);
    
    float3 da = a3 - A0;
    float3 db = b3 - B0;
    float3 dc = c3 - C0;
 

    xs[0] = a3 + com1;
    xs[1] = b3 + com1;
    xs[2] = c3 + com1;
}

__device__ void settle_vs(float dt_, float dtf_, float3 *vs_0, float3 *dvs_0, float3 *vs, float3 *xs, float3 *xs_0, float *mass, float3 *fs_0, float3 *fs) {
    float dt = dt_;
    float dtf = dtf_;
    // calculate velocities
    
    float ma = mass[0];
    float mb = mass[1];
    float mc = mass[2];

    float3 v0a = vs_0[0];
    float3 v0b = vs_0[1];
    float3 v0c = vs_0[2];

    float3 a3 = xs[0];
    float3 b3 = xs[1];
    float3 c3 = xs[2];

    float3 v0ab = v0b - v0a;
    float3 v0bc = v0c - v0b;
    float3 v0ca = v0a - v0c;

    // direction vectors
    float3 eab = b3 - a3;
    float3 ebc = c3 - b3;
    float3 eca = a3 - c3;
   
    eab = normalize(eab);
    ebc = normalize(ebc);
    eca = normalize(eca);
    
    float cosA = dot(-eab,eca);
    float cosB = dot(-ebc,eab);
    float cosC = dot(-eca,ebc);
    
    float d = 2*(ma+mb)*(ma+mb) + 2*ma*mb*cosA*cosB*cosC;
    d -= 2*mb*mb*cosA*cosA + ma*(ma+mb)*(cosB*cosB + cosC*cosC);
    d *= dt/(2*mb);
    float vab = dot(eab,v0ab);
    float vbc = dot(ebc,v0bc);
    float vca = dot(eca,v0ca);
    float tab = vab * (2*(ma + mb) - ma*cosC*cosC);
   
    tab += vbc * (mb*cosC*cosA - (ma + mb)*cosB);
    tab += vca * (ma*cosB*cosC - 2*mb*cosA);
    tab *= ma/d;
   
    float tbc = vbc * ((ma+mb)*(ma+mb) - mb*mb*cosA*cosA);
    tbc += vca*ma * (mb*cosA*cosB - (ma + mb)*cosC);
    tbc += vab*ma * (mb*cosC*cosA - (ma + mb)*cosB);
    tbc /= d;
    float tca = vca * (2*(ma + mb) - ma*cosB*cosB);
   
    tca += vab * (ma*cosB*cosC - 2*mb*cosA);
    tca += vbc * (mb*cosA*cosB - (ma + mb)*cosC);
    tca *= ma/d;
   
    float3 dva = (dt/(2*ma))*(tab*eab - tca*eca);
    float3 dvb = (dt/(2*mb))*(tbc*ebc - tab*eab);
    float3 dvc = (dt/(2*mc))*(tca*eca - tbc*ebc);


    dvs_0[0] = dva;
    dvs_0[1] = dvb;
    dvs_0[2] = dvc;
    
    v0a += dva;
    v0b += dvb;
    v0c += dvc;
    
    float3 va = v0a;
    float3 vb = v0b;
    float3 vc = v0c;

    vs[0] = va;
    vs[1] = vb;
    vs[2] = vc;
}

/*
__global__ void updateVs(int4 *waterIds, float4 *xs, float4 *xs_0, float4 *vs, float4 *vs_0, float4 *dvs_0, float4 *fs, float4 *fs_0, float4 *comOld, int nMolecules, float dt, float dtf, int *idToIdxs, BoundsGPU bounds)  {

    int idx = GETIDX();
    if (idx < nMolecules) {

        int ids[3];
        float3 dvs_0_mol[3];
        int3 waterId_mol = make_int3(waterIds[idx]);
        ids[0] = waterId_mol.x;
        ids[1] = waterId_mol.y;
        ids[2] = waterId_mol.z;
        for (int i = 0; i < 3; i++) {
            int myIdx = idToIdxs[ids[i]];
            float4 vWhole = vs[myIdx];
            dvs_0_mol[i] = make_float3(dvs_0[idx*3+i]);
            vWhole += dvs_0_mol[i];
            vs[myIdx] = vWhole;
        }
    }
}
*/


__global__ void compute_SETTLE(int4 *waterIds, float4 *xs, float4 *xs_0, float4 *vs, float4 *vs_0, float4 *dvs_0, float4 *fs, float4 *fs_0, float4 *comOld, float4 *fix_len, int nMolecules, float dt, float dtf, int *idToIdxs, BoundsGPU bounds) {
    int idx = GETIDX();
    if (idx < nMolecules) {

        // atom ids, idToIdxs[ids], positions at previous turn, current positions, velocities at previous turn, 
        // current velocities, displacement velocities due to constraint forces, previous forces, current forces, and masses
        int ids[3];
        int idxs[3];
        float3 xs_0_mol[3];
        float3 xs_mol[3];
        float3 vs_0_mol[3];
        float3 vs_mol[3];
        float3 dvs_0_mol[3];
        float3 fs_0_mol[3];
        float3 fs_mol[3];
        float mass[3];

        // the fixed sides of the triangle
        float3 fix_len_mol = make_float3(fix_len[idx]);
        // get the ids of the O, H1, H2 atoms for the molecule at this idx
        int3 waterId_mol = make_int3(waterIds[idx]);
        ids[0] = waterId_mol.x;
        ids[1] = waterId_mol.y;
        ids[2] = waterId_mol.z;

        // here, we get the absolute positions, velocities, and forces, from the previous step and
        // the current step, of the constituent atoms for this molecule
        for (int i = 0; i < 3; i++) {
            // call idToIdxs on the ids
            int myIdx = idToIdxs[ids[i]];
            idxs[i] = myIdx;
            xs_0_mol[i] = make_float3(xs_0[idx*3+i]);
            float4 xWhole = xs[myIdx];
            xs_mol[i] = make_float3(xWhole);
            vs_0_mol[i] = make_float3(vs_0[idx*3+i]);
            float4 vWhole = vs[myIdx];
            vs_mol[i] = make_float3(vWhole);
            dvs_0_mol[i] = make_float3(dvs_0[idx*3+i]);
            mass[i] = 1.0f / vWhole.w;
            fs_0_mol[i] = make_float3(fs_0[idx*3+i]);
            float4 fWhole = fs[myIdx];
            fs_mol[i] = make_float3(fWhole);
        }

        // re-write the positions as minimum image displacements from the position of the oxygen atom
        for (int i=1; i<3; i++) {
            float3 delta = xs_mol[i] - xs_mol[0];
            delta = bounds.minImage(delta);
            xs_mol[i] = xs_mol[0] + delta;
        }

        // do the same thing for the array holding position data from last turn
        for (int i=0; i<3; i++) {
            float3 delta = xs_0_mol[i] - xs_mol[0];
            delta = bounds.minImage(delta);
            xs_0_mol[i] = xs_mol[0] + delta;
        }
        
        // compute the new center of mass using the minimum image coordinates
        float3 comNew = positionsToCOM(xs_mol, mass, comOld[idx].w);
        float3 delta = make_float3(comOld[idx]) - comNew;
        // minimum image vector between the new and old center of mass
        delta = bounds.minImage(delta);
        float3 comOldWrap = comNew + delta;

        // routine solving positional restraints
        settle_xs(dt, comOldWrap, comNew, xs_0_mol, xs_mol, fix_len_mol);

        // routine solving velocity constraints
        settle_vs(dt, dtf, vs_0_mol, dvs_0_mol, vs_mol, xs_mol, xs_0_mol, mass, fs_0_mol, fs_mol);

        // set the value of the positions in global memory to the constrained solutions
        for (int i=0; i<3; i++) {
            xs[idxs[i]] = make_float4(xs_mol[i]);
        }
        // same thing with velocities; save the constraint forces for this computation
        for (int i=0; i<3; i++) {
            vs[idxs[i]] = make_float4(vs_mol[i]);
            vs[idxs[i]].w = 1.0f/mass[i];
            dvs_0[idx*3+i] = make_float4(dvs_0_mol[i]);
        }
    }
}


void FixRigid::handleBoundsChange() {

    if (TIP4P) {
    
        GPUData &gpd = state->gpd;
        int activeIdx = gpd.activeIdx();
        BoundsGPU &bounds = state->boundsGPU;
        
        int nAtoms = state->atoms.size();
        if (printing) {
            printf("Calling printGPD_Rigid at turn %d\n in FixRigid::handleBoundsChange, before doing anything\n", state->turn);
            cudaDeviceSynchronize();
            printGPD_Rigid<<<NBLOCK(nAtoms), PERBLOCK>>>(gpd.ids(activeIdx),
                                                         gpd.xs(activeIdx),gpd.vs(activeIdx),gpd.fs(activeIdx),nAtoms);
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

            printf("Calling printGPD_Rigid at turn %d\n in FixRigid::handleBoundsChange, after calling setMSite\n", state->turn);
            printGPD_Rigid<<<NBLOCK(nAtoms), PERBLOCK>>>(gpd.ids(activeIdx),gpd.xs(activeIdx),
                                                         gpd.vs(activeIdx),gpd.fs(activeIdx),nAtoms);
            cudaDeviceSynchronize();
        }
    }

    return;


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
    if (TIP3P && TIP4P) {
        mdError("An attempt was made to use both TIP3P and TIP4P in a simulation");
    }

    nMolecules = waterIds.size();
    int n = waterIds.size();
    printf("number of molecules in waterIds: %d\n", n);
    waterIdsGPU = GPUArrayDeviceGlobal<int4>(n);
    waterIdsGPU.set(waterIds.data());

    xs_0 = GPUArrayDeviceGlobal<float4>(3*n);
    vs_0 = GPUArrayDeviceGlobal<float4>(3*n);
    dvs_0 = GPUArrayDeviceGlobal<float4>(3*n);
    fs_0 = GPUArrayDeviceGlobal<float4>(3*n);
    com = GPUArrayDeviceGlobal<float4>(n);
    com.set(invMassSums.data());
    fix_len = GPUArrayDeviceGlobal<float4>(n);
    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();

    // compute the force partition constant
    if (TIP4P) {
        compute_gamma();
    }

    BoundsGPU &bounds = state->boundsGPU;
    printf("FixRigid::prepareForRun: compute_COM<<<>>> about to call\n");
    SAFECALL((compute_COM<<<NBLOCK(n), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), gpd.vs(activeIdx), 
                                         gpd.idToIdxs.d_data.data(), n, com.data(), bounds)));

    cudaDeviceSynchronize();
    printf("FixRigid::prepareForRun: set_fixed_sides<<<>>> about to call\n");
    
    set_fixed_sides<<<NBLOCK(n), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), com.data(), 
                                             fix_len.data(), n, gpd.idToIdxs.d_data.data());
    cudaDeviceSynchronize();
    printf("FixRigid::prepareForRun: set_init_vel_correction<<<>>> about to call\n");
    set_init_vel_correction<<<NBLOCK(n), PERBLOCK>>>(waterIdsGPU.data(), dvs_0.data(), n);
    cudaDeviceSynchronize();
    printf("done with FixRigid::prepareForRun()\n");
    int nAtoms = state->atoms.size();
    printf("Calling printGPD_Rigid at turn %d\n in FixRigid::prepareForRun, before doing anything\n", state->turn);
    //printGPD_Rigid<<<NBLOCK(nAtoms), PERBLOCK>>>(gpd.ids(activeIdx),gpd.xs(activeIdx),gpd.vs(activeIdx),gpd.fs(activeIdx),nAtoms);
  
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
            printf("Calling printGPD_Rigid at turn %d\n in FixRigid::stepInit, before doing anything\n", state->turn);
            printGPD_Rigid<<<NBLOCK(nAtoms), PERBLOCK>>>(gpd.ids(activeIdx),gpd.xs(activeIdx),gpd.vs(activeIdx),gpd.fs(activeIdx),nAtoms);
            cudaDeviceSynchronize();
        }
    }

    // compute the current center of mass for the solved constraints at the beginning of this turn
    compute_COM<<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                           gpd.vs(activeIdx), gpd.idToIdxs.d_data.data(), 
                                           nMolecules, com.data(), bounds);

    // save the positions, velocities, forces from the previous, fully updated turn in to our local arrays
    compute_prev_val<<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), xs_0.data(), 
                                                gpd.vs(activeIdx), vs_0.data(), gpd.fs(activeIdx), 
                                                fs_0.data(), nMolecules, gpd.idToIdxs.d_data.data());

    // update the velocities /after/ doing a half step again.  Else, we are updating on the constraints
    /*
    updateVs<<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), xs_0.data(),
                                          gpd.vs(activeIdx), vs_0.data(), dvs_0.data(),  gpd.fs(activeIdx),
                                          fs_0.data(), com.data(), nMolecules, dt, gpd.idToIdxs.d_data.data(), bounds);
    
    // and reset the prev vals arrays, since we modified the velocities
    compute_prev_val<<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), xs_0.data(), 
                                                gpd.vs(activeIdx), vs_0.data(), gpd.fs(activeIdx), 
                                                fs_0.data(), nMolecules, gpd.idToIdxs.d_data.data());
    */
//__global__ void updateVs(int4 *waterIds, float4 *xs, float4 *xs_0, float4 *vs, float4 *vs_0, float4 *dvs_0, float4 *fs, float4 *fs_0, int nMolecules, float dt)  {


    if (TIP4P) {
        if (printing) {
            cudaDeviceSynchronize();
            printf("Calling printGPD_Rigid at turn %d\n in FixRigid::stepInit, after doing compute_COM and compute_prev_val\n", state->turn);
            printGPD_Rigid<<<NBLOCK(nAtoms), PERBLOCK>>>(gpd.ids(activeIdx),gpd.xs(activeIdx),
                                                         gpd.vs(activeIdx),gpd.fs(activeIdx),nAtoms);
            cudaDeviceSynchronize();
        }
    
    }
    //xs_0.get(cpu_com);
    //std::cout << cpu_com[0] << "\n";
    return true;
}

bool FixRigid::postNVE_V() {

    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();
    BoundsGPU &bounds = state->boundsGPU;
    //float dtf = 0.5f * state->dt * state->units.ftm_to_v;
    int nAtoms = state->atoms.size();
    float dt = state->dt;

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

    // from IntegratorVerlet
    if (TIP4P) {

        if (printing) {
            cudaDeviceSynchronize();
            printf("Calling printGPD_Rigid at turn %d\n in FixRigid::stepFinal, before doing anything\n", state->turn);
            printGPD_Rigid<<<NBLOCK(nAtoms), PERBLOCK>>>(gpd.ids(activeIdx),gpd.xs(activeIdx),gpd.vs(activeIdx),gpd.fs(activeIdx),nAtoms);
            cudaDeviceSynchronize();
        }
        float dtf = 0.5f * state->dt * state->units.ftm_to_v;
        distributeMSite<<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                                     gpd.vs(activeIdx),  gpd.fs(activeIdx),
                                                     gpd.virials.d_data.data(),
                                                     nMolecules, gamma, dtf, gpd.idToIdxs.d_data.data(), bounds);

        if (printing) { 
            cudaDeviceSynchronize();
            printf("Calling printGPD_Rigid at turn %d\n in FixRigid::stepFinal, after calling distributeMSite\n", state->turn);
            printGPD_Rigid<<<NBLOCK(nAtoms), PERBLOCK>>>(gpd.ids(activeIdx),gpd.xs(activeIdx),gpd.vs(activeIdx),gpd.fs(activeIdx),nAtoms);
            cudaDeviceSynchronize();
        }

    }


    compute_SETTLE<<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                                xs_0.data(), gpd.vs(activeIdx), vs_0.data(), 
                                                dvs_0.data(), gpd.fs(activeIdx), fs_0.data(), 
                                                com.data(), fix_len.data(), nMolecules, dt, dtf,  
                                                gpd.idToIdxs.d_data.data(), bounds);

 
    //printf("did compute_SETTLE!\n");
    // finally, reset the position of the M-site to be consistent with that of the new, constrained water molecule
    if (TIP4P) {
        if (printing) { 
            printf("Calling printGPD_Rigid at turn %d\n in FixRigid::stepFinal, before calling setMSite \n", state->turn);
            cudaDeviceSynchronize();
            printGPD_Rigid<<<NBLOCK(nAtoms), PERBLOCK>>>(gpd.ids(activeIdx),gpd.xs(activeIdx),gpd.vs(activeIdx),gpd.fs(activeIdx),nAtoms);
        
            cudaDeviceSynchronize();
        }
        setMSite<<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.idToIdxs.d_data.data(), gpd.xs(activeIdx), nMolecules, bounds);
        
        if (printing) { 
            cudaDeviceSynchronize();
            printf("Calling printGPD_Rigid at turn %d\n in FixRigid::stepFinal, after calling setMSite \n", state->turn);
            printGPD_Rigid<<<NBLOCK(nAtoms), PERBLOCK>>>(gpd.ids(activeIdx),gpd.xs(activeIdx),gpd.vs(activeIdx),gpd.fs(activeIdx),nAtoms);
            cudaDeviceSynchronize();
        }
    }
 
    validateConstraints<<<NBLOCK(nMolecules), PERBLOCK>>> (waterIdsGPU.data(), gpd.idToIdxs.d_data.data(), gpd.xs(activeIdx), 
                                                      gpd.vs(activeIdx), nMolecules, bounds);
    
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
	.def_readwrite("printing", &FixRigid::printing)
    ;
}



