#include "FixRigid.h"

#include "State.h"
#include "VariantPyListInterface.h"
#include "boost_for_export.h"
#include "cutils_math.h"
#include "cutils_func.h"
#include <math.h>
using namespace std;
namespace py = boost::python;
const string rigidType = "Rigid";

FixRigid::FixRigid(boost::shared_ptr<State> state_, string handle_, string groupHandle_) : Fix(state_, handle_, groupHandle_, rigidType, true, true, false, 1) {

    // set both to false initially; using one of the createRigid functions will flip the pertinent flag to true
    firstPrepare = true;
    TIP4P = false;
    TIP3P = false;
    style = "DEFAULT"; 
    styleSet = false;
    printing = false;
}

__device__ inline float3 positionsToCOM(float3 *pos, float *mass, float ims) {
  return (pos[0]*mass[0] + pos[1]*mass[1] + pos[2]*mass[2])*ims;
}

inline __host__ __device__ float3 rotateCoords(float3 vector, float3 matrix[]) {
  return make_float3(dot(matrix[0],vector),dot(matrix[1],vector),dot(matrix[2],vector));
}

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

__global__ void distributeMSite(int4 *waterIds, float4 *xs, float4 *vs, float4 *fs, 
                                int nMols, float gamma, float dtf, int* idToIdxs, BoundsGPU bounds)

{
    int idx = GETIDX();
    if (idx < nMols) {
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

__global__ void setMSite(int4 *waterIds, int *idToIdxs, float4 *xs, int nMols, BoundsGPU bounds) {

    int idx = GETIDX();
    if (idx < nMols) {
    
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
        /* so maybe this is what is causing a lot of the problems?
        // now, referring to 'periodicWrap in ../GridGPU.cu, apply PBC to the computed M site position
        // -- we can assume the atoms positions from which M-site is composed are already in the box (raw positions, at least)
        float3 trace = bounds.trace();
        float3 diffFromLo = r_M - bounds.lo;
        float3 imgs = floorf(diffFromLo / trace);
        r_M -= (trace * imgs * bounds.periodic);
        float4 pos_M_new = make_float4(r_M.x, r_M.y, r_M.z, pos_M_whole.w);
        if (imgs.x != 0 or imgs.y != 0 or imgs.z != 0) {
            xs[idToIdxs[id_M]] = pos_M_new;
        }
        */

    }

}

// computes the center of mass for a given water molecule
__global__ void compute_COM(int4 *waterIds, float4 *xs, float4 *vs, int *idToIdxs, int nMols, float4 *com, BoundsGPU bounds) {
  int idx = GETIDX();
  if (idx  < nMols) {
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

__global__ void compute_prev_val(int4 *waterIds, float4 *xs, float4 *xs_0, float4 *vs, float4 *vs_0, float4 *fs, float4 *fs_0, int nMols, int *idToIdxs) {
  int idx = GETIDX();
  if (idx < nMols) {
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
__global__ void set_fixed_sides(int4 *waterIds, float4 *xs, float4 *com, float4 *fix_len, int nMols, int *idToIdxs) {
  int idx = GETIDX();
  if (idx < nMols) {
    int ids[3];
    ids[0] = waterIds[idx].x;
    ids[1] = waterIds[idx].y;
    ids[2] = waterIds[idx].z;
    float4 pts[3];
    //float side_ab = length(xs[idToIdxs[ids[1]]] - xs[idToIdxs[ids[0]]]);
    //float side_bc = length(xs[idToIdxs[ids[2]]] - xs[idToIdxs[ids[1]]]);
    //float side_ca = length(xs[idToIdxs[ids[0]]] - xs[idToIdxs[ids[2]]]);
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

__global__ void set_init_vel_correction(int4 *waterIds, float4 *dvs_0, int nMols) {
  int idx = GETIDX();
  if (idx < nMols) {
    for (int i = 0; i < 3; i++) {
      dvs_0[idx*3 + i] = make_float4(0.0f,0.0f,0.0f,0.0f);
    }
  }
}

/* ------- Based off of the SETTLE Algorithm outlined in ------------
   ------ Miyamoto et al., J Comput Chem. 13 (8): 952–962 (1992). ------ */

__device__ void settle_xs(float timestep, float3 com, float3 com1, float3 *xs_0, float3 *xs, float3 fix_len) {

 __device__ void settle_vs(float timestep, float3 *vs_0, float3 *dvs_0, float3 *vs, float3 *xs, float3 *xs_0, float *mass, float3 *fs_0, float3 *fs) {

 
}



__global__ void compute_SETTLE(int4 *waterIds, float4 *xs, float4 *xs_0, float4 *vs, float4 *vs_0, float4 *dvs_0, float4 *fs, float4 *fs_0, float4 *comOld, float4 *fix_len, int nMols, float dt, int *idToIdxs, BoundsGPU bounds) {
  int idx = GETIDX();
  if (idx < nMols) {

  
  
  
}


void FixRigid::handleBoundsChange() {

    if (TIP4P) {
    
        int nMols = waterIdsGPU.size();
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
        
        setMSite<<<NBLOCK(nMols), PERBLOCK>>>(waterIdsGPU.data(), gpd.idToIdxs.d_data.data(), gpd.xs(activeIdx), nMols, bounds);
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

    /*
    printf("waterMoleculeIds: %d %d %d %d\n", waterMolecule.x,
                                              waterMolecule.y,
                                              waterMolecule.z, 
                                              waterMolecule.w);
    */
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
    // we need the forces for this fix
    if (firstPrepare) {
        firstPrepare = false;
        return false;
    }

    // cannot have more than one water model present
    if (TIP3P && TIP4P) {
        mdError("An attempt was made to use both TIP3P and TIP4P models or variants in a simulation");
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


    int n = waterIds.size();
    printf("number of molecules in waterIds: %d\n", n);
    waterIdsGPU = GPUArrayDeviceGlobal<int4>(n);
    waterIdsGPU.set(waterIds.data());

    // stores local copies of the bond lengths OH, HH, and OM (if necessary) for the given model
    setStyleBondLengths();
    
    // computes the side lengths associated with the Pseudo-Euler lengths 
    // as defined by Miyamoto & Kollman
    computeEulerLengths();
    // this varies from model to model - it will use the 

    xs_0 = GPUArrayDeviceGlobal<float4>(3*n);
    vs_0 = GPUArrayDeviceGlobal<float4>(3*n);
    dvs_0 = GPUArrayDeviceGlobal<float4>(3*n);
    fs_0 = GPUArrayDeviceGlobal<float4>(3*n);
    com = GPUArrayDeviceGlobal<float4>(n);
    com.set(invMassSums.data());
    fix_len = GPUArrayDeviceGlobal<float4>(n);
    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();

    // bool TIP4P == true if we are using any 4-site model (any model with m-site)
    // compute the force partition constant
    if (TIP4P) {
        compute_gamma();
    }

    BoundsGPU &bounds = state->boundsGPU;
    int nAtoms = state->atoms.size();
  
    // we need to compute the center of mass for each particle
    computeCOM<<<NBLOCK(nMolecules),PERBLOCK>>>(*args);

    // solve the rigid bond constraints
    computeXs<<<NBLOCK(nMolecules),PERBLOCK>>>(*args);

    // compute the constraint forces
    computeConstraintForces<<<NBLOCK(nMolecules),PERBLOCK>>>(*args);

    // add the constraint forces to the global array of forces
    // ---- we can either add the constraint forces manually in this routine, 
    //      or modify the global array of forces acting on the atoms in the molecule.  Which is better?
    computeVs<<<NBLOCK(nMolecules),PERBLOCK>>>(*args)


    return true;
}

bool FixRigid::stepInit() {
    
    // so, stepInit, what we want to do is...?
    int nMols = waterIdsGPU.size();
    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();
    BoundsGPU &bounds = state->boundsGPU;
    //float dtf = 0.5f * state->dt * state->units.ftm_to_v;
    int nAtoms = state->atoms.size();
    if (TIP4P) {
        if (printing) {
            cudaDeviceSynchronize();
            printf("Calling printGPD_Rigid at turn %d\n in FixRigid::stepInit, before doing anything\n", state->turn);
            printGPD_Rigid<<<NBLOCK(nAtoms), PERBLOCK>>>(gpd.ids(activeIdx),gpd.xs(activeIdx),gpd.vs(activeIdx),gpd.fs(activeIdx),nAtoms);
            cudaDeviceSynchronize();
        }
    }
    compute_COM<<<NBLOCK(nMols), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                           gpd.vs(activeIdx), gpd.idToIdxs.d_data.data(), 
                                           nMols, com.data(), bounds);
    compute_prev_val<<<NBLOCK(nMols), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), xs_0.data(), 
                                                gpd.vs(activeIdx), vs_0.data(), gpd.fs(activeIdx), 
                                                fs_0.data(), nMols, gpd.idToIdxs.d_data.data());


    if (TIP4P) {
        if (printing) {
            cudaDeviceSynchronize();
            printf("Calling printGPD_Rigid at turn %d\n in FixRigid::stepInit, after doing compute_COM and compute_prev_val\n", state->turn);
            printGPD_Rigid<<<NBLOCK(nAtoms), PERBLOCK>>>(gpd.ids(activeIdx),gpd.xs(activeIdx),
                                                         gpd.vs(activeIdx),gpd.fs(activeIdx),nAtoms);
            cudaDeviceSynchronize();
        }
    
    }
    //float4 cpu_com[nMols*3];
    //xs_0.get(cpu_com);
    //std::cout << cpu_com[0] << "\n";
    return true;
}

bool FixRigid::stepFinal() {
    int nMols = waterIdsGPU.size();
    float dt = state->dt;
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
        distributeMSite<<<NBLOCK(nMols), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                                     gpd.vs(activeIdx),  gpd.fs(activeIdx),
                                                     nMols, gamma, dtf, gpd.idToIdxs.d_data.data(), bounds);

        if (printing) { 
            cudaDeviceSynchronize();
            printf("Calling printGPD_Rigid at turn %d\n in FixRigid::stepFinal, after calling distributeMSite\n", state->turn);
            printGPD_Rigid<<<NBLOCK(nAtoms), PERBLOCK>>>(gpd.ids(activeIdx),gpd.xs(activeIdx),gpd.vs(activeIdx),gpd.fs(activeIdx),nAtoms);
            cudaDeviceSynchronize();
        }

    }


    compute_SETTLE<<<NBLOCK(nMols), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                                xs_0.data(), gpd.vs(activeIdx), vs_0.data(), 
                                                dvs_0.data(), gpd.fs(activeIdx), fs_0.data(), 
                                                com.data(), fix_len.data(), nMols, dt, 
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
        setMSite<<<NBLOCK(nMols), PERBLOCK>>>(waterIdsGPU.data(), gpd.idToIdxs.d_data.data(), gpd.xs(activeIdx), nMols, bounds);
        
        if (printing) { 
            cudaDeviceSynchronize();
            printf("Calling printGPD_Rigid at turn %d\n in FixRigid::stepFinal, after calling setMSite \n", state->turn);
            printGPD_Rigid<<<NBLOCK(nAtoms), PERBLOCK>>>(gpd.ids(activeIdx),gpd.xs(activeIdx),gpd.vs(activeIdx),gpd.fs(activeIdx),nAtoms);
            cudaDeviceSynchronize();
        }
    }
    
    
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



