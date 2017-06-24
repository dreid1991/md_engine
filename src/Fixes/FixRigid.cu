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
    TIP4P = false;
    TIP3P = false;
}

__device__ inline float3 positionsToCOM(float3 *pos, float *mass, float ims) {
  return (pos[0]*mass[0] + pos[1]*mass[1] + pos[2]*mass[2])*ims;
}

inline __host__ __device__ float3 rotateCoords(float3 vector, float3 matrix[]) {
  return make_float3(dot(matrix[0],vector),dot(matrix[1],vector),dot(matrix[2],vector));
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

        //printf("In distributeMSite, working on ids %d %d %d %d\n", id_O, id_H1, id_H2, id_M);
        // so, we need the forces and velocities of the atoms in the molecule
        // -- just the velocities of the real atoms: O, H1, H2
        int O_idx = idToIdxs[id_O];
        int H1_idx =idToIdxs[id_H1];
        int H2_idx= idToIdxs[id_H2];
        int M_idx = idToIdxs[id_M];
        
        //printf("In thread %d, O H1 H2 M idxs are %d %d %d %d\n", idx, O_idx, H1_idx, H2_idx, M_idx);

        float4 vel_O_whole = vs[idToIdxs[id_O]];
        float4 vel_H1 = vs[idToIdxs[id_H1]];
        float4 vel_H2 = vs[idToIdxs[id_H2]];

        float3 vel_O = make_float3(vel_O_whole);
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
        float invMassO = vel_O_whole.w;

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
        /*
        vs[idToIdxs[id_O]] = vel_O;
        vs[idToIdxs[id_H1]]= vel_H1;
        vs[idToIdxs[id_H2]]= vel_H2;
        */
        // finally, modify the forces; this way, the distributed force from M-site is incorporated in to nve_v() integration step
        // at beginning of next iteration in IntegratorVerlet.cu
        fs_O += fs_O_d;
        fs_H1 += fs_H_d;
        fs_H2 += fs_H_d;
       
        // set the global variables *fs[idToIdx[id]] to the new values
        fs[idToIdxs[id_O]] = fs_O;
        fs[idToIdxs[id_H1]]= fs_H1;
        fs[idToIdxs[id_H2]]= fs_H2;

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
    ids[0] = waterIds[idx].x;
    ids[1] = waterIds[idx].y;
    ids[2] = waterIds[idx].z;
    for (int i = 0; i < 3; i++) {
      int myIdx = idToIdxs[ids[i]];
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
  //printf("COM = %f %f %f  len = %f %f %f\n", com.x, com.y, com.x, fix_len.x, fix_len.y, fix_len.z);
  float3 a0 = xs_0[0];
  float3 b0 = xs_0[1];
  float3 c0 = xs_0[2];
  //printf("a0=%f %f %f\n",a0.x,a0.y,a0.z);

  float3 a1 = xs[0];
  float3 b1 = xs[1];
  float3 c1 = xs[2];
  //printf("a1=%f %f %f  b1=%f %f %f  c1=%f %f %f\n",a1.x,a1.y,a1.z,b1.x,b1.y,b1.z,c1.x,c1.y,c1.z);
  
  float ra = fix_len.x;
  float rc = fix_len.z;
  float rb = fix_len.y;
  float3 ap0 = make_float3(0,ra,0);
  float3 bp0 = make_float3(-rc,-rb,0);
  float3 cp0 = make_float3(rc,-rb,0);
  //printf("ap0=%f %f %f  ra=%f\n",ap0.x,ap0.y,ap0.z,ra);
  //printf("bp0=%f %f %f  rb=%f\n",bp0.x,bp0.y,bp0.z,rb);
  //printf("cp0=%f %f %f  rc=%f\n",cp0.x,cp0.y,cp0.z,rc);
  
  // move ∆'1 to the origin
  float3 ap1 = a1 - com1;
  float3 bp1 = b1 - com1;
  float3 cp1 = c1 - com1;
  //printf("com1=%f %f %f  ap1=%f %f %f  bp1=%f %f %f  cp1=%f %f %f\n",com1.x,com1.y,com1.z, ap1.x,ap1.y,ap1.z, bp1.x,bp1.y,bp1.z, cp1.x,cp1.y,cp1.z);
  //printf("coms = %f %f %f\n", (com1-com).x, (com1-com).y, (com1-com).z);
  // construct rotation matrix for z-axis to normal
  float3 normal = cross(b0-a0,c0-a0);
  normal = normalize(normal);
  float3 zaxis = make_float3(0,0,1);
  //printf("normal= %f %f %f\n", normal.x, normal.y, normal.z);
  float3 r[3] = {make_float3(1,0,0),make_float3(0,1,0),make_float3(0,0,1)};
  fillRotMatrix(normal,zaxis,r);
  //printf("r=%f %f %f | %f %f %f | %f %f %f \n",r[0].x,r[0].y,r[0].z,r[1].x,r[1].y,r[1].z,r[2].x,r[2].y,r[2].z);

  float3 tr[3] = {make_float3(1,0,0),make_float3(0,1,0),make_float3(0,0,1)};
  fillRotMatrix(zaxis,normal,tr);

  a0 -= com;
  b0 -= com;
  c0 -= com;
  a0 = rotateCoords(a0,r);
  b0 = rotateCoords(b0,r);
  c0 = rotateCoords(c0,r);
  //printf("a0 = %f %f %f  b0 = %f %f %f  c0 = %f %f %f\n",a0.x,a0.y,a0.z,b0.x,b0.y,b0.z,c0.x,c0.y,c0.z);

  float3 rt[3] = {make_float3(r[0].x,r[1].x,r[2].x),make_float3(r[0].y,r[1].y,r[2].y),make_float3(r[0].z,r[1].z,r[2].z)};
  float3 xaxis = make_float3(1,0,0);
  float3 yaxis = make_float3(0,1,0);
  
  zaxis = rotateCoords(zaxis,tr);
  yaxis = rotateCoords(yaxis,tr);
  xaxis = rotateCoords(xaxis,tr);
  
  if (zaxis.x != normal.x or zaxis.y != normal.y) {
    //printf("Rotation matrix is not correct  normal: %f %f\n",normal.x,normal.y);
  }

  float3 rt0 = normal;
  float3 rt1 = cross(ap1, rt0);
  float3 rt2 = cross(rt0, rt1);
  //printf("normal = %f %f %f  cross(ap1,n) = %f %f %f cross(n,m) = %f %f %f\n", rt0.x, rt0.y, rt0.z, rt1.x, rt1.y, rt1.z, rt2.x, rt2.y, rt2.z);

  rt0 = normalize(rt0);
  rt1 = normalize(rt1);
  rt2 = normalize(rt2); 
  //printf("normal = %f %f %f  cross(ap1,n) = %f %f %f cross(n,m) = %f %f %f\n", rt0.x, rt0.y, rt0.z, rt1.x, rt1.y, rt1.z, rt2.x,rt2.y, rt2.z);

  //float3 rtn[3] = {rt1, rt2, rt0};
  //printf("rtn=%f %f %f | %f %f %f | %f %f %f\n",rtn[0].x,rtn[0].y,rtn[0].z,rtn[1].x,rtn[1].y,rtn[1].z,rtn[2].x,rtn[2].y,rtn[2].z);
  //float3 trn[3] = {make_float3(rt1.x, rt2.x, rt0.x), make_float3(rt1.y, rt2.y, rt0.y), make_float3(rt1.z, rt2.z, rt0.z)};

  ap1 = rotateCoords(ap1,r);
  bp1 = rotateCoords(bp1,r);
  cp1 = rotateCoords(cp1,r);

  //ap1 = rotateCoords(ap1,rtn);
  //bp1 = rotateCoords(bp1,rtn);
  //cp1 = rotateCoords(cp1,rtn); 

  //printf("plane-rotated ap1 = %f %f %f  bp1 = %f %f %f  cp1 = %f %f %f\n",ap1.x,ap1.y,ap1.z,bp1.x,bp1.y,bp1.z,cp1.x,cp1.y,cp1.z);
  /*
  float3 yz_plane = make_float3(0,fabs(ap1.y),ap1.z);
  yz_plane = normalize(yz_plane);
  float3 rotz[3] = {make_float3(1,0,0),make_float3(0,1,0),make_float3(0,0,1)};
  fillRotMatrix(ap1,yz_plane,rotz);

  float3 trot[3] = {make_float3(1,0,0),make_float3(0,1,0),make_float3(0,0,1)};
  fillRotMatrix(yz_plane,ap1,trot);

  printf("rotz=%f %f %f | %f %f %f | %f %f %f\n",rotz[0].x,rotz[0].y,rotz[0].z,rotz[1].x,rotz[1].y,rotz[1].z,rotz[2].x,rotz[2].y,rotz[2].z);
  */
  
  /*float3 y_axis = make_float3(0,1,0);
  float cos_rotz = 1.0;
  float sin_rotz = 0.0;
  // check for orintation with normal along the x-axis where a_unit = 0
  if (ap1.x != 0.0 or ap1.y != 0.0) {
    float3 a_unit = make_float3(ap1.x,ap1.y,0);
    //a_unit = a_unit/length(a_unit);
    a_unit = normalize(a_unit);
    cos_rotz = dot(a_unit,y_axis) / (length(a_unit)*length(y_axis));
    sin_rotz = 0;
    if (cos_rotz > -1.000001 and cos_rotz < -0.999999) {
      sin_rotz = 0;
    } else {
      sin_rotz = sqrtf(1.0 - cos_rotz*cos_rotz);
    }
    //cos_rotz *= q;
    //sin_rotz *= q;
    }*/
  
  float3 a_unit = normalize(ap1);
  float a_py = sqrt(1 - a_unit.z*a_unit.z);
  float3 a_plane = make_float3(0, a_py, a_unit.z);
  //float3 dis = a_plane - a_unit;
  //printf("distance = %f %f %f\n", dis.x,dis.y,dis.z);
  a_unit = make_float3(a_unit.x,a_unit.y,0);
  a_unit = normalize(a_unit);
  a_plane = make_float3(0,1,0);
  //printf("a_unit = %f %f %f  a_plane = %f %f %f\n", a_unit.x, a_unit.y, a_unit.z, a_plane.x, a_plane.y, a_plane.z);
  float3 rotz[3] = {make_float3(1,0,0),make_float3(0,1,0),make_float3(0,0,1)};
  //float3 trot[3] = {make_float3(1,0,0),make_float3(0,1,0),make_float3(0,0,1)};
  fillRotMatrix(a_unit,a_plane,rotz); 
  float3 trot[3] = {make_float3(rotz[0].x, rotz[1].x, rotz[2].x), make_float3(rotz[0].y, rotz[1].y, rotz[2].y), make_float3(rotz[0].z, rotz[1].z, rotz[2].z)};
  //fillRotMatrix(a_plane,a_unit,trot);
  //printf("rotz = %f %f %f | %f %f %f | %f %f %f\n", rotz[0].x, rotz[0].y, rotz[0].z, rotz[1].x, rotz[1].y, rotz[1].z, rotz[2].x, rotz[2].y, rotz[2].z);
  //printf("trot = %f %f %f | %f %f %f | %f %f %f\n", trot[0].x, trot[0].y, trot[0].z, trot[1].x, trot[1].y, trot[1].z, trot[2].x, trot[2].y, trot[2].z);
  //printf("cos_rotz = %f  sin_rotz = %f  rotz = %f\n",cos_rotz,sin_rotz,asinf(sin_rotz));
  //float cos_rotz = ap1.x/length(a_unit);
  //float sin_rotz = ap1.y/length(a_unit);
  //printf("r=%f %f %f | %f %f %f | %f %f %f  cos_rotz = %f  sin_rotz = %f\n",r[0].x,r[0].y,r[0].z,r[1].x,r[1].y,r[1].z,r[2].x,r[2].y,r[2].z, a_unit.y,a_unit.z,cos_rotz, sin_rotz);
  //float3 rotz[3] = {make_float3(cos_rotz,sin_rotz,0),make_float3(-sin_rotz,cos_rotz,0),make_float3(0,0,1)};
  //float3 trot[3] = {make_float3(cos_rotz,-sin_rotz,0),make_float3(sin_rotz,cos_rotz,0),make_float3(0,0,1)};
  
  ap1 = rotateCoords(ap1,rotz);
  bp1 = rotateCoords(bp1,rotz);
  cp1 = rotateCoords(cp1,rotz);
  
  //printf("swiveled ap1=%f %f %f  bp1=%f %f %f  cp1=%f %f %f\n",ap1.x,ap1.y,ap1.z,bp1.x,bp1.y,bp1.z,cp1.x,cp1.y,cp1.z);
  
  //float sin_phi = (ap1.z)/length(com1 - ap1);
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
  //printf("sin_phi: %f  cos_phi: %f  sin_psi: %f  cos-psi: %f\n", sin_phi, cos_phi, sin_psi, cos_psi);
  float3 a2 = make_float3(0,ra*cos_phi,ra*sin_phi);
  float3 b2 = make_float3(-rc*cos_psi,-rb*cos_phi-rc*sin_psi*sin_phi,-rb*sin_phi+rc*sin_psi*cos_phi);
  float3 c2 = make_float3(rc*cos_psi,-rb*cos_phi+rc*sin_psi*sin_phi,-rb*sin_phi-rc*sin_psi*cos_phi);
  //printf("a2=%f %f %f  b2=%f %f %f  c2=%f %f %f  %f\n",a2.x,a2.y,a2.z,b2.x,b2.y,b2.z,c2.x,c2.y,c2.z,sin_psi*sin_psi);

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
  /*
  if (sin_theta > -0.0001 and sin_theta < 0.0001) {
    sin_theta = 0.0;
    cos_theta = 1.0;
    }*/
  if(sin_theta*alpha + cos_theta*beta > gamma + 0.001 or sin_theta*alpha + cos_theta*beta < gamma - 0.001) {
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

  //printf("psi=%f phi=%f theta=%f\n", asinf(sin_psi), asinf(sin_phi), asinf(sin_theta));
  //printf("adding = %f %f\n", b2.x*cos_theta-b2.y*sin_theta, b2.x*sin_theta+b2.y*cos_theta);
  float3 a3 = make_float3(a2.x*cos_theta-a2.y*sin_theta,a2.x*sin_theta+a2.y*cos_theta,a2.z);
  float3 b3 = make_float3(b2.x*cos_theta-b2.y*sin_theta,b2.x*sin_theta+b2.y*cos_theta,b2.z);
  float3 c3 = make_float3(c2.x*cos_theta-c2.y*sin_theta,c2.x*sin_theta+c2.y*cos_theta,c2.z);
  //printf("a3=%f %f %f  b=%f %f %f  c=%f %f %f\n",a3.x,a3.y,a3.z,b3.x,b3.y,b3.z,c3.x,c3.y,c3.z);

  // rotate back to original coordinate system
  a3 = rotateCoords(a3, trot);
  b3 = rotateCoords(b3, trot);
  c3 = rotateCoords(c3, trot);
  
  //printf("a3=%f %f %f  b=%f %f %f  c=%f %f %f\n",a3.x,a3.y,a3.z,b3.x,b3.y,b3.z,c3.x,c3.y,c3.z);

  a3 = rotateCoords(a3,tr);
  b3 = rotateCoords(b3,tr);
  c3 = rotateCoords(c3,tr);
  
  //a3 = rotateCoords(a3, trn);
  //b3 = rotateCoords(b3, trn);
  //c3 = rotateCoords(c3, trn);
  //printf("a3=%f %f %f  b=%f %f %f  c=%f %f %f\n",a3.x,a3.y,a3.z,b3.x,b3.y,b3.z,c3.x,c3.y,c3.z);
  // update original values
  
  //xs_0[0] = xs[0];
  //xs_0[1] = xs[1];
  //xs_0[2] = xs[2];
  float3 da = a3 - a0;
  float3 db = b3 - b0;
  float3 dc = c3 - c0;
  //printf("da=%f %f %f  db=%f %f %f  dc=%f %f %f\n",da.x,da.y,da.z,db.x,db.y,db.z,dc.x,dc.y,dc.z);
  
  xs[0] = a3 + com1;
  xs[1] = b3 + com1;
  xs[2] = c3 + com1;
  //printf("a3=%f %f %f  b3=%f %f %f  c3=%f %f %f\n",xs[0].x,xs[0].y,xs[0].z,xs[1].x,xs[1].y,xs[1].z,xs[2].x,xs[2].y,xs[2].z);
  //printf("------------------------------- VEL -------\n");
}

__device__ void settle_vs(float timestep, float3 *vs_0, float3 *dvs_0, float3 *vs, float3 *xs, float3 *xs_0, float *mass, float3 *fs_0, float3 *fs) {
  float dt = timestep;

  // calculate velocities
  float3 v0a = (xs[0] - xs_0[0])/dt;
  float3 v0b = (xs[1] - xs_0[1])/dt;
  float3 v0c = (xs[2] - xs_0[2])/dt;
  
  //printf("xs a = %f %f %f  xs b = %f %f %f  xs c = %f %f %f  dt = %f  mass = %f %f %f\n", xs[0].x, xs[0].y, xs[0].z, xs[1].x, xs[1].y, xs[1].z, xs[2].x, xs[2].y, xs[2].z, timestep, mass[0], mass[1], mass[2]);
  float ma = mass[0];
  float mb = mass[1];
  float mc = mass[2];

  float3 v0a_ = vs_0[0] + (0.5*dt*fs_0[0]) + dt*dvs_0[0] + (0.5*dt*fs[0]);
  float3 v0b_ = vs_0[1] + 0.5*dt*fs_0[1] + dt*dvs_0[1] + 0.5*dt*fs[1];
  float3 v0c_ = vs_0[2] + 0.5*dt*fs_0[2] + dt*dvs_0[2] + 0.5*dt*fs[2];
  //printf("v0a.x = %f  1/2*dt*f0a.x = %f  dt*dv0a.x = %f  1/2*dt*fa.x = %f\n", vs_0[0].x, (0.5*dt*fs_0[0]).x, (dt*dvs_0[0]).x, (0.5*dt*fs[0]).x);
  //printf("f=%f %f %f  dvs=%f %f %f v0=%f %f %f v=%f %f %f\n", fs_0[0].x, fs_0[0].y, fs_0[0].z, dvs_0[0].x, dvs_0[0].y, dvs_0[0].z, v0a_.x, v0b_.y, v0c_.z);
  //printf("v0a_ = %f %f %f  v0b_ = %f %f %f  v0c_ = %f %f %f\n", v0a_.x, v0a_.y, v0a_.z, v0b_.x, v0b_.y, v0b_.z, v0c_.x, v0c_.y, v0c_.z);
  //printf("v0a = %f %f %f  v0b = %f %f %f  v0c = %f %f %f\n", v0a.x, v0a.y, v0a.z, v0b.x, v0b.y, v0b.z, v0c.x, v0c.y, v0c.z);
  /*
  if (length(vs_0[0]) > 1) {
    assert(length(v0a) < 2*length(vs_0[0]));
  }
  if (length(vs_0[1]) > 1){
    assert(length(v0b) < 2*length(vs_0[1]));
  }
  if (length(vs_0[2]) > 1){
    assert(length(v0c) < 2*length(vs_0[2])); 
    }*/

  //v0a = v0a_;
  //v0b = v0b_;
  //v0c = v0c_;
  //printf("m=%f %f %f\n",ma,mb,mc);
  float3 a3 = xs[0];
  float3 b3 = xs[1];
  float3 c3 = xs[2];
  float3 v0ab = v0b - v0a;
  float3 v0bc = v0c - v0b;
  float3 v0ca = v0a - v0c;
  //printf("v0ab=%f %f %f\n",v0ab.x,v0ab.y,v0ab.z);
  //printf("v0bc=%f %f %f\n",v0bc.x,v0bc.y,v0bc.z);
  //printf("v0ca=%f %f %f\n",v0ca.x,v0ca.y,v0ca.z);
  // direction vectors
  float3 eab = b3 - a3;
  float3 ebc = c3 - b3;
  float3 eca = a3 - c3;
  //printf("eab=%f %f %f ",eab.x,eab.y,eab.z);                                                                                                                                                          
  //printf("ebc=%f %f %f ",ebc.x,ebc.y,ebc.z);                                                                                                                                                            
  //printf("eca=%f %f %f\n",eca.x,eca.y,eca.z);
  eab = normalize(eab);
  ebc = normalize(ebc);
  eca = normalize(eca);
  //printf("eab=%f %f %f\n ",eab.x,eab.y,eab.z);
  //printf("ebc=%f %f %f\n ",ebc.x,ebc.y,ebc.z);
  //printf("eca=%f %f %f\n",eca.x,eca.y,eca.z);
  //float sideBC = length(b3-c3);
  //float sideCA = length(c3-a3);
  //float sideAB = length(a3-b3);
  //printf("sides=%f %f %f\n",sideAB,sideBC,sideCA);
  //float cosA = (sideBC*sideBC - sideAB*sideAB - sideCA*sideCA)/(-2*sideCA*sideAB);
  //float cosB = (sideCA*sideCA - sideBC*sideBC - sideAB*sideAB)/(-2*sideAB*sideBC);
  //float cosC = (sideAB*sideAB - sideBC*sideBC - sideCA*sideCA)/(-2*sideBC*sideCA);
  
  float cosA = dot(-eab,eca);
  float cosB = dot(-ebc,eab);
  float cosC = dot(-eca,ebc);
  
  //float cosA = powf(sideBC,2) / (powf(sideCA,2) + powf(sideAB,2) - 2*sideAB*sideCA);
  //float cosB = powf(sideCA,2) / (powf(sideBC,2) + powf(sideAB,2) - 2*sideBC*sideAB);
  //float cosC = powf(sideAB,2) / (powf(sideBC,2) + powf(sideCA,2) - 2*sideBC*sideCA);
  //cosA = fabs(cosA);
  //cosB = fabs(cosB);
  //cosC = fabs(cosC);

  //printf("cos=%f %f %f\n",cosA,cosB,cosC);
  float d = 2*(ma+mb)*(ma+mb) + 2*ma*mb*cosA*cosB*cosC;
  d -= 2*mb*mb*cosA*cosA + ma*(ma+mb)*(cosB*cosB + cosC*cosC);
  d *= dt/(2*mb);
  //printf("d=%f\n",d);
  float vab = dot(eab,v0ab);
  float vbc = dot(ebc,v0bc);
  float vca = dot(eca,v0ca);
  //printf("v0ab = %f  v0bc = %f  v0ca = %f\n",vab,vbc,vca);
  //printf("dot vab = %f  vbc = %f  vca = %f\n",vab,vbc,vca);
  float tab = vab * (2*(ma + mb) - ma*cosC*cosC);
  tab += vbc * (mb*cosC*cosA - (ma + mb)*cosB);
  tab += vca * (ma*cosB*cosC - 2*mb*cosA);
  tab *= ma/d;
  float tbc = vbc * ((ma+mb)*(ma+mb) - mb*mb*cosA*cosA);
  //printf("tbc check 1 = %f\n", mb*mb*cosA*cosA);
  tbc += vca*ma * (mb*cosA*cosB - (ma + mb)*cosC);
  //printf("tbc check 2 = %f\n", tbc);
  tbc += vab*ma * (mb*cosC*cosA - (ma + mb)*cosB);
  //printf("tbc check 3 = %f\n", tbc);
  tbc /= d;
  float tca = vca * (2*(ma + mb) - ma*cosB*cosB);
  tca += vab * (ma*cosB*cosC - 2*mb*cosA);
  tca += vbc * (mb*cosA*cosB - (ma + mb)*cosC);
  tca *= ma/d;
  //printf("tab = %f  tbc = %f   tca = %f  d = %f\n",tab,tbc,tca,d);
  //printf("(tab*eab - tca*eca) = %f  (tbc*ebc - tab*eab) = %f   (tca*eca - tbc*ebc) = %f  d = %f\n",(tab*eab - tca*eca).x,(tbc*ebc - tab*eab).x,(tca*eca - tbc*ebc).x,d);
  //float check1 = 2*mc*ma*vca;
  //float check2 = dt*mc*cosA*tab + dt*ma*cosC*tbc + dt*(mc+ma)*tca;
  //printf("check = %f\n",check1 - check2);
  //printf("tab=%f tbc=%f tca=%f\n",tab,tbc,tca);
  float3 dva = (dt/(2*ma))*(tab*eab - tca*eca);
  float3 dvb = (dt/(2*mb))*(tbc*ebc - tab*eab);
  float3 dvc = (dt/(2*mc))*(tca*eca - tbc*ebc);


  // ------------
  /*
  dva = (xs[0] - xs_0[0])/dt;
  dvb = (xs[1] - xs_0[1])/dt;
  dvc = (xs[2] - xs_0[2])/dt;
  */
  // ------------
  //printf("da=%f %f %f db=%f %f %f dc=%f %f %f\n", dva.x*dt, dva.y*dt, dva.z*dt, dvb.x*dt, dvb.y*dt, dvb.z*dt, dvc.x*dt, dvc.y*dt, dvc.z*dt);
  //printf("dva=%f %f %f dvb=%f %f %f dvc=%f %f %f\n", dva.x, dva.y, dva.z, dvb.x, dvb.y, dvb.z, dvc.x, dvc.y, dvc.z);
  dvs_0[0] = dva;
  dvs_0[1] = dvb;
  dvs_0[2] = dvc;
  
  //printf("v'a=%f %f %f vb=%f %f %f vc=%f %f %f\n",va.x,va.y,va.z,vb.x,vb.y,vb.z,vc.x,vc.y,vc.z);
  v0a += dva;
  v0b += dvb;
  v0c += dvc;
  
  float3 va = v0a;
  float3 vb = v0b;
  float3 vc = v0c;
  //printf("va=%f %f %f vb=%f %f %f vc=%f %f %f\n",va.x,va.y,va.z,vb.x,vb.y,vb.z,vc.x,vc.y,vc.z);
  //float orth = dot(b3 - a3, vb - va);
  //printf("r * v = %f\n", orth);

  vs[0] = va;
  vs[1] = vb;
  vs[2] = vc;
}



__global__ void compute_SETTLE(int4 *waterIds, float4 *xs, float4 *xs_0, float4 *vs, float4 *vs_0, float4 *dvs_0, float4 *fs, float4 *fs_0, float4 *comOld, float4 *fix_len, int nMols, float dt, int *idToIdxs, BoundsGPU bounds) {
  int idx = GETIDX();
  if (idx < nMols) {
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
    float3 fix_len_mol = make_float3(fix_len[idx]);
    int3 waterId_mol = make_int3(waterIds[idx]);
    ids[0] = waterId_mol.x;
    ids[1] = waterId_mol.y;
    ids[2] = waterId_mol.z;
    for (int i = 0; i < 3; i++) {
      int myIdx = idToIdxs[ids[i]];
      idxs[i] = myIdx;
      xs_0_mol[i] = make_float3(xs_0[idx*3+i]);
      float4 xWhole = xs[myIdx];
      xs_mol[i] = make_float3(xWhole);
      vs_0_mol[i] = make_float3(vs_0[idx*3+i]);
      float4 vWhole = vs[myIdx];
      vs_mol[i] = make_float3(vWhole);
      dvs_0_mol[i] = make_float3(dvs_0[idx*3+i]);
      //printf("dvs=%f %f %f\n", dvs_0_mol[i].x, dvs_0_mol[i].y, dvs_0_mol[i].z);
      mass[i] = 1.0f / vWhole.w;
      fs_0_mol[i] = make_float3(fs_0[idx*3+i]);
      float4 fWhole = fs[myIdx];
      fs_mol[i] = make_float3(fWhole);
      //fix_len_mol[i] = make_float3(fix_len[idx*3+i]);
    }
    for (int i=1; i<3; i++) {
      //printf("xs = %f %f %f\n", xs_mol[i].x, xs_mol[i].y, xs_mol[i].z);
      float3 delta = xs_mol[i] - xs_mol[0];
      delta = bounds.minImage(delta);
      xs_mol[i] = xs_mol[0] + delta;
      //printf("xd = %f %f %f\n", xs_mol[i].x, xs_mol[i].y, xs_mol[i].z);
    }
    for (int i=0; i<3; i++) {
      float3 delta = xs_0_mol[i] - xs_mol[0];
      delta = bounds.minImage(delta);
      xs_0_mol[i] = xs_mol[0] + delta;
    }
    //printf("---------------------------------------------\n");
    //printf("mass = %f %f %f\n", mass[0], mass[1], mass[2]);
    float3 comNew = positionsToCOM(xs_mol, mass, comOld[idx].w);
    float3 delta = make_float3(comOld[idx]) - comNew;
    delta = bounds.minImage(delta);
    float3 comOldWrap = comNew + delta;
    //printf("comNew = %f %f %f  delta = %f %f %f  comOldWrap = %f %f %f\n", comNew.x, comNew.y, comNew.z, delta.x, delta.y, delta.z, comOldWrap.x*1000, comOldWrap.y*1000, comOldWrap.z*1000);
    //printf("xs_x=%f xs_y=%f xs_z=%f\n", xs_mol[0].x, xs_mol[0].y, xs_mol[0].z);
    settle_xs(dt, comOldWrap, comNew, xs_0_mol, xs_mol, fix_len_mol);
    /*printf("Settle  positions: xs_x=%f xs_y=%f xs_z=%f\n", xs_mol[0].x, xs_mol[0].y, xs_mol[0].z);
    printf("                   xs_x=%f xs_y=%f xs_z=%f\n", xs_mol[1].x, xs_mol[1].y, xs_mol[1].z);
    printf("                   xs_x=%f xs_y=%f xs_z=%f\n", xs_mol[2].x, xs_mol[2].y, xs_mol[2].z);*/
    //printf("vs_mol = %f %f %f\n", vs_mol[0].x*100, vs_mol[0].y*100, vs_mol[0].z*100);
    settle_vs(dt, vs_0_mol, dvs_0_mol, vs_mol, xs_mol, xs_0_mol, mass, fs_0_mol, fs_mol);
    /*printf("Settle velocities: vs_x=%f vs_y=%f vs_z=%f\n", vs_mol[0].x, vs_mol[0].y, vs_mol[0].z);
    printf("                   vs_x=%f vs_y=%f vs_z=%f\n", vs_mol[1].x, vs_mol[1].y, vs_mol[1].z);
    printf("                   vs_x=%f vs_y=%f vs_z=%f\n", vs_mol[2].x, vs_mol[2].y, vs_mol[2].z);*/
    for (int i=0; i<3; i++) {
      xs[idxs[i]] = make_float4(xs_mol[i]);
    }
    for (int i=0; i<3; i++) {
      vs[idxs[i]] = make_float4(vs_mol[i]);
      vs[idxs[i]].w = 1.0f/mass[i];
      dvs_0[idx*3+i] = make_float4(dvs_0_mol[i]);
    }
  }
}


void FixRigid::handleBoundsChange() {

    if (TIP4P) {
    
        int nMols = waterIdsGPU.size();
        GPUData &gpd = state->gpd;
        int activeIdx = gpd.activeIdx();
        BoundsGPU &bounds = state->boundsGPU;

        // get a few pieces of data as required
        // -- all we're doing here is setting the position of the M-Site prior to computing the forces
        //    within the simulation.  Otherwise, the M-Site will likely be far away from where it should be, 
        //    relative to the moleule.  We do not solve the constraints on the rigid body at this time.
        // we need to reset the position of the M-Site prior to calculating the forces
        setMSite<<<NBLOCK(nMols), PERBLOCK>>>(waterIdsGPU.data(), gpd.idToIdxs.d_data.data(), gpd.xs(activeIdx), nMols, bounds);

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
    
    //printf("position Oxygen: %f %f %f\n", r_i.x, r_i.y, r_i.z);
    //printf("position H1    : %f %f %f\n", r_j.x, r_j.y, r_j.z);
    //printf("position H2    : %f %f %f\n", r_k.x, r_k.y, r_k.z);
    // get the minimum image vectors r_ij, r_ikz
    float3 r_ij = bounds.minImage(r_j - r_i);
    float3 r_ik = bounds.minImage(r_k - r_i);

    // denominator of expression 3, written using the minimum image
    float denominator = length( ( r_ij + r_ik) );

    //printf("denominator value: %f\n", denominator);
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
    if (TIP3P && TIP4P) {
        mdError("An attempt was made to use both TIP3P and TIP4P in a simulation");
    }

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
    compute_COM<<<NBLOCK(n), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), gpd.vs(activeIdx), 
                                         gpd.idToIdxs.d_data.data(), n, com.data(), bounds);
    set_fixed_sides<<<NBLOCK(n), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), com.data(), 
                                             fix_len.data(), n, gpd.idToIdxs.d_data.data());
    set_init_vel_correction<<<NBLOCK(n), PERBLOCK>>>(waterIdsGPU.data(), dvs_0.data(), n);

  return true;
}

bool FixRigid::stepInit() {
    int nMols = waterIdsGPU.size();
    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();
    BoundsGPU &bounds = state->boundsGPU;
    float dtf = 0.5f * state->dt * state->units.ftm_to_v;
    
    //printf("after FixRigid::stepInit.distributeMSite<<>>\n");
    compute_COM<<<NBLOCK(nMols), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                           gpd.vs(activeIdx), gpd.idToIdxs.d_data.data(), 
                                           nMols, com.data(), bounds);
    compute_prev_val<<<NBLOCK(nMols), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), xs_0.data(), 
                                                gpd.vs(activeIdx), vs_0.data(), gpd.fs(activeIdx), 
                                                fs_0.data(), nMols, gpd.idToIdxs.d_data.data());

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

    //printf("got to stepFinal in FixRigid!\n");
    //float4 cpu_xs[nMols*3];
    //gpd.xs.dataToHost(activeIdx);
    //std::cout << "before settle: " << cpu_xs[0] << "\n";

    // first, unconstrained velocity update continues: distribute the force from the M-site
    //        and integrate the velocities accordingly.  Update the forces as well.
    // Next,  do compute_SETTLE as usual on the (as-yet) unconstrained positions & velocities

    // from IntegratorVerlet
    if (TIP4P) {
        float dtf = 0.5f * state->dt * state->units.ftm_to_v;
        distributeMSite<<<NBLOCK(nMols), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                                     gpd.vs(activeIdx),  gpd.fs(activeIdx),
                                                     nMols, gamma, dtf, gpd.idToIdxs.d_data.data(), bounds);


    }

    //printf("Distributed the MSite forces!\n");

    compute_SETTLE<<<NBLOCK(nMols), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                                xs_0.data(), gpd.vs(activeIdx), vs_0.data(), 
                                                dvs_0.data(), gpd.fs(activeIdx), fs_0.data(), 
                                                com.data(), fix_len.data(), nMols, dt, 
                                                gpd.idToIdxs.d_data.data(), bounds);

 
    //printf("did compute_SETTLE!\n");
    // finally, reset the position of the M-site to be consistent with that of the new, constrained water molecule
    if (TIP4P) {
        setMSite<<<NBLOCK(nMols), PERBLOCK>>>(waterIdsGPU.data(), gpd.idToIdxs.d_data.data(), gpd.xs(activeIdx), nMols, bounds);
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
     );
    
}



