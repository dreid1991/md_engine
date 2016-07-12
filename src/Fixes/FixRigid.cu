#include "FixRigid.h"

#include "State.h"
#include "boost_for_export.h"
#include "cutils_math.h"
#include <math.h>
using namespace std;
namespace py = boost::python;
const string rigidType = "Rigid";

FixRigid::FixRigid(boost::shared_ptr<State> state_, string handle_, string groupHandle_) : Fix(state_, handle_, groupHandle_, rigidType, true, false, false, 1) {

}

__device__ inline float3 positionsToCOM(float3 *pos, float *mass) {
    return (mass[0]*pos[0] + mass[1]*pos[1] + mass[2]*pos[2])/(mass[0] + mass[1] + mass[2]);
}

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
        com[idx] = make_float4(positionsToCOM(pos, mass));
        //com[idx] = (mass[0]*pos[0] + mass[1]*pos[1] + mass[2]*pos[2])/(mass[0] + mass[1] + mass[2]);
    }
}

__global__ void compute_prev_val(int4 *waterIds, float4 *xs, float4 *xs_0, float4 *vs, float4 *vs_0, int nMols, int *idToIdxs) {
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
        }
    }
}




/* ------- Based off of the SETTLE Algorithm outlined in ------------
------ Miyamoto et al., J Comput Chem. 13 (8): 952–962 (1992). ------ */

__device__ void settle_xs(float timestep, float3 com, float3 *xs_0, float3 *xs) {

  float3 a0 = xs_0[0];
  float3 b0 = xs_0[1];
  float3 c0 = xs_0[2];

  float3 a1 = xs[0];
  float3 b1 = xs[1];
  float3 c1 = xs[2];

  float ra = length(com - a0);
  float rb = length(com - b0);
  float rc = length(com - c0);
  float3 ap0 = make_float3(0,ra,0);
  float3 bp0 = make_float3(-rc,-rb,0);
  float3 cp0 = make_float3(rc,-rb,0);

  float3 ap1 = a1 - com;
  float3 bp1 = b1 - com;
  float3 cp1 = c1 - com;
  float sin_phi = ap1.z/ra;
  float cos_phi = sqrtf(1-(sin_phi*sin_phi));
  float sin_psi = (bp1.z-cp1.z)/(2*rc*cos_phi);
  float cos_psi = sqrtf(1-(sin_psi*sin_psi));
  float3 a2 = make_float3(0,rc*cos_phi,rc*sin_phi);
  float3 b2 = make_float3(-rc*cos_psi,-rb*cos_phi-rc*sin_psi*sin_phi,-rb*sin_phi+rc*sin_psi*cos_phi);
  float3 c2 = make_float3(rc*cos_psi,-rb*cos_phi+rc*sin_psi*sin_phi,-rb*sin_phi+rc*sin_psi*cos_phi);
  
  float alpha = b2.x*(bp0.x - cp0.x) + (bp0.y - ap0.y)*b2.y + (cp0.y - ap0.y)*c2.y;
  float beta = b2.x*(cp0.y - bp0.y) + (bp0.x - ap0.x)*b2.y + (cp0.x - ap0.x)*c2.y;
  float gamma = (bp0.x - ap0.x)*bp1.y - bp1.x*(bp0.y - ap0.y) + (cp0.x - ap0.x)*cp1.y - cp1.x*(cp0.y - ap0.y);
  float sin_theta = (alpha*gamma - beta*sqrt(alpha*alpha + beta*beta - gamma*gamma))/(alpha*alpha + beta*beta);
  float cos_theta = sqrtf(1 - sin_theta*sin_theta);
  float3 a3 = make_float3(a2.x*cos_theta-a2.y*sin_theta,a2.x*sin_theta+a2.y*cos_theta,a2.z);
  float3 b3 = make_float3(b2.x*cos_theta-b2.y*sin_theta,b2.x*sin_theta+b2.y*cos_theta,b2.z);
  float3 c3 = make_float3(c2.x*cos_theta-c2.y*sin_theta,c2.x*sin_theta+c2.y*cos_theta,c2.z);
  // update original values
  xs[0] = a3 + com;
  xs[1] = b3 + com;
  xs[2] = c3 + com;
 }

__device__ void settle_vs(float timestep, float3 *vs_0, float3 *vs, float3 *xs, float *mass) {
  // calculate velocities
  float3 v0a = vs_0[0];
  float3 v0b = vs_0[1];
  float3 v0c = vs_0[2];

  float dt = timestep;
  float ma = mass[0];
  float mb = mass[1];
  float mc = mass[2];
  float3 a3 = xs[0];
  float3 b3 = xs[1];
  float3 c3 = xs[2];
  float3 v0ab = v0a + v0b;
  float3 v0bc = v0b + v0c;
  float3 v0ca = v0c + v0a;
  int eab = 1,ebc = 1,eca = 1;


  float sideBC = length(b3-c3);
  float sideCA = length(c3-a3);
  float sideAB = length(a3-b3);
  float cosA = powf(sideBC,2) / (powf(sideCA,2) + powf(sideAB,2) - 2*sideAB*sideCA);
  float cosB = powf(sideCA,2) / (powf(sideBC,2) + powf(sideAB,2) - 2*sideBC*sideAB);
  float cosC = powf(sideAB,2) / (powf(sideBC,2) + powf(sideCA,2) - 2*sideBC*sideCA);
  float d = 2*powf(ma+mb,2) + 2*ma*mb*cosA*cosB*cosC;
  d -= 2*mb*mb*cosA*cosA - ma*(ma+mb)*(cosB*cosB + cosC*cosC);
  d *= dt/(2*mb);
  float3 tab = eab*v0ab * (2*(ma + mb) - ma*cosB*cosB);
  tab += ebc*v0bc * (mb*cosC*cosA - (ma + mb)*cosB);
  tab += eca*v0ca * (ma*cosB*cosC - 2*mb*cosA);
  tab *= ma/d;
  float3 tbc = ebc*v0bc * (powf(ma+mb,2) - powf(mb*cosA,2));
  tbc += eca*v0ca*ma * (mb*cosA*cosB - (ma + mb)*cosC);
  tbc += eab*v0ab*ma * (mb*cosC*cosA - (ma + mb)*cosB);
  tbc /= d;
  float3 tca = eca*v0ca * (2*(ma + mb) - ma*cosB*cosB);
  tca += eab*v0ab * (ma*cosB*cosC - 2*mb*cosA);
  tca += ebc*v0bc * (mb*cosA*cosB - (ma + mb)*cosC);
  tca *= ma/d;
  float3 va = v0a + dt/(2*ma)*(tab - tca);
  float3 vb = v0b + dt/(2*mb)*(tbc - tab);
  float3 vc = v0c + dt/(2*mc)*(tca - tbc);
  vs[0] = va;
  vs[1] = vb;
  vs[2] = vc;
 }



__global__ void compute_SETTLE(int4 *waterIds, float4 *xs, float4 *xs_0, float4 *vs, float4 *vs_0, float4 *comOld, int nMols, float dt, int *idToIdxs, BoundsGPU bounds) {
  int idx = GETIDX();
  if (idx < nMols) {
    int ids[3];
    int idxs[3];
    float3 xs_0_mol[3];
    float3 xs_mol[3];
    float3 vs_0_mol[3];
    float3 vs_mol[3];
    float mass[3];
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
      mass[i] = 1.0f / vWhole.w;
    }
    for (int i=1; i<3; i++) {
        float3 delta = xs_mol[i] - xs_mol[0];
        delta = bounds.minImage(delta);
        xs_mol[i] = xs_mol[i] + delta;
    }
    for (int i=0; i<3; i++) {
        float3 delta = xs_0_mol[i] - xs_mol[0];
        delta = bounds.minImage(delta);
        xs_0_mol[i] = xs_mol[0] + delta;
    }
    float3 comNew = positionsToCOM(xs_mol, mass);
    float3 delta = make_float3(comOld[idx]) - comNew;
    delta = bounds.minImage(delta);
    float3 comOldWrap = comNew + delta;
    settle_xs(dt, comOldWrap, xs_0_mol, xs_mol);
    settle_vs(dt, vs_0_mol, vs_mol, xs_mol, mass);
    for (int i=0; i<3; i++) {
        xs[idxs[i]] = make_float4(xs_mol[i]);
    }
    for (int i=0; i<3; i++) {
        vs[idxs[i]] = make_float4(vs_mol[i]);
    }
  }
}

void FixRigid::createRigid(int id_a, int id_b, int id_c) {
  int4 waterMol = make_int4(id_a,id_b,id_c,0);
  waterIds.push_back(waterMol);
}

bool FixRigid::prepareForRun() {
  int n = waterIds.size();
  waterIdsGPU = GPUArrayDeviceGlobal<int4>(n);
  waterIdsGPU.set(waterIds.data());

  xs_0 = GPUArrayDeviceGlobal<float4>(3*n);
  vs_0 = GPUArrayDeviceGlobal<float4>(3*n);
  com = GPUArrayDeviceGlobal<float4>(n);
  return true;
}

bool FixRigid::stepInit() {
    int nMols = waterIdsGPU.size();
    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();
    BoundsGPU &bounds = state->boundsGPU;
    compute_COM<<<NBLOCK(nMols), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), gpd.vs(activeIdx), gpd.idToIdxs.d_data.data(), nMols, com.data(), bounds);
    compute_prev_val<<<NBLOCK(nMols), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), xs_0.data(), gpd.vs(activeIdx), vs_0.data(), nMols, gpd.idToIdxs.d_data.data());
    return true;
}

bool FixRigid::stepFinal() {
    int nMols = waterIdsGPU.size();
    int dt = state->dt;
    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();
    BoundsGPU &bounds = state->boundsGPU;
    compute_SETTLE<<<NBLOCK(nMols), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), xs_0.data(), gpd.vs(activeIdx), vs_0.data(), com.data(), nMols, dt, gpd.idToIdxs.d_data.data(), bounds);
    return true;
}



/*
 void export_FixRigid() {
   boost::python::class_<FixRigid, SHARED(FixRigid), boost::python::bases<Fix> > (
										  "FixRigid",
										  boost::python::init<SHARED(State), string, string (
																     boost::python::args("state", "handle", "groupHandle")
																     )
										  );
										  }*/


















