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

inline __host__ __device__ float3 rotateCoords(float3 vector, float3 matrix[]) {
  return make_float3(dot(matrix[0],vector),dot(matrix[1],vector),dot(matrix[2],vector));
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

__device__ void settle_xs(float timestep, float3 com, float3 com1, float3 *xs_0, float3 *xs) {
  printf("COM = %f %f %f\n", com.x, com.y, com.x);
  float3 a0 = xs_0[0];
  float3 b0 = xs_0[1];
  float3 c0 = xs_0[2];
  printf("a0=%f %f %f\n",a0.x,a0.y,a0.z);

  float3 a1 = xs[0];
  float3 b1 = xs[1];
  float3 c1 = xs[2];
  printf("a1=%f %f %f\n",a1.x,a1.y,a1.z);
  printf("b1=%f %f %f\n",b1.x,b1.y,b1.z);
  printf("c1=%f %f %f\n",c1.x,c1.y,c1.z);

  
  float ra = fabs(length(com - a0));
  float rc = fabs(length(c0 - b0)/2.0f);
  float rb = sqrtf(powf(length(a0-c0),2) - powf(rc,2)) - ra;
  float3 ap0 = make_float3(0,ra,0);
  float3 bp0 = make_float3(-rc,-rb,0);
  float3 cp0 = make_float3(rc,-rb,0);
  printf("ap0=%f %f %f  rc=%f\n",ap0.x,ap0.y,ap0.z,rc);
  printf("bp0=%f %f %f\n",bp0.x,bp0.y,bp0.z);
  printf("cp0=%f %f %f\n",cp0.x,cp0.y,cp0.z);
  
  //float3 ap0 = a0 - com;
  //float3 bp0 = b0 - com;
  //float3 cp0 = c0 - com;

  // move ∆'1 to the origin
  float3 ap1 = a1 - com1;
  float3 bp1 = b1 - com1;
  float3 cp1 = c1 - com1;
  printf("com1=%f %f %f\n",com1.x,com1.y,com1.z);

  float3 a_unit = make_float3(ap1.x,ap1.y,0);
  a_unit /= length(a_unit);
  float3 y_axis = make_float3(0,1,0);
  float cos_rotz = dot(a_unit,y_axis) / (length(a_unit)*length(y_axis));
  float sin_rotz = 0;
  if (cos_rotz > -1.000001 and cos_rotz < -0.999999) {
    sin_rotz = 0;
  } else {
    sin_rotz = sqrtf(1.0 - cos_rotz*cos_rotz);
  }
  //float cos_rotz = ap1.x/length(a_unit);
  //float sin_rotz = ap1.y/length(a_unit);
  printf("cos_rotz = %f  sin_rotz = %f\n", cos_rotz, sin_rotz);
  float3 rotz[3] = {make_float3(cos_rotz,sin_rotz,0),make_float3(-sin_rotz,cos_rotz,0),make_float3(0,0,1)};
  float3 trot[3] = {make_float3(cos_rotz,-sin_rotz,0),make_float3(sin_rotz,cos_rotz,0),make_float3(0,0,1)};
  ap1 = rotateCoords(ap1,rotz);
  bp1 = rotateCoords(bp1,rotz);
  cp1 = rotateCoords(cp1,rotz);
  float3 test = make_float3(0,1,0);
  test = rotateCoords(test, rotz);
  //printf("TEST: %f %f %f\n", test.x,test.y,test.z);
  printf("ap1=%f %f %f\n",ap1.x,ap1.y,ap1.z);
  printf("bp1=%f %f %f\n",bp1.x,bp1.y,bp1.z);
  printf("cp1=%f %f %f\n",cp1.x,cp1.y,cp1.z);
  

  float sin_phi = (bp1.z + cp1.z)/(2*rb);
  float cos_phi = 0;
  if (sin_phi > -1.00000 and sin_phi <= -0.999999) {
    cos_phi = 0;
  } else {
    cos_phi = sqrtf(1-(sin_phi*sin_phi));
  }
  float sin_psi = (bp1.z-cp1.z)/(2*rc*cos_phi);
  float cos_psi = 0;
  if (sin_psi > -1.00000 and sin_psi <= -0.999999) {
    cos_psi = 0;
  } else {
    cos_psi = sqrtf(1-(sin_psi*sin_psi));
  }
  float3 a2 = make_float3(0,ra*cos_phi,ra*sin_phi);
  float3 b2 = make_float3(-rc*cos_psi,-rb*cos_phi-rc*sin_psi*sin_phi,-rb*sin_phi+rc*sin_psi*cos_phi);
  float3 c2 = make_float3(rc*cos_psi,-rb*cos_phi+rc*sin_psi*sin_phi,-rb*sin_phi-rc*sin_psi*cos_phi);
  printf("num: %f  dem: %f  sin_psi: %f  sin_psi^2: %f\n", (bp1.z-cp1.z), (2*rc*cos_phi), sin_psi, sin_psi*sin_psi);
  printf("a2=%f %f %f  b2=%f %f %f  c2=%f %f %f  %f\n",a2.x,a2.y,a2.z,b2.x,b2.y,b2.z,c2.x,c2.y,c2.z,sin_psi*sin_psi);

  float alpha = b2.x*(bp0.x - cp0.x) + (bp0.y - ap0.y)*b2.y + (cp0.y - ap0.y)*c2.y;
  float beta = b2.x*(cp0.y - bp0.y) + (bp0.x - ap0.x)*b2.y + (cp0.x - ap0.x)*c2.y;
  float gamma = (bp0.x - ap0.x)*bp1.y - bp1.x*(bp0.y - ap0.y) + (cp0.x - ap0.x)*cp1.y - cp1.x*(cp0.y - ap0.y);
  float sin_theta = (alpha*gamma - beta*sqrtf(alpha*alpha + beta*beta - gamma*gamma))/(alpha*alpha + beta*beta);
  printf("sin_theta = %f  alpha = %f  beta = %f  gamma = %f\n", sin_theta, alpha, beta, gamma);
  float cos_theta = 0;
  if (sin_theta > -1.00000 and sin_theta <= -0.999999) {
    cos_theta = 0;
  } else {
    cos_theta = sqrtf(1 - sin_theta*sin_theta);
  }
  if(sin_theta*alpha + cos_theta*beta > gamma + 0.001 or sin_theta*alpha + cos_theta*beta < gamma - 0.001) {
    float sin_theta = (alpha*gamma + beta*sqrtf(alpha*alpha + beta*beta - gamma*gamma))/(alpha*alpha + beta*beta);
    if (sin_theta > -1.00000 and sin_theta <= -0.999999) {
      cos_theta =0;
    } else {
      cos_theta = sqrtf(1 - sin_theta*sin_theta);
    }
  }
  printf("psi=%f phi=%f theta=%f\n", asinf(sin_psi), asinf(sin_phi), asinf(sin_theta));
  float3 a3 = make_float3(a2.x*cos_theta-a2.y*sin_theta,a2.x*sin_theta+a2.y*cos_theta,a2.z);
  float3 b3 = make_float3(b2.x*cos_theta-b2.y*sin_theta,b2.x*sin_theta+b2.y*cos_theta,b2.z);
  float3 c3 = make_float3(c2.x*cos_theta-c2.y*sin_theta,c2.x*sin_theta+c2.y*cos_theta,c2.z);
  printf("a3=%f %f %f  b=%f %f %f  c=%f %f %f\n",a3.x,a3.y,a3.z,b3.x,b3.y,b3.z,c3.x,c3.y,c3.z);

  // rotate back to original coordinate system
  a3 = rotateCoords(a3, trot);
  b3 = rotateCoords(b3, trot);
  c3 = rotateCoords(c3, trot);
  // update original values
  xs[0] = a3 + com1;
  xs[1] = b3 + com1;
  xs[2] = c3 + com1;
  printf("xs_a=%f %f %f\n",xs[0].x,xs[0].y,xs[0].z);
}

__device__ void settle_vs(float timestep, float3 *vs_0, float3 *vs, float3 *xs, float *mass) {
  // calculate velocities
  float3 v0a = vs_0[0];
  float3 v0b = vs_0[1];
  float3 v0c = vs_0[2];
  printf("v0a=%f %f %f\n",v0a.x,v0a.y,v0a.z);

  float dt = timestep;
  float ma = mass[0];
  float mb = mass[1];
  float mc = mass[2];
  printf("m=%f %f %f\n",ma,mb,mc);
  float3 a3 = xs[0];
  float3 b3 = xs[1];
  float3 c3 = xs[2];
  float3 v0ab = v0b - v0a;
  float3 v0bc = v0c - v0b;
  float3 v0ca = v0a - v0c;
  printf("v0ab=%f %f %f\n",v0ab.x,v0ab.y,v0ab.z);
  printf("v0bc=%f %f %f\n",v0bc.x,v0bc.y,v0bc.z);
  printf("v0ca=%f %f %f\n",v0ca.x,v0ca.y,v0ca.z);
  // direction vectors
  float3 eab = b3 - a3;
  float3 ebc = c3 - b3;
  float3 eca = a3 - c3;
  eab = eab/length(eab);
  ebc = ebc/length(ebc);
  eca = eca/length(eca);
  printf("eab=%f %f %f ",eab.x,eab.y,eab.z);
  printf("ebc=%f %f %f ",ebc.x,ebc.y,ebc.z);
  printf("eca=%f %f %f\n",eca.x,eca.y,eca.z);
  float sideBC = length(b3-c3);
  float sideCA = length(c3-a3);
  float sideAB = length(a3-b3);
  printf("sides=%f %f %f\n",sideAB,sideBC,sideCA);
  float cosA = (powf(sideBC,2) - powf(sideAB,2) - powf(sideCA,2))/(-2*sideCA*sideAB);
  float cosB = (powf(sideCA,2) - powf(sideBC,2) - powf(sideAB,2))/(-2*sideAB*sideBC);
  float cosC = (powf(sideAB,2) - powf(sideBC,2) - powf(sideCA,2))/(-2*sideBC*sideCA);
  //float cosA = powf(sideBC,2) / (powf(sideCA,2) + powf(sideAB,2) - 2*sideAB*sideCA);
  //float cosB = powf(sideCA,2) / (powf(sideBC,2) + powf(sideAB,2) - 2*sideBC*sideAB);
  //float cosC = powf(sideAB,2) / (powf(sideBC,2) + powf(sideCA,2) - 2*sideBC*sideCA);
  cosA = fabs(cosA);
  cosB = fabs(cosB);
  cosC = fabs(cosC);
  printf("cos=%f %f %f\n",cosA,cosB,cosC);
  float d = 2*powf((ma+mb),2) + 2*ma*mb*cosA*cosB*cosC;
  d -= 2*mb*mb*cosA*cosA + ma*(ma+mb)*(cosB*cosB + cosC*cosC);
  d *= dt/(2*mb);
  printf("d=%f\n",d);
  float3 tab = eab*v0ab * (2*(ma + mb) - ma*cosC*cosC);
  tab += ebc*v0bc * (mb*cosC*cosA - (ma + mb)*cosB);
  tab += eca*v0ca * (ma*cosB*cosC - 2*mb*cosA);
  tab *= ma/d;
  float3 tbc = ebc*v0bc * (powf((ma+mb),2) - powf(mb*cosA,2));
  //printf("tbc check 1 = %f %f %f\n", tbc.x, tbc.y, tbc.z);
  tbc += eca*v0ca*ma * (mb*cosA*cosB - (ma + mb)*cosC);
  //printf("tbc check 2 = %f %f %f\n", tbc.x, tbc.y, tbc.z);
  tbc += eab*v0ab*ma * (mb*cosC*cosA - (ma + mb)*cosB);
  //printf("tbc check 3 = %f %f %f\n", tbc.x, tbc.y, tbc.z);
  tbc /= d;
  float3 tca = eca*v0ca * (2*(ma + mb) - ma*cosB*cosB);
  tca += eab*v0ab * (ma*cosB*cosC - 2*mb*cosA);
  tca += ebc*v0bc * (mb*cosA*cosB - (ma + mb)*cosC);
  tca *= ma/d;
  printf("tab=%f %f %f\n",tab.x,tab.y,tab.z);
  printf("tbc=%f %f %f\n",tbc.x,tbc.y,tbc.z);
  printf("tca=%f %f %f\n",tca.x,tca.y,tca.z);
  float3 va = v0a + (dt/(2*ma))*(tab*eab - tca*eca);
  float3 vb = v0b + (dt/(2*mb))*(tbc*ebc - tab*eab);
  float3 vc = v0c + (dt/(2*mc))*(tca*eca - tbc*ebc);
  printf("va=%f %f %f\n",va.x,va.y,va.z);
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
      printf("xs = %f %f %f\n", xs_mol[i].x, xs_mol[i].y, xs_mol[i].z);
      float3 delta = xs_mol[i] - xs_mol[0];
      delta = bounds.minImage(delta);
      xs_mol[i] = xs_mol[0] + delta;
      printf("xd = %f %f %f\n", xs_mol[i].x, xs_mol[i].y, xs_mol[i].z);
    }
    for (int i=0; i<3; i++) {
      float3 delta = xs_0_mol[i] - xs_mol[0];
      delta = bounds.minImage(delta);
      xs_0_mol[i] = xs_mol[0] + delta;
    }
    printf("---------------------------------------------\n");
    printf("mass = %f %f %f\n", mass[0], mass[1], mass[2]);
    float3 comNew = positionsToCOM(xs_mol, mass);
    float3 delta = make_float3(comOld[idx]) - comNew;
    delta = bounds.minImage(delta);
    float3 comOldWrap = comNew + delta;
    printf("comNew = %f %f %f  delta = %f %f %f  comOldWrap = %f %f %f\n", comNew.x, comNew.y, comNew.z, delta.x, delta.y, delta.z, comOldWrap.x, comOldWrap.y, comOldWrap.z);
    //printf("xs_x=%f xs_y=%f xs_z=%f\n", xs_mol[0].x, xs_mol[0].y, xs_mol[0].z);
    settle_xs(dt, comOldWrap, comNew, xs_0_mol, xs_mol);
    printf("Settle  positions: xs_x=%f xs_y=%f xs_z=%f\n", xs_mol[0].x, xs_mol[0].y, xs_mol[0].z);
    printf("                   xs_x=%f xs_y=%f xs_z=%f\n", xs_mol[1].x, xs_mol[1].y, xs_mol[1].z);
    printf("                   xs_x=%f xs_y=%f xs_z=%f\n", xs_mol[2].x, xs_mol[2].y, xs_mol[2].z);
    settle_vs(dt, vs_0_mol, vs_mol, xs_mol, mass);
    printf("Settle velocities: vs_x=%f vs_y=%f vs_z=%f\n", vs_mol[0].x, vs_mol[0].y, vs_mol[0].z);
    printf("                   vs_x=%f vs_y=%f vs_z=%f\n", vs_mol[1].x, vs_mol[1].y, vs_mol[1].z);
    printf("                   vs_x=%f vs_y=%f vs_z=%f\n", vs_mol[2].x, vs_mol[2].y, vs_mol[2].z);
    for (int i=0; i<3; i++) {
      xs[idxs[i]] = make_float4(xs_mol[i]);
    }
    for (int i=0; i<3; i++) {
      vs[idxs[i]] = make_float4(vs_mol[i]);
      vs[idxs[i]].w = 1.0f/mass[i];
    }
  }
}

void FixRigid::createRigid(int id_a, int id_b, int id_c) {
  int4 waterMol = make_int4(0,0,0,0);
  Vector v = state->idToAtom(id_b).pos - state->idToAtom(id_a).pos;
  Vector w = state->idToAtom(id_c).pos - state->idToAtom(id_a).pos;
  float3 normal = cross(v.asFloat3(),w.asFloat3());
  if (state->idToAtom(id_a).mass == state->idToAtom(id_b).mass) {
    waterMol = make_int4(id_c,id_a,id_b,0);
  }
  else if (state->idToAtom(id_b).mass == state->idToAtom(id_c).mass) {
    waterMol = make_int4(id_a,id_b,id_c,0);
  }
  else if (state->idToAtom(id_c).mass == state->idToAtom(id_a).mass) {
    waterMol = make_int4(id_b,id_c,id_a,0);
  } else {
    assert("waterMol set" == "true");
  }
  if (normal.z < 0.0) {
    waterMol = make_int4(waterMol.x,waterMol.z,waterMol.y,0);
  }
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
  //float4 cpu_xs[nMols*3];
  //gpd.xs.dataToHost(activeIdx);
  //std::cout << "before settle: " << cpu_xs[0] << "\n";
  compute_SETTLE<<<NBLOCK(nMols), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), xs_0.data(), gpd.vs(activeIdx), vs_0.data(), com.data(), nMols, dt, gpd.idToIdxs.d_data.data(), bounds);
  //xs_0.get(cpu_xs);
  //std::cout << cpu_xs[0] << "\n";
  return true;
}



void export_FixRigid() {
  py::class_<FixRigid, boost::shared_ptr<FixRigid>, py::bases<Fix> > ( 
								      "FixRigid",
								      py::init<boost::shared_ptr<State>, std::string, std::string>
								      (py::args("state", "handle", "groupHandle")
								       ))
    .def("createRigid", &FixRigid::createRigid,
	 (py::arg("id_a"), py::arg("id_b"), py::arg("id_c"))
	 );
}
    


























