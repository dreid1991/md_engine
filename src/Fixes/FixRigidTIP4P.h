#pragma once
#ifndef FIXRIGIDTIP4P_H
#define FIXRIGIDTIP4P_H

#include "Python.h"
#include "Fix.h"
#include "FixBond.h"
#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include "GPUArrayDeviceGlobal.h"

//void settle_xs(float timestep, float3 com, float3 com1, float3 *xs_0, float3 *xs, float3 *fix_len);
//void settle_vs(float timestep, float3 *vs_0, float3 *vs, float3 *xs, float *mass, float3 *fix_len);

void export_FixRigidTIP4P();

class FixRigidTIP4P : public Fix {
 private:
  GPUArrayDeviceGlobal<int4> waterIdsGPU;
  GPUArrayDeviceGlobal<float4> xs_0;
  GPUArrayDeviceGlobal<float4> vs_0;
  GPUArrayDeviceGlobal<float4> com;
  GPUArrayDeviceGlobal<float4> fix_len;
  std::vector<int4> waterIds;
  std::vector<BondVariant> bonds;
  std::vector<float4> invMassSums;

 public:
  FixRigidTIP4P(SHARED(State), std::string handle_, std::string groupHandle_);
  bool stepInit();
  bool stepFinal();
  bool prepareForRun();
  void createTIP4P(int id_a, int id_b, int id_c, int id_m);

  std::vector<BondVariant> *getBonds() {
    return &bonds;
  }
};

#endif
