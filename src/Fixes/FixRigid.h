#pragma once
#ifndef FIXRIGID_H
#define FIXRIGID_H

#include "Python.h"
#include "Fix.h"
#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include "GPUArrayDeviceGlobal.h"

float4 settle_xs(float timestep, float4 com, float4 com1, float4 *xs_0, float4 *xs);
float4 settle_vs(float timestep, float4 *vs_0, float4 *vs);

void export_FixRigid();

class FixRigid : public Fix {
 private:
  GPUArrayDeviceGlobal<int4> waterIdsGPU;
  GPUArrayDeviceGlobal<float4> xs_0;
  GPUArrayDeviceGlobal<float4> vs_0;
  GPUArrayDeviceGlobal<float4> com;
  std::vector<int4> waterIds;

 public:
  FixRigid(SHARED(State), std::string handle_, std::string groupHandle_);
  bool stepInit();
  bool stepFinal();
  bool prepareForRun();
  void createRigid(int, int, int);
};

#endif
