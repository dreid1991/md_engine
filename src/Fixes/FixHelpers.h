#pragma once
#ifndef FIX_HELPERS_H
#define FIX_HELPERS_H
#include "BoundsGPU.h"
//__device__ real3 harmonicForce(BoundsGPU bounds, real3 posSelf, real3 posOther, real k, real rEq);
inline __device__ real3 harmonicForce(BoundsGPU bounds, real3 posSelf, real3 posOther, real k, real rEq) {
    real3 bondVec = bounds.minImage(posSelf - posOther);
    real r = length(bondVec);
    real dr = r - rEq;
    real rk = k * dr;
    if (r > 0) {//MAKE SURE ALL THIS WORKS, I JUST BORROWED FROM LAMMPS
        real fBond = -rk/r;
        return bondVec * fBond;
    } 
    return make_real3(0, 0, 0);

}



inline __device__ real4 perAtomFromId(cudaTextureObject_t &idToIdxs, real4 *xs, int id) {
    int idx = tex2D<int>(idToIdxs, XIDX(id, sizeof(int)), YIDX(id, sizeof(int)));
    return xs[idx];
}

inline __device__ real4 real4FromIndex(cudaTextureObject_t &xs, int index) {
    return tex2D<real4>(xs, XIDX(index, sizeof(real4)), YIDX(index, sizeof(real4)));
}
#endif
