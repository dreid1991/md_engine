#pragma once
#ifndef VIRIAL_H
#define VIRIAL_H
//#include "Logging.h"
//as xx, yy, zz, xy, xz, yz
#include "globalDefs.h"

class Virial {
    public:
        real vals[6];
        real &__host__ __device__ operator[] (int idx) {
            return vals[idx];
        }
        __host__ __device__ Virial() {};
        __host__ __device__ Virial(real xx, real yy, real zz, real xy, real xz, real yz) {
            vals[0] = xx;
            vals[1] = yy;
            vals[2] = zz;
            vals[3] = xy;
            vals[4] = xz;
            vals[5] = yz;
        }
        inline __host__ __device__ void operator += (Virial &other) {
            for (int i=0; i<6; i++) {
                vals[i] += other[i];
            }
        }
        inline __host__ __device__ void operator *=(real x) {
            for (int i=0; i<6; i++) {
                vals[i] *= x;
            }
        }
        inline __host__ __device__ Virial operator *(real x) {
            Virial res;
            for (int i=0; i<6; i++) {
                res[i] = vals[i] * x;
            }
            return res;
        }
        /*
        inline __host__ __device__ void operator *=(double x) {
            for (int i=0; i<6; i++) {
                vals[i] *= x;
            }
        } 
        */
        /*
        real operator[] (int n) { //for python interface
            if (n > 0 and n < 6) {
                return vals[n];
            } else {
                mdAssert(n>0 and n<6);
                return 0;
            }
        }
        */
};

#endif
