#pragma once
#ifndef VIRIAL_H
#define VIRIAL_H
//#include "Logging.h"
//as xx, yy, zz, xy, xz, yz
class Virial {
    public:
        float vals[6];
        float &__host__ __device__ operator[] (int idx) {
            return vals[idx];
        }
        __host__ Virial() {};
        __host__ __device__ Virial(float xx, float yy, float zz, float xy, float xz, float yz) {
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
        inline __host__ void operator /= (double n) {
            for (int i=0; i<6; i++) {
                vals[i] /= n;
            }
        }
        /*
        float operator[] (int n) { //for python interface
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
