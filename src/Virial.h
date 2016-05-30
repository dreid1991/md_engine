#pragma once
#ifndef VIRIAL_H
#define VIRIAL_H

class Virial {
    public:
        float vals[6];
        float &__host__ __device__ operator[] (int idx) {
            return vals[idx];
        }
        __host__ __device__ Virial() {
            for (int i=0; i<6; i++) {
                vals[i] = 0;
            }
        }
        inline __host__ __device__ void operator += (Virial &other) {
            for (int i=0; i<6; i++) {
                vals[i] += other[i];
            }
        }
};

#endif
