#pragma once
#ifndef IMPROPER_CVFF_H
#include "Improper.h"
class ImproperEvaluatorCVFF {
public:
    inline __device__ real dPotential(ImproperCVFFType improperType, real theta) {
        return -improperType.d * improperType.k * improperType.n * sinf(improperType.n * theta);
    }
    
    inline __device__ real potential(ImproperCVFFType improperType, real theta) {
        return improperType.k * (1.0f + improperType.d * cosf(improperType.n * theta));

    }


};

#endif

