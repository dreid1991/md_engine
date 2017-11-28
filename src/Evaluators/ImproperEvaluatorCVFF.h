#pragma once
#ifndef IMPROPER_CVFF_H
#include "Improper.h"
class ImproperEvaluatorCVFF {
public:
    inline __device__ real dPotential(ImproperCVFFType improperType, real theta) {
#ifdef DASH_DOUBLE
        return -improperType.d * improperType.k * improperType.n * sin(improperType.n * theta);
#else
        return -improperType.d * improperType.k * improperType.n * sinf(improperType.n * theta);
#endif
    }
    
    inline __device__ real potential(ImproperCVFFType improperType, real theta) {
#ifdef DASH_DOUBLE
        return improperType.k * (1.0 + improperType.d * cos(improperType.n * theta));
#else
        return improperType.k * (1.0f + improperType.d * cosf(improperType.n * theta));
#endif

    }


};

#endif

