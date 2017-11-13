#pragma once
#ifndef IMPROPER_HARMONIC_H
#include "Improper.h"
class ImproperEvaluatorHarmonic {
public:
    inline __device__ real dPotential(ImproperHarmonicType improperType, real theta) {
        real dTheta = theta - improperType.thetaEq;

        real dp = improperType.k * dTheta;
        return dp;
    }



    inline __device__ real potential(ImproperHarmonicType improperType, real theta) {
        real dTheta = theta - improperType.thetaEq;
        return (1.0f/2.0f) * dTheta * dTheta * improperType.k;

    }


};

#endif

