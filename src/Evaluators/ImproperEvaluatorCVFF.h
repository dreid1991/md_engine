#pragma once
#ifndef IMPROPER_CVFF_H
#include "Improper.h"
class ImproperEvaluatorCVFF {
public:
    inline __device__ float dPotential(ImproperCVFFType improperType, float theta) {
        //float dTheta = theta - improperType.thetaEq;

        //float dp = improperType.k * dTheta;
        //return dp;
        return 0;
    }
    inline __device__ float potential(ImproperCVFFType improperType, float theta) {
        //float dTheta = theta - improperType.thetaEq;
        //return (1.0f/2.0f) * dTheta * dTheta * improperType.k;
        return 0;

    }


};

#endif

