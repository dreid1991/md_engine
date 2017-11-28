#pragma once
#ifndef EVALUATOR_DIHEDRAL_CHARMM
#define EVALUATOR_DIHEDRAL_CHARMM

#include "cutils_math.h"
#include "Dihedral.h"
class DihedralEvaluatorCHARMM {
    public:
        //dihedralType, phi, c, scValues, invLenSqrs, c12Mags, c0,

                //real3 myForce = evaluator.force(dihedralType, phi, c, scValues, invLenSqrs, c12Mags, c0, c, invMagProds, c12Mags, invLens, directors, myIdxInDihedral);
        inline __device__ real dPotential(DihedralCHARMMType dihedralType, real phi) {
#ifdef DASH_DOUBLE
            return dihedralType.k * dihedralType.n * sin(dihedralType.d - dihedralType.n*phi);
#else
            return dihedralType.k * dihedralType.n * sinf(dihedralType.d - dihedralType.n*phi);
#endif
        }

        inline __device__ real potential(DihedralCHARMMType dihedralType, real phi) {
#ifdef DASH_DOUBLE
            return dihedralType.k * (1.0 + cos(dihedralType.n*phi - dihedralType.d));
#else
            return dihedralType.k * (1.0f + cosf(dihedralType.n*phi - dihedralType.d));
#endif

        }

};

#endif


