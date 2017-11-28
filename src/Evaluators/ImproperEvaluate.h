template <class IMPROPERTYPE, class EVALUATOR, bool COMPUTEVIRIALS> 
__global__ void compute_force_improper(int nImpropers, real4 *xs, real4 *fs, int *idToIdxs, ImproperGPU *impropers, BoundsGPU bounds, IMPROPERTYPE *parameters_arg, int nParameters, Virial *virials, bool usingSharedMemForParams, EVALUATOR evaluator) {

    int idx = GETIDX();
    extern __shared__ char all_shr[];
    IMPROPERTYPE *parameters;
    if (usingSharedMemForParams) {
        parameters = (IMPROPERTYPE *) (all_shr);
        copyToShared<IMPROPERTYPE>(parameters_arg, parameters, nParameters);
    } else {
        parameters = parameters_arg;
    }
    __syncthreads();
    if (idx < nImpropers) { 
        //      printf("going to compute %d\n", idx);
        Virial sumVirials (0, 0, 0, 0, 0, 0);
        ImproperGPU improper = impropers[idx];
        uint32_t typeFull = improper.type;
        int type = static_cast<int>((typeFull << 3) >> 3);   
        IMPROPERTYPE improperType = parameters[type];
        real3 positions[4];
        int idxs[4];
        for (int i=0; i<4; i++) {
            int idxOther = idToIdxs[improper.ids[i]];
            idxs[i] = idxOther;
            positions[i] = make_real3(xs[idxOther]);
        }
        for (int i=1; i<4; i++) {
            positions[i] = positions[0] + bounds.minImage(positions[i]-positions[0]);
        }
        real3 directors[3]; //vb_xyz in lammps
        real lenSqrs[3]; //bnmag2 in lammps
        real lens[3]; //bnmag in lammps
        real invLenSqrs[3]; //sb in lammps
        real invLens[3];
        directors[0] = positions[0] - positions[1];
        directors[1] = positions[2] - positions[1];
        directors[2] = positions[3] - positions[2];
        for (int i=0; i<3; i++) {
            //printf("directors %d is %f %f %f\n", i, directors[i].x, directors[i].y, directors[i].z);
            lenSqrs[i] = lengthSqr(directors[i]);
#ifdef DASH_DOUBLE
            lens[i] = sqrt(lenSqrs[i]);
            invLenSqrs[i] = 1.0 / lenSqrs[i];
            invLens[i] = 1.0 / lens[i];
#else
            lens[i] = sqrtf(lenSqrs[i]);
            invLenSqrs[i] = 1.0f / lenSqrs[i];
            invLens[i] = 1.0f / lens[i];
#endif
            //   printf("inv len sqrs %d is %f\n", i, invLenSqrs[i]);
        }

        real angleBits[3]; //c0, 1, 2
        angleBits[0] = dot(directors[0], directors[2]) * invLens[0] * invLens[2];
        angleBits[1] = dot(directors[0], directors[1]) * invLens[0] * invLens[1];
        angleBits[2] = -dot(directors[2], directors[1]) * invLens[2] * invLens[1];

        real scValues[3]; //???, is s1, s2, s12 in lammps
        for (int i=0; i<2; i++) {
            scValues[i] = 1.0 - angleBits[i+1] * angleBits[i+1];
            if (scValues[i] < SMALL) {
                scValues[i] = SMALL;
            }
            scValues[i] = 1.0 / scValues[i];
        }
        scValues[2] = sqrtf(scValues[0] * scValues[1]);
        real c = (angleBits[1]*angleBits[2] + angleBits[0]) * scValues[2];

        if (c > 1.0) {
            c = 1.0;
        } else if (c < -1.0) {
            c = -1.0;
        }
#ifdef DASH_DOUBLE
        real s = sqrt(1.0 - c*c);
#else
        real s = sqrtf(1.0f - c*c);
#endif
        if (s < SMALL) {
            s = SMALL;
        }
#ifdef DASH_DOUBLE
        real theta = acos(c);
#else
        real theta = acosf(c);
#endif
        real dPotential = evaluator.dPotential(improperType, theta);

        dPotential *= -2.0 / s;
        scValues[2] *= dPotential;
        c *= dPotential;

        real a11 = c * invLenSqrs[0] * scValues[0];
        real a22 = - invLenSqrs[1] * (2.0 * angleBits[0] * scValues[2] - c * (scValues[0] + scValues[1]));
        real a33 = c * invLenSqrs[2] * scValues[1];
        real a12 = -invLens[0] * invLens[1] * (angleBits[1] * c * scValues[0] + angleBits[2] * scValues[2]);
        real a13 = -invLens[0] * invLens[2] * scValues[2];
        real a23 = invLens[1] * invLens[2] * (angleBits[2] * c * scValues[1] + angleBits[1] * scValues[2]);

        real3 sreal3 = make_real3(
                                     a22*directors[1].x + a23*directors[2].x + a12*directors[0].x
                                     ,  a22*directors[1].y + a23*directors[2].y + a12*directors[0].y
                                     ,  a22*directors[1].z + a23*directors[2].z + a12*directors[0].z
                                    );
        real3 a11Dir1 = directors[0] * a11;
        real3 a12Dir2 = directors[1] * a12;
        real3 a13Dir3 = directors[2] * a13;
        real3 forces[4];
        forces[0].x = a11Dir1.x + a12Dir2.x + a13Dir3.x;
        forces[0].y = a11Dir1.y + a12Dir2.y + a13Dir3.y;
        forces[0].z = a11Dir1.z + a12Dir2.z + a13Dir3.z;

        forces[1] = -sreal3 - forces[0];
        real3 a13Dir1 = directors[0] * a13;
        real3 a23Dir2 = directors[1] * a23;
        real3 a33Dir3 = directors[2] * a33;
        forces[3].x = a13Dir1.x + a23Dir2.x + a33Dir3.x;
        forces[3].y = a13Dir1.y + a23Dir2.y + a33Dir3.y;
        forces[3].z = a13Dir1.z + a23Dir2.z + a33Dir3.z;
        forces[2] = sreal3 - forces[3];
        for (int i=0; i<4; i++) {
            atomicAdd(&(fs[idxs[i]].x), (forces[i].x));
            atomicAdd(&(fs[idxs[i]].y), (forces[i].y));
            atomicAdd(&(fs[idxs[i]].z), (forces[i].z));
            //printf("imp f %d is %f %f %f\n", i, forces[i].x, forces[i].y, forces[i].z);
        }

        if (COMPUTEVIRIALS) {
            computeVirial(sumVirials, forces[0], directors[0]);
            computeVirial(sumVirials, forces[2], directors[1]);
            computeVirial(sumVirials, forces[3], directors[1] + directors[2]);
            for (int i=0; i<6; i++) {
                //printf("imp vir %d %f\n", i, sumVirials[i]);
                atomicAdd(&(virials[idxs[0]][i]), sumVirials[i]);
            }

        } 


    }
 //   forces[idxSelf] += forceSum;
  //  if (COMPUTEVIRIALS) {
   //     sumVirials *= 0.25f;
  //      virials[idx] += sumVirials;
  //  }
}




template <class IMPROPERTYPE, class EVALUATOR> 
__global__ void compute_energy_improper(int nImpropers, real4 *xs, real *perParticleEng, int *idToIdxs, ImproperGPU *impropers, BoundsGPU bounds, IMPROPERTYPE *parameters_arg, int nParameters, bool usingSharedMemForParams, EVALUATOR evaluator) {

    int idx = GETIDX();
    extern __shared__ char all_shr[];
    IMPROPERTYPE *parameters;
    if (usingSharedMemForParams) {
        parameters = (IMPROPERTYPE *) (all_shr);
        copyToShared<IMPROPERTYPE>(parameters_arg, parameters, nParameters);
    } else {
        parameters = parameters_arg;
    }
    __syncthreads();
    if (idx < nImpropers) { 
        //      printf("going to compute %d\n", idx);
        ImproperGPU improper = impropers[idx];
        uint32_t typeFull = improper.type;
        int type = static_cast<int>((typeFull << 3) >> 3);   
        IMPROPERTYPE improperType = parameters[type];
        real3 positions[4];
        int idxs[4];
        for (int i=0; i<4; i++) {
            int idxOther = idToIdxs[improper.ids[i]];
            idxs[i] = idxOther;
            positions[i] = make_real3(xs[idxOther]);
        }
        for (int i=1; i<4; i++) {
            positions[i] = positions[0] + bounds.minImage(positions[i]-positions[0]);
        }
        real3 directors[3]; //vb_xyz in lammps
        real lenSqrs[3]; //bnmag2 in lammps
        real lens[3]; //bnmag in lammps
        real invLens[3];
        directors[0] = positions[0] - positions[1];
        directors[1] = positions[2] - positions[1];
        directors[2] = positions[3] - positions[2];
        for (int i=0; i<3; i++) {
            //printf("directors %d is %f %f %f\n", i, directors[i].x, directors[i].y, directors[i].z);
            //
            lenSqrs[i] = lengthSqr(directors[i]);
#ifdef DASH_DOUBLE
            lens[i] = sqrt(lenSqrs[i]);
            invLens[i] = 1.0 / lens[i];
#else
            lens[i] = sqrtf(lenSqrs[i]);
            invLens[i] = 1.0f / lens[i];
#endif
        }

        real angleBits[3]; //c0, 1, 2
        angleBits[0] = dot(directors[0], directors[2]) * invLens[0] * invLens[2];
        angleBits[1] = dot(directors[0], directors[1]) * invLens[0] * invLens[1];
        angleBits[2] = -dot(directors[2], directors[1]) * invLens[2] * invLens[1];

        real scValues[3]; //???, is s1, s2, s12 in lammps
        for (int i=0; i<2; i++) {
            scValues[i] = 1.0 - angleBits[i+1] * angleBits[i+1];
            if (scValues[i] < SMALL) {
                scValues[i] = SMALL;
            }
            scValues[i] = 1.0 / scValues[i];
        }
#ifdef DASH_DOUBLE
        scValues[2] = sqrt(scValues[0] * scValues[1]);
#else
        scValues[2] = sqrtf(scValues[0] * scValues[1]);
#endif
        real c = (angleBits[1]*angleBits[2] + angleBits[0]) * scValues[2];

        if (c > 1.0) {
            c = 1.0;
        } else if (c < -1.0) {
            c = -1.0;
        }
#ifdef DASH_DOUBLE
        real s = sqrt(1.0 - c*c);
#else
        real s = sqrtf(1.0f - c*c);
#endif
        if (s < SMALL) {
            s = SMALL;
        }
#ifdef DASH_DOUBLE
        real theta = acos(c);
#else
        real theta = acosf(c);
#endif
        real potential = 0.25 * evaluator.potential(improperType, theta);
        for (int i=0; i<4; i++) {
            atomicAdd(perParticleEng + idxs[i], potential);
        }


    }
}
