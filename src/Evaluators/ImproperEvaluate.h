template <class IMPROPERTYPE, class EVALUATOR, bool COMPUTEVIRIALS> 
__global__ void compute_force_improper(int nImpropers, float4 *xs, float4 *fs, int *idToIdxs, ImproperGPU *impropers, BoundsGPU bounds, IMPROPERTYPE *parameters_arg, int nParameters, Virial *virials, bool usingSharedMemForParams, EVALUATOR evaluator) {

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
        float3 positions[4];
        int idxs[4];
        for (int i=0; i<4; i++) {
            int idxOther = idToIdxs[improper.ids[i]];
            idxs[i] = idxOther;
            positions[i] = make_float3(xs[idxOther]);
        }
        for (int i=1; i<4; i++) {
            positions[i] = positions[0] + bounds.minImage(positions[i]-positions[0]);
        }
        float3 directors[3]; //vb_xyz in lammps
        float lenSqrs[3]; //bnmag2 in lammps
        float lens[3]; //bnmag in lammps
        float invLenSqrs[3]; //sb in lammps
        float invLens[3];
        directors[0] = positions[0] - positions[1];
        directors[1] = positions[2] - positions[1];
        directors[2] = positions[3] - positions[2];
        for (int i=0; i<3; i++) {
            //printf("directors %d is %f %f %f\n", i, directors[i].x, directors[i].y, directors[i].z);
            lenSqrs[i] = lengthSqr(directors[i]);
            lens[i] = sqrtf(lenSqrs[i]);
            invLenSqrs[i] = 1.0f / lenSqrs[i];
            invLens[i] = 1.0f / lens[i];
            //   printf("inv len sqrs %d is %f\n", i, invLenSqrs[i]);
        }

        float angleBits[3]; //c0, 1, 2
        angleBits[0] = dot(directors[0], directors[2]) * invLens[0] * invLens[2];
        angleBits[1] = dot(directors[0], directors[1]) * invLens[0] * invLens[1];
        angleBits[2] = -dot(directors[2], directors[1]) * invLens[2] * invLens[1];

        float scValues[3]; //???, is s1, s2, s12 in lammps
        for (int i=0; i<2; i++) {
            scValues[i] = 1.0f - angleBits[i+1] * angleBits[i+1];
            if (scValues[i] < SMALL) {
                scValues[i] = SMALL;
            }
            scValues[i] = 1.0 / scValues[i];
        }
        scValues[2] = sqrtf(scValues[0] * scValues[1]);
        float c = (angleBits[1]*angleBits[2] + angleBits[0]) * scValues[2];

        if (c > 1.0f) {
            c = 1.0f;
        } else if (c < -1.0f) {
            c = -1.0f;
        }
        float s = sqrtf(1.0f - c*c);
        if (s < SMALL) {
            s = SMALL;
        }
        float theta = acosf(c);
        float dPotential = evaluator.dPotential(improperType, theta);

        dPotential *= -2.0f / s;
        scValues[2] *= dPotential;
        c *= dPotential;

        float a11 = c * invLenSqrs[0] * scValues[0];
        float a22 = - invLenSqrs[1] * (2.0f * angleBits[0] * scValues[2] - c * (scValues[0] + scValues[1]));
        float a33 = c * invLenSqrs[2] * scValues[1];
        float a12 = -invLens[0] * invLens[1] * (angleBits[1] * c * scValues[0] + angleBits[2] * scValues[2]);
        float a13 = -invLens[0] * invLens[2] * scValues[2];
        float a23 = invLens[1] * invLens[2] * (angleBits[2] * c * scValues[1] + angleBits[1] * scValues[2]);

        float3 sFloat3 = make_float3(
                                     a22*directors[1].x + a23*directors[2].x + a12*directors[0].x
                                     ,  a22*directors[1].y + a23*directors[2].y + a12*directors[0].y
                                     ,  a22*directors[1].z + a23*directors[2].z + a12*directors[0].z
                                    );
        float3 a11Dir1 = directors[0] * a11;
        float3 a12Dir2 = directors[1] * a12;
        float3 a13Dir3 = directors[2] * a13;
        float3 forces[4];
        forces[0].x = a11Dir1.x + a12Dir2.x + a13Dir3.x;
        forces[0].y = a11Dir1.y + a12Dir2.y + a13Dir3.y;
        forces[0].z = a11Dir1.z + a12Dir2.z + a13Dir3.z;

        forces[1] = -sFloat3 - forces[0];
        float3 a13Dir1 = directors[0] * a13;
        float3 a23Dir2 = directors[1] * a23;
        float3 a33Dir3 = directors[2] * a33;
        forces[3].x = a13Dir1.x + a23Dir2.x + a33Dir3.x;
        forces[3].y = a13Dir1.y + a23Dir2.y + a33Dir3.y;
        forces[3].z = a13Dir1.z + a23Dir2.z + a33Dir3.z;
        forces[2] = sFloat3 - forces[3];
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
__global__ void compute_energy_improper(int nAtoms, float4 *xs, float *perParticleEng, int *idToIdxs, ImproperGPU *impropers, int *startstops, BoundsGPU bounds, IMPROPERTYPE *parameters_arg, int nParameters, bool usingSharedMemForParams, EVALUATOR T) {

    int idx = GETIDX();
    extern __shared__ char all_shr[];
    int idxBeginCopy = startstops[blockDim.x*blockIdx.x];
    int idxEndCopy = startstops[min(nAtoms, blockDim.x*(blockIdx.x+1))];
    ImproperGPU *impropers_shr = (ImproperGPU *) all_shr;
    int sizeImpropers = (idxEndCopy - idxBeginCopy) * sizeof(ImproperGPU);
    copyToShared<ImproperGPU>(impropers + idxBeginCopy, impropers_shr, idxEndCopy - idxBeginCopy);
    IMPROPERTYPE *parameters;
    if (usingSharedMemForParams) {
        parameters = (IMPROPERTYPE *) (all_shr + sizeImpropers);
        copyToShared<IMPROPERTYPE>(parameters_arg, parameters, nParameters);
    } else {
        parameters = parameters_arg;
    }
    __syncthreads();

    __syncthreads();
    if (idx < nAtoms) { //HEY - THIS SHOULD BE < nAtoms
  //      printf("going to compute %d\n", idx);
        int startIdx = startstops[idx];
        int endIdx = startstops[idx+1];
        //so start/end is the index within the entire bond list.
        //startIdx - idxBeginCopy gives my index in shared memory
        int shr_idx = startIdx - idxBeginCopy;
        int n = endIdx - startIdx;
        if (n) {
            int myIdxInImproper = impropers_shr[shr_idx].type >> 29;
            int idSelf = impropers_shr[shr_idx].ids[myIdxInImproper];
            
            int idxSelf = idToIdxs[idSelf]; 
        
            float3 pos = make_float3(xs[idxSelf]);
           // printf("I am idx %d and I am evaluating atom with pos %f %f %f\n", idx, pos.x, pos.y, pos.z);
            float energySum = 0;
            for (int i=0; i<n; i++) {
                ImproperGPU improper = impropers_shr[shr_idx + i];
                uint32_t typeFull = improper.type;
                myIdxInImproper = typeFull >> 29;
                int type = static_cast<int>((typeFull << 3) >> 3);   
                IMPROPERTYPE improperType = parameters[type];
                float3 positions[4];
                positions[myIdxInImproper] = pos;
                int toGet[3];
                if (myIdxInImproper==0) {
                    toGet[0] = 1;
                    toGet[1] = 2;
                    toGet[2] = 3;
                } else if (myIdxInImproper==1) {
                    toGet[0] = 0;
                    toGet[1] = 2;
                    toGet[2] = 3;
                } else if (myIdxInImproper==2) {
                    toGet[0] = 0;
                    toGet[1] = 1;
                    toGet[2] = 3;
                } else if (myIdxInImproper==3) {
                    toGet[0] = 0;
                    toGet[1] = 1;
                    toGet[2] = 2;
                }
                for (int i=0; i<3; i++) {
                    int idxOther = idToIdxs[improper.ids[toGet[i]]];
                    positions[toGet[i]] = make_float3(xs[idxOther]);
                }
                for (int i=1; i<3; i++) {
                    positions[i] = positions[0] + bounds.minImage(positions[i]-positions[0]);
                }
                float3 directors[3]; //vb_xyz in lammps
                float lenSqrs[3]; //bnmag2 in lammps
                float lens[3]; //bnmag in lammps
                float invLenSqrs[3]; //sb in lammps
                float invLens[3];
                directors[0] = positions[0] - positions[1];
                directors[1] = positions[2] - positions[1];
                directors[2] = positions[3] - positions[2];
                for (int i=0; i<3; i++) {
                    //printf("directors %d is %f %f %f\n", i, directors[i].x, directors[i].y, directors[i].z);
                    lenSqrs[i] = lengthSqr(directors[i]);
                    lens[i] = sqrtf(lenSqrs[i]);
                    invLenSqrs[i] = 1.0f / lenSqrs[i];
                    invLens[i] = 1.0f / lens[i];
                 //   printf("inv len sqrs %d is %f\n", i, invLenSqrs[i]);
                }

                float angleBits[3]; //c0, 1, 2
                angleBits[0] = dot(directors[0], directors[2]) * invLens[0] * invLens[2];
                angleBits[1] = dot(directors[0], directors[1]) * invLens[0] * invLens[1];
                angleBits[2] = -dot(directors[2], directors[1]) * invLens[2] * invLens[1];

                float scValues[3]; //???, is s1, s2, s12 in lammps
                for (int i=0; i<2; i++) {
                    scValues[i] = 1.0f - angleBits[i+1] * angleBits[i+1];
                    if (scValues[i] < SMALL) {
                        scValues[i] = SMALL;
                    }
                    scValues[i] = 1.0 / scValues[i];
                }
                scValues[2] = sqrtf(scValues[0] * scValues[1]);
                float c = (angleBits[1]*angleBits[2] + angleBits[0]) * scValues[2];

                if (c > 1.0f) {
                    c = 1.0f;
                } else if (c < -1.0f) {
                    c = -1.0f;
                }
                float s = sqrtf(1.0f - c*c);
                if (s < SMALL) {
                    s = SMALL;
                }
                float theta = acosf(c);
                energySum += T.energy(improperType, theta, scValues, invLenSqrs, invLens, angleBits, s, c, directors, myIdxInImproper);



            }
            perParticleEng[idxSelf] += energySum;
        }
    }
}
