template <class IMPROPERTYPE, class EVALUATOR, bool COMPUTEVIRIALS> 
__global__ void compute_force_improper(int nAtoms, float4 *xs, float4 *forces, int *idToIdxs, ImproperGPU *impropers, int *startstops, BoundsGPU bounds, IMPROPERTYPE *parameters, int nParameters, Virial *virials, EVALUATOR evaluator) {


    int idx = GETIDX();
    extern __shared__ int all_shr[];
    int idxBeginCopy = startstops[blockDim.x*blockIdx.x];
    int idxEndCopy = startstops[min(nAtoms, blockDim.x*(blockIdx.x+1))];

    ImproperGPU *impropers_shr = (ImproperGPU *) all_shr;
    IMPROPERTYPE *parameters_shr = (IMPROPERTYPE *) (impropers_shr + (idxEndCopy - idxBeginCopy));
    copyToShared<ImproperGPU>(impropers + idxBeginCopy, impropers_shr, idxEndCopy - idxBeginCopy);
    copyToShared<IMPROPERTYPE>(parameters, parameters_shr, nParameters);

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
            Virial sumVirials (0, 0, 0, 0, 0, 0);
            int myIdxInImproper = impropers_shr[shr_idx].type >> 29;
            int idSelf = impropers_shr[shr_idx].ids[myIdxInImproper];
            
            int idxSelf = idToIdxs[idSelf]; 
        
            float3 pos = make_float3(xs[idxSelf]);
           // printf("I am idx %d and I am evaluating atom with pos %f %f %f\n", idx, pos.x, pos.y, pos.z);
            float3 forceSum = make_float3(0, 0, 0);
            for (int i=0; i<n; i++) {
                ImproperGPU improper = impropers_shr[shr_idx + i];
                uint32_t typeFull = improper.type;
                myIdxInImproper = typeFull >> 29;
                int type = static_cast<int>((typeFull << 3) >> 3);   
                IMPROPERTYPE improperType = parameters_shr[type];
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
                if (COMPUTEVIRIALS) {
                    float3 allForces[4];
                    evaluator.forcesAll(improperType, theta, scValues, invLenSqrs, invLens, angleBits, s, c, directors, allForces);
                    //printf("forces %f %f %f , %f %f %f , %f %f %f , %f %f %f\n", allForces[0].x, allForces[0].y, allForces[0].z,allForces[1].x, allForces[1].y, allForces[1].z,allForces[2].x, allForces[2].y, allForces[2].z,allForces[3].x, allForces[3].y, allForces[3].z);
                    computeVirial(sumVirials, allForces[0], directors[0]);
                    computeVirial(sumVirials, allForces[2], directors[1]);
                    computeVirial(sumVirials, allForces[3], directors[1] + directors[2]);
                    forceSum += allForces[myIdxInImproper];
                    
                } else {
                    float3 force = evaluator.force(improperType, theta, scValues, invLenSqrs, invLens, angleBits, s, c, directors, myIdxInImproper);
                    forceSum += force;
                }



            }
            forces[idxSelf] += forceSum;
            if (COMPUTEVIRIALS) {
                sumVirials *= 0.25f;
                virials[idx] += sumVirials;
            }
        }
    }
}





template <class IMPROPERTYPE, class EVALUATOR> 
__global__ void compute_energy_improper(int nAtoms, float4 *xs, float *perParticleEng, int *idToIdxs, ImproperGPU *impropers, int *startstops, BoundsGPU bounds, IMPROPERTYPE *parameters, int nParameters, EVALUATOR T) {


    int idx = GETIDX();
    extern __shared__ int all_shr[];
    int idxBeginCopy = startstops[blockDim.x*blockIdx.x];
    int idxEndCopy = startstops[min(nAtoms, blockDim.x*(blockIdx.x+1))];

    ImproperGPU *impropers_shr = (ImproperGPU *) all_shr;
    IMPROPERTYPE *parameters_shr = (IMPROPERTYPE *) (impropers_shr + (idxEndCopy - idxBeginCopy));
    copyToShared<ImproperGPU>(impropers + idxBeginCopy, impropers_shr, idxEndCopy - idxBeginCopy);
    copyToShared<IMPROPERTYPE>(parameters, parameters_shr, nParameters);

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
                IMPROPERTYPE improperType = parameters_shr[type];
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
