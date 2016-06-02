#define SMALL 0.0001f
template <class BONDTYPE, class EVALUATOR>
__global__ void compute_force_bond(int nAtoms, float4 *xs, float4 *forces, cudaTextureObject_t idToIdxs, BondGPU *bonds, int *startstops, BONDTYPE *parameters, int nTypes, BoundsGPU bounds, EVALUATOR T) {
    int idx = GETIDX();
    extern __shared__ int all_shr[];
    int idxBeginCopy = startstops[blockDim.x*blockIdx.x];
    int idxEndCopy = startstops[min(nAtoms, blockDim.x*(blockIdx.x+1))];
    BondGPU *bonds_shr = (BondGPU *) all_shr;
    BONDTYPE *parameters_shr = (BONDTYPE *) (bonds_shr + (idxEndCopy - idxBeginCopy));
    copyToShared<BondGPU>(bonds + idxBeginCopy, bonds_shr, idxEndCopy - idxBeginCopy);
    copyToShared<BONDTYPE>(parameters, parameters_shr, nTypes);
    __syncthreads();
    if (idx < nAtoms) {
  //      printf("going to compute %d\n", idx);
        int startIdx = startstops[idx]; 
        int endIdx = startstops[idx+1];
        //so start/end is the index within the entire bond list.
        //startIdx - idxBeginCopy gives my index in shared memory
        int shr_idx = startIdx - idxBeginCopy;
        int n = endIdx - startIdx;
        if (n>0) { //if you have atoms w/ zero bonds at the end, they will read one off the end of the bond list
            int myId = bonds_shr[shr_idx].myId;

            int myIdx = tex2D<int>(idToIdxs, XIDX(myId, sizeof(int)), YIDX(myId, sizeof(int)));


            float3 pos = make_float3(xs[myIdx]);
            float3 forceSum = make_float3(0, 0, 0);
            for (int i=0; i<n; i++) {
                BondGPU b = bonds_shr[shr_idx + i];
                int type = b.type;
                BONDTYPE bondType = parameters_shr[type];

                int otherId = b.otherId;
                int otherIdx = tex2D<int>(idToIdxs, XIDX(otherId, sizeof(int)), YIDX(otherId, sizeof(int)));

                float3 posOther = make_float3(xs[otherIdx]);
                // printf("atom %d bond %d gets force %f\n", idx, i, harmonicForce(bounds, pos, posOther, b.k, b.rEq));
                // printf("xs %f %f\n", pos.x, posOther.x);
                float3 bondVec  = bounds.minImage(pos - posOther);
                float r = length(bondVec);
                forceSum += T.force(bondVec, r, bondType);
            }
            forces[myIdx] += forceSum;
        }
    }
}

