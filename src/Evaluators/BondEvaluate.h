#define SMALL 0.0001f
template <class ANGLETYPE, class EVALUATOR>
__global__ void compute_force_bond(int nAtoms, float4 *xs, float4 *forces, cudaTextureObject_t idToIdxs, BondHarmonicGPU *bonds, int *startstops, BoundsGPU bounds) {
    int idx = GETIDX();
    extern __shared__ BondHarmonicGPU bonds_shr[];
    int idxBeginCopy = startstops[blockDim.x*blockIdx.x];
    int idxEndCopy = startstops[min(nAtoms, blockDim.x*(blockIdx.x+1))];
    copyToShared<BondHarmonicGPU>(bonds + idxBeginCopy, bonds_shr, idxEndCopy - idxBeginCopy);
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
            int idSelf = bonds_shr[shr_idx].myId;

            int idxSelf = tex2D<int>(idToIdxs, XIDX(idSelf, sizeof(int)), YIDX(idSelf, sizeof(int)));


            float3 pos = make_float3(xs[idxSelf]);
            float3 forceSum = make_float3(0, 0, 0);
            for (int i=0; i<n; i++) {
                BondHarmonicGPU b = bonds_shr[shr_idx + i];
                int idOther = b.idOther;
                int idxOther = tex2D<int>(idToIdxs, XIDX(idOther, sizeof(int)), YIDX(idOther, sizeof(int)));

                float3 posOther = make_float3(xs[idxOther]);
                // printf("atom %d bond %d gets force %f\n", idx, i, harmonicForce(bounds, pos, posOther, b.k, b.rEq));
                // printf("xs %f %f\n", pos.x, posOther.x);
                forceSum += harmonicForce(bounds, pos, posOther, b.k, b.rEq);
            }
            forces[idxSelf] += forceSum;
        }
    }
}

