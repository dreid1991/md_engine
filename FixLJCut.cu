#include "FixLJCut.h"
#include "State.h"
#include "cutils_func.h"
FixLJCut::FixLJCut(SHARED(State) state_, string handle_) : FixPair(state_, handle_, "all", LJCutType, 1), epsHandle("eps"), sigHandle("sig"), rCutHandle("rCut") {
    initializeParameters(epsHandle, epsilons);
    initializeParameters(sigHandle, sigmas);
    initializeParameters(rCutHandle, rCuts);
    forceSingle = true;

}



__global__ void compute_cu(int nAtoms, float4 *xs, float4 *fs, uint16_t *neighborCounts, uint *neighborlist, uint32_t *cumulSumMaxPerBlock, int warpSize, float *sigs, float *eps, float *rCutSqrs, int numTypes,  BoundsGPU bounds, float onetwoStr, float onethreeStr, float onefourStr) {
    float multipliers[4] = {1, onetwoStr, onethreeStr, onefourStr};
    extern __shared__ float paramsAll[];
    int sqrSize = numTypes*numTypes;
    float *sigs_shr = paramsAll;
    float *eps_shr = paramsAll + sqrSize;
    float *rCutSqrs_shr = paramsAll + 2*sqrSize;
    copyToShared<float>(eps, eps_shr, sqrSize);
    copyToShared<float>(sigs, sigs_shr, sqrSize);
    copyToShared<float>(rCutSqrs, rCutSqrs_shr, sqrSize);
    __syncthreads();

    int idx = GETIDX();
    if (idx < nAtoms) {
        int baseIdx = baseNeighlistIdx(cumulSumMaxPerBlock, warpSize);
        float4 posWhole = xs[idx];
        int type = * (int *) &posWhole.w;
        float3 pos = make_float3(posWhole);

        float3 forceSum = make_float3(0, 0, 0);

        int numNeigh = neighborCounts[idx];
        for (int i=0; i<numNeigh; i++) {
            int nlistIdx = baseIdx + warpSize * i;
            uint otherIdxRaw = neighborlist[nlistIdx];
            uint neighDist = otherIdxRaw >> 30;
            float multiplier = multipliers[neighDist];
            if (multiplier) {
                uint otherIdx = otherIdxRaw & EXCL_MASK;
                float4 otherPosWhole = xs[otherIdx];
                int otherType = * (int *) &otherPosWhole.w;
                float3 otherPos = make_float3(otherPosWhole);
                //then wrap and compute forces!
                float sig6 = squareVectorItem(sigs_shr, numTypes, type, otherType);
                float epstimes24 = squareVectorItem(eps_shr, numTypes, type, otherType);
                float3 dr = bounds.minImage(pos - otherPos);
                float lenSqr = lengthSqr(dr);
                float rCutSqr = squareVectorItem(rCutSqrs_shr, numTypes, type, otherType);
                if (lenSqr < rCutSqr) {
                    float p1 = epstimes24*2*sig6*sig6;
                    float p2 = epstimes24*sig6;
                    float r2inv = 1/lenSqr;
                    float r6inv = r2inv*r2inv*r2inv;
                    float forceScalar = r6inv * r2inv * (p1 * r6inv - p2) * multiplier;

                    float3 forceVec = dr * forceScalar;
                    forceSum += forceVec;
                }
            }

        }   
    //    printf("LJ force %f %f %f \n", forceSum.x, forceSum.y, forceSum.z);
        float4 forceCur = fs[idx];
        forceCur += forceSum;
        fs[idx] = forceCur;
        //fs[idx] += forceSum;

    }

}

__global__ void computeEng_cu(int nAtoms, float4 *xs, float *perParticleEng, uint16_t *neighborCounts, uint *neighborlist, uint32_t *cumulSumMaxPerBlock, int warpSize, float *sigs, float *eps, float *rCuts, int numTypes, BoundsGPU bounds, float onetwoStr, float onethreeStr, float onefourStr) {
    float multipliers[4] = {1, onetwoStr, onethreeStr, onefourStr};
    extern __shared__ float paramsAll[];
    int sqrSize = numTypes*numTypes;
    float *sigs_shr = paramsAll;
    float *eps_shr = paramsAll + sqrSize;
    float *rCuts_shr = paramsAll + 2*sqrSize;
    copyToShared<float>(eps, eps_shr, sqrSize);
    copyToShared<float>(sigs, sigs_shr, sqrSize);
    copyToShared<float>(rCuts, rCuts_shr, sqrSize);
    __syncthreads();

    int idx = GETIDX();
    if (idx < nAtoms) {
        int baseIdx = baseNeighlistIdx(cumulSumMaxPerBlock, warpSize);
        float4 posWhole = xs[idx];
        int type = * (int *) &posWhole.w;
       // printf("type is %d\n", type);
        float3 pos = make_float3(posWhole);

        float sumEng = 0;

        int numNeigh = neighborCounts[idx];
        //printf("start, end %d %d\n", start, end);
        for (int i=0; i<numNeigh; i++) {
            int nlistIdx = baseIdx + warpSize * i;
            uint otherIdxRaw = neighborlist[nlistIdx];
            uint neighDist = otherIdxRaw >> 30;
            float multiplier = multipliers[neighDist];
            if (multiplier) {
                uint otherIdx = otherIdxRaw & EXCL_MASK;
                float4 otherPosWhole = xs[otherIdx];
                int otherType = * (int *) &otherPosWhole.w;
                float3 otherPos = make_float3(otherPosWhole);
                //then wrap and compute forces!
                float sig6 = squareVectorItem(sigs_shr, numTypes, type, otherType);
                float epstimes24 = squareVectorItem(eps_shr, numTypes, type, otherType);
                float3 dr = bounds.minImage(pos - otherPos);
                float lenSqr = lengthSqr(dr);
                //PRE-SQR THIS VALUE ON CPU
                float rCut = squareVectorItem(rCuts_shr, numTypes, type, otherType);
             //   printf("dist is %f %f %f\n", dr.x, dr.y, dr.z);
                if (lenSqr < rCut*rCut) {
                   // printf("mult is %f between idxs %d %d\n", multiplier, idx, otherIdx);
                    float r2inv = 1/lenSqr;
                    float r6inv = r2inv*r2inv*r2inv;
                    float sig6r6inv = sig6 * r6inv;
                    sumEng += 0.5 * 4*(epstimes24 / 24)*sig6r6inv*(sig6r6inv-1.0f) * multiplier; //0.5 b/c we need to half-count energy b/c pairs are redundant
                }

            }

        }   
        //printf("force %f %f %f with %d atoms \n", forceSum.x, forceSum.y, forceSum.z, end-start);
        perParticleEng[idx] += sumEng;
        //fs[idx] += forceSum;

    }

}
//__global__ void compute_cu(int nAtoms, cudaTextureObject_t xs, float4 *fs, int *neighborIdxs, cudaTextureObject_t neighborlist, float *sigs, float *eps, cudaTextureObject_t types, int numTypes, float rCut, BoundsGPU bounds) {
void FixLJCut::compute(bool computeVirials) {
    int nAtoms = state->atoms.size();
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    float *neighborCoefs = state->specialNeighborCoefs;

    compute_cu<<<NBLOCK(nAtoms), PERBLOCK, 3*numTypes*numTypes*sizeof(float)>>>(nAtoms, gpd.xs(activeIdx), gpd.fs(activeIdx), neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(), state->devManager.prop.warpSize, sigmas.getDevData(), epsilons.getDevData(), rCuts.getDevData(), numTypes, state->boundsGPU, neighborCoefs[0], neighborCoefs[1], neighborCoefs[2]);



}

void FixLJCut::singlePointEng(float *perParticleEng) {
    int nAtoms = state->atoms.size();
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    float *neighborCoefs = state->specialNeighborCoefs;

    computeEng_cu<<<NBLOCK(nAtoms), PERBLOCK, 3*numTypes*numTypes*sizeof(float)>>>(nAtoms, gpd.xs(activeIdx), perParticleEng, neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(), state->devManager.prop.warpSize, sigmas.getDevData(), epsilons.getDevData(), rCuts.getDevData(), numTypes, state->boundsGPU, neighborCoefs[0], neighborCoefs[1], neighborCoefs[2]);



}

bool FixLJCut::prepareForRun() {
    //loop through all params and fill with appropriate lambda function, then send all to device
    auto fillEps = [] (float a, float b) {
        return sqrt(a*b);
    };
    auto fillSig = [] (float a, float b) {
        return (a+b) / 2.0;
    };
    auto fillRCut = [this] (float a, float b) {
        return (float) std::fmax(a, b);
    };
    auto none = [] (float a){};

    auto fillRCutDiag = [this] () {
        return (float) state->rCut;
    };

    auto processEps = [] (float a) {
        return 24*a;
    };
    auto processSig = [] (float a) {
        return pow(a, 6);
    };
    auto processRCut = [] (float a) {
        return a*a;
    };
    prepareParameters(epsHandle, fillEps, processEps, false);
    prepareParameters(sigHandle, fillSig, processSig, false);
    prepareParameters(rCutHandle, fillRCut, processRCut, true, fillRCutDiag);
    sendAllToDevice();
    return true;
}

string FixLJCut::restartChunk(string format) {
    //test this
    stringstream ss;
    ss << "<" << restartHandle << ">\n";
    ss << restartChunkPairParams(format);
    ss << "</" << restartHandle << ">\n";
    return ss.str();
}

bool FixLJCut::readFromRestart(pugi::xml_node restData) {
    epsilons = xml_readNums<float>(restData, epsHandle);
    initializeParameters(epsHandle, epsilons);
    sigmas = xml_readNums<float>(restData, sigHandle);
    initializeParameters(sigHandle, sigmas);
    //add rcuts
    return true;

}
void FixLJCut::postRun() {
    resetToPreproc(sigHandle);
    resetToPreproc(epsHandle);
    resetToPreproc(rCutHandle);
}

void FixLJCut::addSpecies(string handle) {
    initializeParameters(epsHandle, epsilons);
    initializeParameters(sigHandle, sigmas);
    initializeParameters(rCutHandle, rCuts);

}

vector<float> FixLJCut::getRCuts() { //to be called after prepare.  These are squares now
    return LISTMAP(float, float, rc, rCuts.h_data, sqrt(rc));
}

void export_FixLJCut() {
    boost::python::class_<FixLJCut,
                          SHARED(FixLJCut),
                          boost::python::bases<FixPair>, boost::noncopyable > (
        "FixLJCut",
        boost::python::init<SHARED(State), string> (
            boost::python::args("state", "handle"))
    );

}
