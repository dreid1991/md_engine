#include "FixLJCut.h"
#include "State.h"
#include "cutils_func.h"
FixLJCut::FixLJCut(SHARED(State) state_, string handle_, string groupHandle_) : FixPair(state_, handle_, groupHandle_, LJCutType, 1), epsHandle("eps"), sigHandle("sig"), rCutHandle("rCut") {
    initializeParameters(epsHandle, epsilons);
    initializeParameters(sigHandle, sigmas);
    initializeParameters(rCutHandle, rCuts);
    forceSingle = true;

}



__global__ void compute_cu(int nAtoms, float4 *xs, float4 *fs, int *neighborCounts, uint *neighborlist, int *cumulSumMaxPerBlock, int warpSize, float *sigs, float *eps, float *rCuts, int numTypes,  BoundsGPU bounds, float onetwoStr, float onethreeStr, float onefourStr) {
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
        int baseIdx = baseNeighlistIdx<void>(cumulSumMaxPerBlock, warpSize);
        float4 posWhole = xs[idx];
        int type = * (int *) &posWhole.w;
       // printf("type is %d\n", type);
        float3 pos = make_float3(posWhole);

        float3 forceSum = make_float3(0, 0, 0);

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
                float sig = squareVectorItem(sigs_shr, numTypes, type, otherType);
                float eps = squareVectorItem(eps_shr, numTypes, type, otherType);
                float3 dr = bounds.minImage(pos - otherPos);
                float lenSqr = lengthSqr(dr);
                //PRE-SQR THIS VALUE ON CPU
                float rCut = squareVectorItem(rCuts_shr, numTypes, type, otherType);
                //if (threadIdx.x == 100) {
                //    printf("%f %f %f\n", sig, eps, rCut);
               // }
             //   printf("dist is %f %f %f\n", dr.x, dr.y, dr.z);
                if (lenSqr < rCut*rCut) {
                   // printf("mult is %f between idxs %d %d\n", multiplier, idx, otherIdx);
                    //HEY - PRECOMPUTE THESE VALUES ON CPU IN PREPARE FOR RUN
                    float sig6 = powf(sig, 6);//compiler should optimize this 
                    float p1 = eps*48*sig6*sig6;
                    float p2 = eps*24*sig6;
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

__global__ void computeEng_cu(int nAtoms, float4 *xs, float *perParticleEng, int *neighborCounts, uint *neighborlist, int *cumulSumMaxPerBlock, int warpSize, float *sigs, float *eps, float *rCuts, int numTypes, BoundsGPU bounds, float onetwoStr, float onethreeStr, float onefourStr) {
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
        int baseIdx = baseNeighlistIdx<void>(cumulSumMaxPerBlock, warpSize);
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
                float sig = squareVectorItem(sigs_shr, numTypes, type, otherType);
                float eps = squareVectorItem(eps_shr, numTypes, type, otherType);
                float3 dr = bounds.minImage(pos - otherPos);
                float lenSqr = lengthSqr(dr);
                //PRE-SQR THIS VALUE ON CPU
                float rCut = squareVectorItem(rCuts_shr, numTypes, type, otherType);
             //   printf("dist is %f %f %f\n", dr.x, dr.y, dr.z);
                if (lenSqr < rCut*rCut) {
                   // printf("mult is %f between idxs %d %d\n", multiplier, idx, otherIdx);
                    float sig6 = powf(sig, 6);//compiler should optimize this 
                    float r2inv = 1/lenSqr;
                    float r6inv = r2inv*r2inv*r2inv;
                    float sig6r6inv = sig6 * r6inv;
                    sumEng += 0.5 * 4*eps*sig6r6inv*(sig6r6inv-1.0f) * multiplier; //0.5 b/c we need to half-count energy b/c pairs are redundant
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
    int activeIdx = gpd.activeIdx;
    int *neighborCounts = grid.perAtomArray.d_data.data();
    float *neighborCoefs = state->specialNeighborCoefs;

    SAFECALL((compute_cu<<<NBLOCK(nAtoms), PERBLOCK, 3*numTypes*numTypes*sizeof(float)>>>(nAtoms, gpd.xs(activeIdx), gpd.fs(activeIdx), neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(), state->devManager.prop.warpSize, sigmas.getDevData(), epsilons.getDevData(), rCuts.getDevData(), numTypes, state->boundsGPU, neighborCoefs[0], neighborCoefs[1], neighborCoefs[2])), "CUT");



}

void FixLJCut::singlePointEng(float *perParticleEng) {
    int nAtoms = state->atoms.size();
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx;
    int *neighborCounts = grid.perAtomArray.d_data.data();
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
        cout << "RETURNING " << state->rCut << endl;
        return (float) state->rCut;
    };

    prepareParameters(epsHandle, fillEps, false);
    prepareParameters(sigHandle, fillSig, false);
    prepareParameters(rCutHandle, fillRCut, true, fillRCutDiag);
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
    
    vector<float> epsilons_raw = xml_readNums<float>(restData, epsHandle);
    epsilons.set(epsilons_raw);
    initializeParameters(epsHandle, epsilons);
    vector<float> sigmas_raw = xml_readNums<float>(restData, sigHandle);
    sigmas.set(sigmas_raw);
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

vector<float> FixLJCut::getRCuts() {
    return rCuts.h_data;
}

void export_FixLJCut() {
    boost::python::class_<FixLJCut,
                          SHARED(FixLJCut),
                          boost::python::bases<FixPair>, boost::noncopyable > (
        "FixLJCut",
        boost::python::init<SHARED(State), string, string> (
            boost::python::args("state", "handle", "groupHandle"))
    );

}
