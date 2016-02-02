
#include "helpers.h"
#include "FixAngleHarmonic.h"
#include "cutils_func.h"
__global__ void compute_cu(int nAtoms, cudaTextureObject_t xs, float4 *forces, cudaTextureObject_t idToIdxs, AngleHarmonicGPU *angles, int *startstops, BoundsGPU bounds) {
    int idx = GETIDX();
    extern __shared__ AngleHarmonicGPU angles_shr[];
    int idxBeginCopy = startstops[blockDim.x*blockIdx.x];
    int idxEndCopy = startstops[min(nAtoms, blockDim.x*(blockIdx.x+1))];
    copyToShared<AngleHarmonicGPU>(angles + idxBeginCopy, angles_shr, idxEndCopy - idxBeginCopy);
    __syncthreads();
    if (idx < nAtoms) {
  //      printf("going to compute %d\n", idx);
        int startIdx = startstops[idx];
        int endIdx = startstops[idx+1];
        //so start/end is the index within the entire bond list.
        //startIdx - idxBeginCopy gives my index in shared memory
        int shr_idx = startIdx - idxBeginCopy;
        int n = endIdx - startIdx;
        int idSelf = angles_shr[startIdx].ids[angles_shr[startIdx].myIdx];
        
        int idxSelf = tex2D<int>(idToIdxs, XIDX(idSelf, sizeof(int)), YIDX(idSelf, sizeof(int)));
    
        float3 pos = make_float3(float4FromIndex(xs, idxSelf));
        float3 forceSum = make_float3(0, 0, 0);
        for (int i=0; i<n; i++) {
            AngleHarmonicGPU angle = angles_shr[shr_idx + i];
            float3 positions[3];
            positions[angle.myIdx] = pos;
            int toGet[2];
            if (angle.myIdx==0) {
                toGet[0] = 1;
                toGet[1] = 2;
            } else if (angle.myIdx==1) {
                toGet[0] = 0;
                toGet[1] = 2;
            } else if (angle.myIdx==2) {
                toGet[0] = 0;
                toGet[1] = 1;
            }
            for (int i=0; i<2; i++) {
                positions[toGet[i]] = make_float3(perAtomFromId(idToIdxs, xs, angle.ids[toGet[i]]));
            }
            for (int i=1; i<3; i++) {
                positions[i] = positions[0] + bounds.minImage(positions[i]-positions[0]);
            }
            float3 directors[2];
            directors[0] = positions[0] - positions[1];
            directors[1] = positions[2] - positions[1];
            float distSqrs[2];
            float dists[2];
            for (int i=0; i<2; i++) {
                distSqrs[i] = lengthSqr(directors[i]);
                dists[i] = sqrtf(distSqrs[i]);
            }
            float c = dot(directors[0], directors[1]);
            float invDistProd = 1.0f / (dists[0]*dists[1]);
            c *= invDistProd;
            if (c>1) {
                c=1;
            } else if (c<-1) {
                c=-1;
            }
            float s = sqrtf(1-c*c);
            float dTheta = acosf(c) - angle.thetaEq;
            float forceConst = angle.k * dTheta;
            float a = -2 * forceConst * s;
            float a11 = a*c/distSqrs[0];
            float a12 = -a*invDistProd;
            float a22 = a*c/distSqrs[1];

            if (angle.myIdx==0) {
                forceSum += ((directors[0] * a11) + (directors[1] * a12)) * 0.5;
            } else if (angle.myIdx==1) {
                forceSum -= ((directors[0] * a11) + (directors[1] * a12) + (directors[1] * a22) + (directors[0] * a12)) * 0.5; 
            } else {
                forceSum += ((directors[1] * a22) + (directors[0] * a12)) * 0.5;
            }
        }
        forces[idxSelf] += forceSum;
    }
}


FixAngleHarmonic::FixAngleHarmonic(SHARED(State) state_, string handle) : FixPotentialMultiAtom(state_, handle, angleHarmType) {
}


void FixAngleHarmonic::compute() {
    int nAtoms = state->atoms.size();
    int activeIdx = state->gpd.activeIdx;
    compute_cu<<<NBLOCK(nAtoms), PERBLOCK, sizeof(AngleHarmonicGPU) * maxForcersPerBlock>>>(nAtoms, state->gpd.xs.getTex(), state->gpd.fs(activeIdx), state->gpd.idToIdxs.getTex(), forcersGPU.ptr, forcerIdxs.ptr, state->boundsGPU);

}

//void cumulativeSum(int *data, int n);
//okay, so the net result of this function is that two arrays (items, idxs of items) are on the gpu and we know how many bonds are in bondiest  block



void FixAngleHarmonic::createAngle(Atom *a, Atom *b, Atom *c, float k, float rEq) {
    vector<Atom *> atoms = {a, b, c};
    validAtoms(atoms);
    forcers.push_back(AngleHarmonic(a, b, c, k, rEq));
    std::array<int, 3> angleIds = {a->id, b->id, c->id};
    forcerAtomIds.push_back(angleIds);
}
string FixAngleHarmonic::restartChunk(string format) {
    stringstream ss;

    return ss.str();
}

void export_FixAngleHarmonic() {
    class_<FixAngleHarmonic, SHARED(FixAngleHarmonic), bases<Fix> > ("FixAngleHarmonic", init<SHARED(State), string> (args("state", "handle")))
        .def("createAngle", &FixAngleHarmonic::createAngle)
        ;

}

