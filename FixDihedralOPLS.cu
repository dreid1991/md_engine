#include "helpers.h"
#include "FixDihedralOPLS.h"
#include "cutils_func.h"
__global__ void compute_cu(int nAtoms, cudaTextureObject_t xs, float4 *forces, cudaTextureObject_t idToIdxs, DihedralOPLSGPU *dihedrals, int *startstops, BoundsGPU bounds) {
    int idx = GETIDX();
    extern __shared__ DihedralOPLSGPU dihedrals_shr[];
    int idxBeginCopy = startstops[blockDim.x*blockIdx.x];
    int idxEndCopy = startstops[min(nAtoms, blockDim.x*(blockIdx.x+1))];
    copyToShared<DihedralOPLSGPU>(dihedrals + idxBeginCopy, dihedrals_shr, idxEndCopy - idxBeginCopy);
    __syncthreads();
    if (idx < nAtoms) {
  //      printf("going to compute %d\n", idx);
        int startIdx = startstops[idx];
        int endIdx = startstops[idx+1];
        //so start/end is the index within the entire bond list.
        //startIdx - idxBeginCopy gives my index in shared memory
        int shr_idx = startIdx - idxBeginCopy;
        int n = endIdx - startIdx;
        int idSelf = dihedrals_shr[startIdx].ids[dihedrals_shr[startIdx].myIdx];
        
        int idxSelf = tex2D<int>(idToIdxs, XIDX(idSelf, sizeof(int)), YIDX(idSelf, sizeof(int)));
    
        float3 pos = make_float3(float4FromIndex(xs, idxSelf));
        float3 forceSum = make_float3(0, 0, 0);
        for (int i=0; i<n; i++) {
            DihedralOPLSGPU dihedral= dihedrals_shr[shr_idx + i];
            float3 positions[4];
            positions[dihedral.myIdx] = pos;
            int toGet[3];
            if (dihedral.myIdx==0) {
                toGet[0] = 1;
                toGet[1] = 2;
                toGet[2] = 3;
            } else if (dihedral.myIdx==1) {
                toGet[0] = 0;
                toGet[1] = 2;
                toGet[2] = 3;
            } else if (dihedral.myIdx==2) {
                toGet[0] = 0;
                toGet[1] = 1;
                toGet[2] = 3;
            } else if (dihedral.myIdx==3) {
                toGet[0] = 0;
                toGet[1] = 1;
                toGet[2] = 2;
            }
            for (int i=0; i<3; i++) {
                positions[toGet[i]] = make_float3(perAtomFromId(idToIdxs, xs, dihedral.ids[toGet[i]]));
            }
            for (int i=1; i<3; i++) {
                positions[i] = positions[0] + bounds.minImage(positions[i]-positions[0]);
            }
            float3 directors[3]; //vb_xyz in lammps
            float lenSqrs[3];
            float lens[3];
            float invLenSqrs[3]; //sb in lammps
            directors[0] = positions[0] - positions[1];
            directors[1] = positions[2] - positions[1];
            directors[2] = positions[3] - positions[2];

            for (int i=0; i<3; i++) {
                lenSqrs[i] = lengthSqr(directors[i]);
                lens[i] = sqrtf(lenSqrs[i]);
                invLenSqrs[i] = 1.0f / lenSqrs[i];
            }

            float invLenBonds13[2];
            invLenBonds13[0] = 1.0f / lens[0];
            invLenBonds13[1] = 1.0f / lens[2];

            float c0 = dot(directors[0], directors[2]) * invLenBonds13[0] * invLenBonds13[1];
            /*


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
            */
        }
        forces[idxSelf] += forceSum;
    }
}


FixDihedralOPLS::FixDihedralOPLS(SHARED(State) state_, string handle) : FixPotentialMultiAtom (state_, handle, dihedralOPLSType) {
}


void FixDihedralOPLS::compute() {
    int nAtoms = state->atoms.size();
    int activeIdx = state->gpd.activeIdx;
    compute_cu<<<NBLOCK(nAtoms), PERBLOCK, sizeof(DihedralOPLSGPU) * maxForcersPerBlock>>>(nAtoms, state->gpd.xs.getTex(), state->gpd.fs(activeIdx), state->gpd.idToIdxs.getTex(), forcersGPU.ptr, forcerIdxs.ptr, state->boundsGPU);

}


void FixDihedralOPLS::createDihedral(Atom *a, Atom *b, Atom *c, Atom *d, double v1, double v2, double v3, double v4) {
    double vs[4] = {v1, v2, v3, v4};
    forcers.push_back(DihedralOPLS(a, b, c, d, vs));
    std::array<int, 4> ids = {a->id, b->id, c->id, d->id};
    forcerAtomIds.push_back(ids);
}



string FixDihedralOPLS::restartChunk(string format) {
    stringstream ss;

    return ss.str();
}

void export_FixDihedralOPLS() {
    class_<FixDihedralOPLS, SHARED(FixDihedralOPLS), bases<Fix> > ("FixDihedralOPLS", init<SHARED(State), string> (args("state", "handle")))
        .def("createDihedral", &FixDihedralOPLS::createDihedral)
        ;

}

