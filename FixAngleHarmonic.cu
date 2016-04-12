
#include "FixHelpers.h"
#include "helpers.h"
#include "FixAngleHarmonic.h"
#include "cutils_func.h"
#define SMALL 0.0001f
namespace py = boost::python;
__global__ void compute_cu(int nAtoms, float4 *xs, float4 *forces, cudaTextureObject_t idToIdxs, AngleHarmonicGPU *angles, int *startstops, BoundsGPU bounds) {
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
        if (n>0) {
            int idSelf = angles_shr[shr_idx].ids[angles_shr[shr_idx].myIdx];

            int idxSelf = tex2D<int>(idToIdxs, XIDX(idSelf, sizeof(int)), YIDX(idSelf, sizeof(int)));
            float3 pos = make_float3(xs[idxSelf]);
            //float3 pos = make_float3(float4FromIndex(xs, idxSelf));
            float3 forceSum = make_float3(0, 0, 0);
            for (int i=0; i<n; i++) {
             //   printf("ANGLE! %d\n", i);
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
             //   printf("position Xs %f %f %f\n", positions[0].x, positions[1].x, positions[2].x);
              //  printf("director Xs %f %f\n", directors[0].x, directors[1].x);
                float distSqrs[2];
                float dists[2];
                for (int i=0; i<2; i++) {
                    distSqrs[i] = lengthSqr(directors[i]);
                    dists[i] = sqrtf(distSqrs[i]);
                }
                float c = dot(directors[0], directors[1]);
             //   printf("prenorm c is %f\n", c);
                float invDistProd = 1.0f / (dists[0]*dists[1]);
              //  printf("inv dist is %f\n", invDistProd);
                c *= invDistProd;
              //  printf("c is %f\n", c);
                if (c>1) {
                    c=1;
                } else if (c<-1) {
                    c=-1;
                }
                float s = sqrtf(1-c*c);
                if (s < SMALL) {
                    s = SMALL;
                }
                s = 1.0f / s;
                float dTheta = acosf(c) - angle.thetaEq;
              //  printf("%f %f\n", acosf(c), angle.thetaEq);
                float forceConst = angle.k * dTheta;
                float a = -2.0f * forceConst * s;
                float a11 = a*c/distSqrs[0];
                float a12 = -a*invDistProd;
                float a22 = a*c/distSqrs[1];
             //   printf("forceConst %f a %f s %f dists %f %f %f\n", forceConst, a, s, a11, a12, a22);

                if (angle.myIdx==0) {
                    forceSum += ((directors[0] * a11) + (directors[1] * a12)) * 0.5;
                } else if (angle.myIdx==1) {
                    forceSum -= ((directors[0] * a11) + (directors[1] * a12) + (directors[1] * a22) + (directors[0] * a12)) * 0.5; 
                } else {
                    forceSum += ((directors[1] * a22) + (directors[0] * a12)) * 0.5;
                }
             //   printf("%f %f %f\n", forceSum.x, forceSum.y, forceSum.z);
            }
            float4 curForce = forces[idxSelf];
         //   printf("Final force is %f %f %f\n", forceSum.x, forceSum.y, forceSum.z);
            curForce += forceSum;
            forces[idxSelf] = curForce;
        }
    }
}


FixAngleHarmonic::FixAngleHarmonic(SHARED(State) state_, string handle) : FixPotentialMultiAtom(state_, handle, angleHarmType) {
    forceSingle = true;
}


void FixAngleHarmonic::compute(bool computeVirials) {
    int nAtoms = state->atoms.size();
    int activeIdx = state->gpd.activeIdx();
    compute_cu<<<NBLOCK(nAtoms), PERBLOCK, sizeof(AngleHarmonicGPU) * maxForcersPerBlock>>>(nAtoms, state->gpd.xs(activeIdx), state->gpd.fs(activeIdx), state->gpd.idToIdxs.getTex(), forcersGPU.data(), forcerIdxs.data(), state->boundsGPU);

}

//void cumulativeSum(int *data, int n);
//okay, so the net result of this function is that two arrays (items, idxs of items) are on the gpu and we know how many bonds are in bondiest  block

void FixAngleHarmonic::setAngleTypeCoefs(int type, double k, double thetaEq) {
    //cout << type << " " << k << " " << thetaEq << endl;
    assert(thetaEq>=0);
    AngleHarmonic dummy(k, thetaEq);
    setForcerType(type, dummy);
}

void FixAngleHarmonic::createAngle(Atom *a, Atom *b, Atom *c, double k, double thetaEq, int type) {
    vector<Atom *> atoms = {a, b, c};
    validAtoms(atoms);
    if (type == -1) {
        assert(k!=COEF_DEFAULT and thetaEq!=COEF_DEFAULT);
    }
    forcers.push_back(AngleHarmonic(a, b, c, k, thetaEq, type));
    std::array<int, 3> angleIds = {a->id, b->id, c->id};
    forcerAtomIds.push_back(angleIds);
}
string FixAngleHarmonic::restartChunk(string format) {
    stringstream ss;

    return ss.str();
}

void export_FixAngleHarmonic() {
    boost::python::class_<FixAngleHarmonic,
                          SHARED(FixAngleHarmonic),
                          boost::python::bases<Fix, TypedItemHolder> > (
        "FixAngleHarmonic",
        boost::python::init<SHARED(State), string> (
                                        boost::python::args("state", "handle"))
    )
    .def("createAngle", &FixAngleHarmonic::createAngle,
            (boost::python::arg("k")=COEF_DEFAULT,
             boost::python::arg("thetaEq")=COEF_DEFAULT,
             boost::python::arg("type")=-1)
        )
    .def("setAngleTypeCoefs", &FixAngleHarmonic::setAngleTypeCoefs,
            (boost::python::arg("type")=-1,
             boost::python::arg("k")=COEF_DEFAULT,
             boost::python::arg("thetaEq")=COEF_DEFAULT
            )
        )
    ;

}

