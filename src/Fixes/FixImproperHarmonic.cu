#include "helpers.h"
#include "FixImproperHarmonic.h"
#include "FixHelpers.h"
#include "cutils_func.h"
#define SMALL 0.001f
#include "ImproperEvaluate.h"
namespace py = boost::python;
using namespace std;

const std::string improperHarmonicType = "ImproperHarmonic";
/*
__global__ void compute_cu(int nAtoms, float4 *xs, float4 *forces, cudaTextureObject_t idToIdxs, ImproperGPU *impropers, int *startstops, BoundsGPU bounds, ImproperHarmonicType *parameters, int nParameters) {


    int idx = GETIDX();
    extern __shared__ int all_shr[];
    int idxBeginCopy = startstops[blockDim.x*blockIdx.x];
    int idxEndCopy = startstops[min(nAtoms, blockDim.x*(blockIdx.x+1))];

    ImproperGPU *impropers_shr = (ImproperGPU *) all_shr;
    ImproperHarmonicType *parameters_shr = (ImproperHarmonicType *) (impropers_shr + (idxEndCopy - idxBeginCopy));
    copyToShared<ImproperGPU>(impropers + idxBeginCopy, impropers_shr, idxEndCopy - idxBeginCopy);
    copyToShared<ImproperHarmonicType>(parameters, parameters_shr, nParameters);

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
            
            int idxSelf = tex2D<int>(idToIdxs, XIDX(idSelf, sizeof(int)), YIDX(idSelf, sizeof(int)));
        
            float3 pos = make_float3(xs[idxSelf]);
           // printf("I am idx %d and I am evaluating atom with pos %f %f %f\n", idx, pos.x, pos.y, pos.z);
            float3 forceSum = make_float3(0, 0, 0);
            for (int i=0; i<n; i++) {
                ImproperGPU improper = impropers_shr[shr_idx + i];
                uint32_t typeFull = improper.type;
                myIdxInImproper = typeFull >> 29;
                int type = static_cast<int>((typeFull << 3) >> 3);   
                ImproperHarmonicType improperType = parameters_shr[type];
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
                    positions[toGet[i]] = make_float3(perAtomFromId(idToIdxs, xs, improper.ids[toGet[i]]));
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
                float dTheta = acosf(c) - improperType.thetaEq;

                float a = improperType.k * dTheta;
                a *= -2.0f / s;
                scValues[2] *= a;
                c *= a;
                float a11 = c * invLenSqrs[0] * scValues[0];
                float a22 = - invLenSqrs[1] * (2.0f * angleBits[0] * scValues[2] - c * (scValues[0] + scValues[1]));
                float a33 = c * invLenSqrs[2] * scValues[1];
                float a12 = -invLens[0] * invLens[1] * (angleBits[1] * c * scValues[0] + angleBits[2] * scValues[2]);
                float a13 = -invLens[0] * invLens[2] * scValues[2];
                float a23 = invLens[1] * invLens[2] * (angleBits[2] * c * scValues[1] + angleBits[1] * scValues[2]);

                float3 myForce = make_float3(0, 0, 0);
                float3 sFloat3 = make_float3(
                        a22*directors[1].x + a23*directors[2].x + a12*directors[0].x
                        ,  a22*directors[1].y + a23*directors[2].y + a12*directors[0].y
                        ,  a22*directors[1].z + a23*directors[2].z + a12*directors[0].z
                        );
                if (myIdxInImproper <= 1) {
                    float3 a11Dir1 = directors[0] * a11;
                    float3 a12Dir2 = directors[1] * a12;
                    float3 a13Dir3 = directors[2] * a13;
                    myForce.x += a11Dir1.x + a12Dir2.x + a13Dir3.x;
                    myForce.y += a11Dir1.y + a12Dir2.y + a13Dir3.y;
                    myForce.z += a11Dir1.z + a12Dir2.z + a13Dir3.z;

                    if (myIdxInImproper == 1) {
                        
                        myForce = -sFloat3 - myForce;
                    }
                  //      printf("improper idx 1 gets force %f %f %f\n", myForce.x, myForce.y, myForce.z);
                 //   } else {
                   //     printf("improper idx 0 gets force %f %f %f\n", myForce.x, myForce.y, myForce.z);
                  //  }
                } else {
                    float3 a13Dir1 = directors[0] * a13;
                    float3 a23Dir2 = directors[1] * a23;
                    float3 a33Dir3 = directors[2] * a33;
                    myForce.x += a13Dir1.x + a23Dir2.x + a33Dir3.x;
                    myForce.y += a13Dir1.y + a23Dir2.y + a33Dir3.y;
                    myForce.z += a13Dir1.z + a23Dir2.z + a33Dir3.z;
                    if (myIdxInImproper == 2) {
                        myForce = sFloat3 - myForce;
                   //     printf("improper idx 2 gets force %f %f %f\n", myForce.x, myForce.y, myForce.z);
                    }

                   // } else {
                   //     printf("improper idx 3 gets force %f %f %f\n", myForce.x, myForce.y, myForce.z);
                  //  }


                }
                forceSum += myForce;


            }
            forces[idxSelf] += forceSum;
        }
    }
}
*/

FixImproperHarmonic::FixImproperHarmonic(SHARED(State) state_, string handle)
    : FixPotentialMultiAtom (state_, handle, improperHarmonicType, true),
      pyListInterface(&forcers, &pyForcers) {}


void FixImproperHarmonic::compute(bool computeVirials) {
    int nAtoms = state->atoms.size();
    int activeIdx = state->gpd.activeIdx();
    compute_force_improper<<<NBLOCK(nAtoms), PERBLOCK, sizeof(ImproperGPU) * maxForcersPerBlock + forcers.size() * sizeof(ImproperHarmonicType)>>>(nAtoms, state->gpd.xs(activeIdx), state->gpd.fs(activeIdx), state->gpd.idToIdxs.getTex(), forcersGPU.data(), forcerIdxs.data(), state->boundsGPU, parameters.data(), parameters.size(), evaluator);

}
void FixImproperHarmonic::singlePointEng(float *perParticleEng) {
    int nAtoms = state->atoms.size();
    int activeIdx = state->gpd.activeIdx();
    compute_energy_improper<<<NBLOCK(nAtoms), PERBLOCK, sizeof(ImproperGPU) * maxForcersPerBlock + forcers.size() * sizeof(ImproperHarmonicType)>>>(nAtoms, state->gpd.xs(activeIdx), perParticleEng, state->gpd.idToIdxs.getTex(), forcersGPU.data(), forcerIdxs.data(), state->boundsGPU, parameters.data(), parameters.size(), evaluator);

}

void FixImproperHarmonic::createImproper(Atom *a, Atom *b, Atom *c, Atom *d, double k, double thetaEq, int type) {
    vector<Atom *> atoms = {a, b, c, d};
    validAtoms(atoms);
    if (type == -1) {
        assert(k!=COEF_DEFAULT and thetaEq!=COEF_DEFAULT);
    }
    forcers.push_back(ImproperHarmonic(a, b, c, d, k, thetaEq, type));
    pyListInterface.updateAppendedMember();
}
void FixImproperHarmonic::setImproperTypeCoefs(int type, double k, double thetaEq) {
    assert(thetaEq>=0);
    ImproperHarmonic dummy(k, thetaEq, type);
    setForcerType(type, dummy);
}




string FixImproperHarmonic::restartChunk(string format) {
    stringstream ss;

    return ss.str();
}

void export_FixImproperHarmonic() {

    boost::python::class_<FixImproperHarmonic,
                          SHARED(FixImproperHarmonic),
                          boost::python::bases<Fix, TypedItemHolder> > (
        "FixImproperHarmonic",
        boost::python::init<SHARED(State), string> (
                boost::python::args("state", "handle"))
    )
    .def("createImproper", &FixImproperHarmonic::createImproper,
            (boost::python::arg("k")=COEF_DEFAULT,
             boost::python::arg("thetaEq")=COEF_DEFAULT,
             boost::python::arg("type")=-1)
        )
    .def("setImproperTypeCoefs", &FixImproperHarmonic::setImproperTypeCoefs,
            (boost::python::arg("type")=COEF_DEFAULT,
             boost::python::arg("k")=COEF_DEFAULT,
             boost::python::arg("thetaEq")=COEF_DEFAULT
             )
        )
    ;

}

