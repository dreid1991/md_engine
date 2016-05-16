#include "helpers.h"
#include "FixDihedralOPLS.h"
#include "FixHelpers.h"
#include "cutils_func.h"

#define EPSILON 0.00001f
//using namespace boost::python;
namespace py = boost::python;
const std::string dihedralOPLSType = "DihedralOPLS";
template <class DIHEDRALGPU, class DIHEDRALTYPE> //don't need DIHEDRALGPU, are all DihedralGPU.  Worry about later 
__global__ void compute_cu(int nAtoms, float4 *xs, float4 *forces, cudaTextureObject_t idToIdxs, DIHEDRALGPU *dihedrals, int *startstops, BoundsGPU bounds, DIHEDRALTYPE *parameters, int nParameters) {


    int idx = GETIDX();
    extern __shared__ int all_shr[];
    int idxBeginCopy = startstops[blockDim.x*blockIdx.x];
    int idxEndCopy = startstops[min(nAtoms, blockDim.x*(blockIdx.x+1))];
    DIHEDRALGPU *dihedrals_shr = (DIHEDRALGPU *) all_shr;
    DIHEDRALTYPE *parameters_shr = (DIHEDRALTYPE *) (dihedrals_shr + (idxEndCopy - idxBeginCopy));
    copyToShared<DIHEDRALGPU>(dihedrals + idxBeginCopy, dihedrals_shr, idxEndCopy - idxBeginCopy);
    copyToShared<DIHEDRALTYPE>(parameters, parameters_shr, nParameters);
    __syncthreads();
    if (idx < nAtoms) {
  //      printf("going to compute %d\n", idx);
        int startIdx = startstops[idx];
        int endIdx = startstops[idx+1];
        //so start/end is the index within the entire bond list.
        //startIdx - idxBeginCopy gives my index in shared memory
        int shr_idx = startIdx - idxBeginCopy;
        int n = endIdx - startIdx;
        if (n) {
            int myIdxInDihedral = dihedrals_shr[shr_idx].type >> 29;
            int idSelf = dihedrals_shr[shr_idx].ids[myIdxInDihedral];
            
            int idxSelf = tex2D<int>(idToIdxs, XIDX(idSelf, sizeof(int)), YIDX(idSelf, sizeof(int)));
        
            float3 pos = make_float3(xs[idxSelf]);
           // printf("I am idx %d and I am evaluating atom with pos %f %f %f\n", idx, pos.x, pos.y, pos.z);
            float3 forceSum = make_float3(0, 0, 0);
            for (int i=0; i<n; i++) {
                DIHEDRALGPU dihedral = dihedrals_shr[shr_idx + i];
                uint32_t typeFull = dihedral.type;
                myIdxInDihedral = typeFull >> 29;
                int type = static_cast<int>((typeFull << 3) >> 3);
                DIHEDRALTYPE dihedralType = parameters_shr[type];

                float3 positions[4];
                positions[myIdxInDihedral] = pos;
                int toGet[3];
                if (myIdxInDihedral==0) {
                    toGet[0] = 1;
                    toGet[1] = 2;
                    toGet[2] = 3;
                } else if (myIdxInDihedral==1) {
                    toGet[0] = 0;
                    toGet[1] = 2;
                    toGet[2] = 3;
                } else if (myIdxInDihedral==2) {
                    toGet[0] = 0;
                    toGet[1] = 1;
                    toGet[2] = 3;
                } else if (myIdxInDihedral==3) {
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


                float c0 = dot(directors[0], directors[2]) * invLens[0] * invLens[2];
             //   printf("c0 is %f\n", c0);
                float c12Mags[2];
                float invMagProds[2]; //r12c1, 2 in lammps
                for (int i=0; i<2; i++) {
                    float dotProd = dot(directors[i+1], directors[i]);
                    if (i==1) {
                        dotProd *= -1;
                    }
              //      printf("ctmp is %f\n", dotProd);
                    invMagProds[i] = invLens[i] * invLens[i+1];
                    c12Mags[i] = dotProd * invMagProds[i]; //lammps variable names are opaque
              //      printf("c12 mag %d %f\n", i, c12Mags[i]);
                }

                float scValues[3]; //???, is s1, s2, s12 in lammps
                for (int i=0; i<2; i++) {
                    float x = max(1 - c12Mags[i]*c12Mags[i], 0.0f);
                    float sqrtVal = max(sqrtf(x), EPSILON);
                    scValues[i] = 1.0 / sqrtVal;
                }
                scValues[2] = scValues[0] * scValues[1];


                for (int i=0; i<2; i++) {
                    scValues[i] *= scValues[i]; 
                }
             //   printf("sc values %f %f %f\n", scValues[0], scValues[1], scValues[2]);
                float c = (c0 + c12Mags[0]*c12Mags[1]) * scValues[2];

                float3 cVector;
                cVector.x = directors[0].y*directors[1].z - directors[0].z*directors[1].y;
                cVector.y = directors[0].z*directors[1].x - directors[0].x*directors[1].z;
                cVector.z = directors[0].x*directors[1].y - directors[0].y*directors[1].x;
                float cVectorLen = length(cVector);
                float dx = dot(cVector, directors[2]) * invLens[2] / cVectorLen;
            //    printf("c is %f\n", c);
                if (c > 1.0f) {
                    c = 1.0f;
                } else if (c < -1.0f) {
                    c = -1.0f;
                }
                float phi = acosf(c);
                //printf("phi is %f\n", phi);
                if (dx < 0) {
                    phi = -phi;
                }
            //    printf("phi is %f\n", phi);
                float sinPhi = sinf(phi);
                float absSinPhi = sinPhi < 0 ? -sinPhi : sinPhi;
                if (absSinPhi < EPSILON) {
                    sinPhi = EPSILON;
                }
                float invSinPhi = 1.0f / sinPhi;

                float derivOfPotential = 0.5 * (
                             dihedralType.coefs[0] 
                    - 2.0f * dihedralType.coefs[1] * sinf(2.0f*phi) * invSinPhi
                    + 3.0f * dihedralType.coefs[2] * sinf(3.0f*phi) * invSinPhi
                    - 4.0f * dihedralType.coefs[3] * sinf(4.0f*phi) * invSinPhi
                    )
                    ;
            //    printf("deriv is %f\n", derivOfPotential);
            //    printf("coefs %f %f %f %f\n", dihedral.coefs[0],dihedral.coefs[1] , dihedral.coefs[2] , dihedral.coefs[3]);




                c *= derivOfPotential;
                scValues[2] *= derivOfPotential;
                float a11 = c * invLenSqrs[0] * scValues[0];
                float a22 = -invLenSqrs[1] * (2.0f*c0*scValues[2] - c*(scValues[0]+scValues[1]));
                float a33 = c*invLenSqrs[2]*scValues[1];
                float a12 = -invMagProds[0] * (c12Mags[0] * c * scValues[0] + c12Mags[1] * scValues[2]);
                float a13 = -invLens[0] * invLens[2] * scValues[2];
                float a23 = invMagProds[1] * (c12Mags[1]*c*scValues[1] + c12Mags[0]*scValues[2]);
                float3 myForce = make_float3(0, 0, 0);
                float3 sFloat3 = make_float3(
                        a12*directors[0].x + a22*directors[1].x + a23*directors[2].x
                        ,  a12*directors[0].y + a22*directors[1].y + a23*directors[2].y
                        ,  a12*directors[0].z + a22*directors[1].z + a23*directors[2].z
                        );
                //printf("ssomething valyes %f %f %f\n", sFloat3.x, sFloat3.y, sFloat3.z);
                //printf("comps %f %f %f %f %f %f\n", a12, directors[0].x,  a22, directors[1].x,  a23, directors[2].x);

                if (myIdxInDihedral <= 1) {
                    float3 a11Dir1 = directors[0] * a11;
                    float3 a12Dir2 = directors[1] * a12;
                    float3 a13Dir3 = directors[2] * a13;
                    myForce.x += a11Dir1.x + a12Dir2.x + a13Dir3.x;
                    myForce.y += a11Dir1.y + a12Dir2.y + a13Dir3.y;
                    myForce.z += a11Dir1.z + a12Dir2.z + a13Dir3.z;

                    if (myIdxInDihedral == 1) {
                        
                        myForce = -sFloat3 - myForce;
                  //      printf("dihedral idx 1 gets force %f %f %f\n", myForce.x, myForce.y, myForce.z);
                    }
                    //} else {
                     //   printf("dihedral idx 0 gets force %f %f %f\n", myForce.x, myForce.y, myForce.z);
                    //}
                } else {
                    float3 a13Dir1 = directors[0] * a13;
                    float3 a23Dir2 = directors[1] * a23;
                    float3 a33Dir3 = directors[2] * a33;
                    myForce.x += a13Dir1.x + a23Dir2.x + a33Dir3.x;
                    myForce.y += a13Dir1.y + a23Dir2.y + a33Dir3.y;
                    myForce.z += a13Dir1.z + a23Dir2.z + a33Dir3.z;
                    if (myIdxInDihedral == 2) {
                        myForce = sFloat3 - myForce;
                   //     printf("dihedral idx 2 gets force %f %f %f\n", myForce.x, myForce.y, myForce.z);
                    }

                   // } else {
                   //     printf("dihedral idx 3 gets force %f %f %f\n", myForce.x, myForce.y, myForce.z);
                    //}


                }
                forceSum += myForce;


            }
            forces[idxSelf] += forceSum;
        }
    }
}


FixDihedralOPLS::FixDihedralOPLS(SHARED(State) state_, string handle) : FixPotentialMultiAtom (state_, handle, dihedralOPLSType, true), pyListInterface(&forcers, &pyForcers) {}


void FixDihedralOPLS::compute(bool computeVirials) {
    int nAtoms = state->atoms.size();
    int activeIdx = state->gpd.activeIdx();


    //cout << "max forcers " << maxForcersPerBlock << endl;

    compute_cu<<<NBLOCK(nAtoms), PERBLOCK, sizeof(DihedralGPU) * maxForcersPerBlock + sizeof(DihedralOPLSType) * parameters.size() >>>(nAtoms, state->gpd.xs(activeIdx), state->gpd.fs(activeIdx), state->gpd.idToIdxs.getTex(), forcersGPU.data(), forcerIdxs.data(), state->boundsGPU, parameters.data(), parameters.size());

}


void FixDihedralOPLS::createDihedral(Atom *a, Atom *b, Atom *c, Atom *d, double v1, double v2, double v3, double v4, int type) {
    double vs[4] = {v1, v2, v3, v4};
    if (type==-1) {
        for (int i=0; i<4; i++) {
            assert(vs[i] != COEF_DEFAULT);
        }
    }
    forcers.push_back(DihedralOPLS(a, b, c, d, vs, type));
    pyListInterface.updateAppendedMember();
}


void FixDihedralOPLS::createDihedralPy(Atom *a, Atom *b, Atom *c, Atom *d, py::list coefs, int type) {
    double coefs_c[4];
    if (type!=-1) {
        createDihedral(a, b, c, d, COEF_DEFAULT, COEF_DEFAULT, COEF_DEFAULT, COEF_DEFAULT, type);
    } else {
        assert(len(coefs) == 4);
        for (int i=0; i<4; i++) {
            py::extract<double> coef(coefs[i]);
            assert(coef.check());
            coefs_c[i] = coef;
        }
        createDihedral(a, b, c, d, coefs_c[0], coefs_c[1], coefs_c[2], coefs_c[3], type);

    }
}

void FixDihedralOPLS::setDihedralTypeCoefs(int type, py::list coefs) {
    assert(len(coefs)==4);
    double coefs_c[4];
    for (int i=0; i<4; i++) {
        py::extract<double> coef(coefs[i]);
        assert(coef.check());
        coefs_c[i] = coef;
    }

    DihedralOPLS dummy(coefs_c, type);
    setForcerType(type, dummy);
}



string FixDihedralOPLS::restartChunk(string format) {
    stringstream ss;

    return ss.str();
}

void export_FixDihedralOPLS() {
    py::class_<FixDihedralOPLS,
                          SHARED(FixDihedralOPLS),
                          py::bases<Fix, TypedItemHolder> > (
        "FixDihedralOPLS",
        py::init<SHARED(State), string> (
            py::args("state", "handle")
        )
    )
    .def("createDihedral", &FixDihedralOPLS::createDihedralPy,
            (py::arg("coefs")=py::list(),
             py::arg("type")=-1)
        )

    .def("setDihedralTypeCoefs", &FixDihedralOPLS::setDihedralTypeCoefs, 
            (py::arg("type"), 
             py::arg("coefs"))
            )
    .def_readonly("dihedrals", &FixDihedralOPLS::pyForcers)

    ;

}

