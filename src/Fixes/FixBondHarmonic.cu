#include "helpers.h"
#include "FixBondHarmonic.h"
#include "cutils_func.h"
#include "FixHelpers.h"

namespace py = boost::python;
using namespace std;

const std::string bondHarmonicType = "BondHarmonic";

__global__ void compute_cu(int nAtoms, float4 *xs, float4 *forces, cudaTextureObject_t idToIdxs, BondHarmonicGPU *bonds, int *startstops, BoundsGPU bounds) {
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


FixBondHarmonic::FixBondHarmonic(SHARED(State) state_, string handle)
    : FixBond(state_, handle, string("None"), bondHarmonicType, true, 1),
      pyListInterface(&bonds, &pyBonds) {}



//void cumulativeSum(int *data, int n);
//okay, so the net result of this function is that two arrays (items, idxs of items) are on the gpu and we know how many bonds are in bondiest  block

   


void FixBondHarmonic::createBond(Atom *a, Atom *b, double k, double rEq, int type) {
    vector<Atom *> atoms = {a, b};
    validAtoms(atoms);
    if (type == -1) {
        assert(k!=-1 and rEq!=-1);
    }
    bonds.push_back(BondHarmonic(a, b, k, rEq, type));
    pyListInterface.updateAppendedMember();
    
}

void FixBondHarmonic::setBondTypeCoefs(int type, double k, double rEq) {
    assert(rEq>=0);
    BondHarmonic dummy(k, rEq, type);
    setForcerType(type, dummy);
}

void FixBondHarmonic::compute(bool computeVirials) {
    int nAtoms = state->atoms.size();
    int activeIdx = state->gpd.activeIdx();
    //cout << "Max bonds per block is " << maxBondsPerBlock << endl;
    compute_cu<<<NBLOCK(nAtoms), PERBLOCK, sizeof(BondHarmonicGPU) * maxBondsPerBlock>>>(nAtoms, state->gpd.xs(activeIdx), state->gpd.fs(activeIdx), state->gpd.idToIdxs.getTex(), bondsGPU.data(), bondIdxs.data(), state->boundsGPU);

}


/*
vector<pair<int, vector<int> > > FixBondHarmonic::neighborlistExclusions() {
    map<int, vector<int> > exclusions;
    for (BondHarmonic &b : bonds) {
        int ids[2];
        for (int i=0; i<2; i++) {
            ids[i] = b.atoms[i]->id;
        }
        for (int i=0; i<2; i++) {
            auto it = exclusions.find(ids[i]);
            if (it == exclusions.end()) {
                vector<int> vals(1);
                vals[0] = ids[!i];
                exclusions[ids[i]] = vals;
            } else {
                it->second.push_back(ids[!i]);
            }
        }
    }
    vector<pair<int, vector<int> > > res;
    for (auto it = exclusions.begin(); it != exclusions.end(); it++) {
        res.push_back(make_pair(it->first, it->second));
    }
    sort(res.begin(), res.end(), [] (const pair<int, vector<int> > &a, const pair<int, vector<int> > &b) { return a.first > b.first;});
    return res;
}
*/

string FixBondHarmonic::restartChunk(string format) {
    stringstream ss;
    ss << "<" << restartHandle << ">\n";
    for (BondVariant &bv : bonds) {
        BondHarmonic &b = get<BondHarmonic>(bv);
        //ss << b.atoms[0]->id << " " << b.atoms[1]->id << " " << b.k << " " << b.rEq << "\n";
    }
    ss << "</" << restartHandle << ">\n";
    //NOT DONE
    cout << "BOND REST CHUNK NOT DONE" << endl;
    return ss.str();
}

void export_FixBondHarmonic() {
  

  
    py::class_<FixBondHarmonic, SHARED(FixBondHarmonic), py::bases<Fix, TypedItemHolder> >
    (
        "FixBondHarmonic", py::init<SHARED(State), string> (py::args("state", "handle"))
    )
    .def("createBond", &FixBondHarmonic::createBond,
            (py::arg("k")=-1,
             py::arg("rEq")=-1,
             py::arg("type")=-1)
        )
    .def("setBondTypeCoefs", &FixBondHarmonic::setBondTypeCoefs,
            (py::arg("type"),
             py::arg("k"),
             py::arg("rEq"))
        )
    .def_readonly("bonds", &FixBondHarmonic::pyBonds)    
    ;

}
