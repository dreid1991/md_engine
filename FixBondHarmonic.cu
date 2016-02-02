#include "helpers.h"
#include "FixBondHarmonic.h"
#include "cutils_func.h"
#include "FixHelpers.h"
__global__ void compute_cu(int nAtoms, cudaTextureObject_t xs, float4 *forces, cudaTextureObject_t idToIdxs, BondHarmonicGPU *bonds, int *startstops, BoundsGPU bounds) {
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
        int idSelf = bonds_shr[startIdx].myId;
        
        int idxSelf = tex2D<int>(idToIdxs, XIDX(idSelf, sizeof(int)), YIDX(idSelf, sizeof(int)));

        float3 pos = make_float3(tex2D<float4>(xs, XIDX(idxSelf, sizeof(float4)), YIDX(idxSelf, sizeof(float4))));
        float3 forceSum = make_float3(0, 0, 0);
        for (int i=0; i<n; i++) {
            BondHarmonicGPU b = bonds_shr[shr_idx + i];
            int idOther = b.idOther;
            int idxOther = tex2D<int>(idToIdxs, XIDX(idOther, sizeof(int)), YIDX(idOther, sizeof(int)));

            float3 posOther = make_float3(tex2D<float4>(xs, XIDX(idxOther, sizeof(float4)), YIDX(idxOther, sizeof(float4))));
           // printf("atom %d bond %d gets force %f\n", idx, i, harmonicForce(bounds, pos, posOther, b.k, b.rEq));
           // printf("xs %f %f\n", pos.x, posOther.x);
            forceSum += harmonicForce(bounds, pos, posOther, b.k, b.rEq);
        }
        int zero = 0;
        float4 forceSumWhole = make_float4(forceSum);
        forceSumWhole.w = * (float *) &zero;
        forces[idxSelf] += forceSumWhole;
    }
}


FixBondHarmonic::FixBondHarmonic(SHARED(State) state_, string handle) : Fix(state_, handle, string("None"), bondHarmType, 1), bondsGPU(1), bondIdxs(1)  {
    forceSingle = true;
    maxBondsPerBlock = 0;
}


void FixBondHarmonic::compute() {
    int nAtoms = state->atoms.size();
    int activeIdx = state->gpd.activeIdx;
    if (bonds.size()) {
        compute_cu<<<NBLOCK(nAtoms), PERBLOCK, sizeof(BondHarmonicGPU) * maxBondsPerBlock>>>(nAtoms, state->gpd.xs.getTex(), state->gpd.fs(activeIdx), state->gpd.idToIdxs.getTex(), bondsGPU.ptr, bondIdxs.ptr, state->boundsGPU);
    }

}

//void cumulativeSum(int *data, int n);
//okay, so the net result of this function is that two arrays (items, idxs of items) are on the gpu and we know how many bonds are in bondiest  block

template <class SRC, class DEST>
int copyBondsToGPU(vector<Atom> &atoms, vector<BondVariant> &src, GPUArrayDevice<DEST> *dest, GPUArrayDevice<int> *destIdxs) {
    vector<int> idxs(atoms.size()+1, 0); //started out being used as counts
    vector<int> numAddedPerAtom(atoms.size(), 0);
    //so I can arbitrarily order.  I choose to do it by the the way atoms happen to be sorted currently.  Could be improved.
    for (BondVariant &s : src) {
        idxs[get<SRC>(s).atoms[0] - atoms.data()]++;
        idxs[get<SRC>(s).atoms[1] - atoms.data()]++;
    }
    cumulativeSum(idxs.data(), atoms.size()+1);  
    vector<DEST> destHost(idxs.back());
    for (BondVariant &sv : src) {
        SRC &s = get<SRC>(sv);
        int bondAtomIds[2];
        int bondAtomIndexes[2];
        bondAtomIds[0] = s.atoms[0]->id;
        bondAtomIds[1] = s.atoms[1]->id;
        bondAtomIndexes[0] = s.atoms[0] - atoms.data();
        bondAtomIndexes[1] = s.atoms[1] - atoms.data();
        for (int i=0; i<2; i++) {
            DEST a;
            a.myId = bondAtomIds[i];
            a.idOther = bondAtomIds[!i];
            a.takeValues(s);
            destHost[idxs[bondAtomIndexes[i]] + numAddedPerAtom[bondAtomIndexes[i]]] = a;
            numAddedPerAtom[bondAtomIndexes[i]]++;
        }
    }
    *dest = GPUArrayDevice<DEST>(destHost.size());
    dest->set(destHost.data());
    *destIdxs = GPUArrayDevice<int>(idxs.size());
    destIdxs->set(idxs.data());

    //getting max # bonds per block
    int maxPerBlock = 0;
    for (int i=0; i<atoms.size(); i+=PERBLOCK) {
        maxPerBlock = fmax(maxPerBlock, idxs[fmin(i+PERBLOCK+1, idxs.size()-1)] - idxs[i]);
    }
    return maxPerBlock;

}


bool FixBondHarmonic::prepareForRun() {
    vector<Atom> &atoms = state->atoms;
    refreshAtoms();
    maxBondsPerBlock = copyBondsToGPU<BondHarmonic, BondHarmonicGPU>(atoms, bonds, &bondsGPU, &bondIdxs);

    return true;

}

void FixBondHarmonic::createBond(Atom *a, Atom *b, float k, float rEq) {
    vector<Atom *> atoms = {a, b};
    validAtoms(atoms);
    bonds.push_back(BondHarmonic(a, b, k, rEq));
    bondAtomIds.push_back(make_int2(a->id, b->id));
}

bool FixBondHarmonic::refreshAtoms() {
    vector<int> idxFromIdCache = state->idxFromIdCache;
    vector<Atom> &atoms = state->atoms;
    for (int i=0; i<bondAtomIds.size(); i++) {
        int2 ids = bondAtomIds[i];
        get<BondHarmonic>(bonds[i]).atoms[0] = &atoms[idxFromIdCache[ids.x]];//state->atomFromId(ids.x);
        get<BondHarmonic>(bonds[i]).atoms[1] = &atoms[idxFromIdCache[ids.y]];//state->atomFromId(ids.y);
    }
    return bondAtomIds.size() == bonds.size();
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
        ss << b.atoms[0]->id << " " << b.atoms[1]->id << " " << b.k << " " << b.rEq << "\n";
    }
    ss << "</" << restartHandle << ">\n";
    //NOT DONE
    cout << "BOND REST CHUNK NOT DONE" << endl;
    return ss.str();
}

void export_FixBondHarmonic() {
    class_<FixBondHarmonic, SHARED(FixBondHarmonic), bases<Fix> > ("FixBondHarmonic", init<SHARED(State), string> (args("state", "handle")))
        .def("createBond", &FixBondHarmonic::createBond)
        ;

}
