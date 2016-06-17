#include "helpers.h"
#include "FixBondHarmonic.h"
#include "cutils_func.h"
#include "FixHelpers.h"
#include "ReadConfig.h"

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
      pyListInterface(&bonds, &pyBonds) {
  if (state->readConfig->fileOpen) {
    auto restData = state->readConfig->readFix(type, handle);
    if (restData) {
      std::cout << "Reading restart data for fix " << handle << std::endl;
      readFromRestart(restData);
    }
  }
}



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


bool FixBondHarmonic::readFromRestart(pugi::xml_node restData) {
  auto curr_node = restData.first_child();
  while (curr_node) {
    std::string tag = curr_node.name();
    if (tag == "types") {
      for (auto type_node = curr_node.first_child(); type_node; type_node = type_node.next_sibling()) {
        int type;
        double k;
        double rEq;
	std::string type_ = type_node.attribute("id").value();
        type = atoi(type_.c_str());
	std::string k_ = type_node.attribute("k").value();
	std::string rEq_ = type_node.attribute("rEq").value();
        k = atof(k_.c_str());
        rEq = atof(rEq_.c_str());

        setBondTypeCoefs(type, k, rEq);
      }
    } else if (tag == "members") {
      for (auto member_node = curr_node.first_child(); member_node; member_node = member_node.next_sibling()) {
        int type;
        double k;
        double rEq;
        int ids[2];
	std::string type_ = member_node.attribute("type").value();
	std::string atom_a = member_node.attribute("atom_a").value();
	std::string atom_b = member_node.attribute("atom_b").value();
	std::string k_ = member_node.attribute("k").value();
	std::string rEq_ = member_node.attribute("rEq").value();
        type = atoi(type_.c_str());
        ids[0] = atoi(atom_a.c_str());
        ids[1] = atoi(atom_b.c_str());
        Atom * a = state->atomFromId(ids[0]);
        Atom * b = state->atomFromId(ids[1]);
        k = atof(k_.c_str());
        rEq = atof(rEq_.c_str());

	createBond(a, b, k, rEq, type);
      }
    }
    curr_node = curr_node.next_sibling();
  }
  return true;
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

/*
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
*/


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
