#include "Integrater.h"
#include "cuda_call.h"
#include "Fix.h"
#include "WriteConfig.h"
// #include "globalDefs.h"
#include "PythonOperation.h"
#include "DataSet.h"
const string IntVerletType = "verlet";
const string IntRelaxType = "relax";


void Integrater::force(bool computeVirials) {
    
	int simTurn = state->turn;
	vector<Fix *> &fixes = state->fixes;
	for (Fix *f : fixes) {
		if (! (simTurn % f->applyEvery)) {
			f->compute(computeVirials);
		}
	}
};

void Integrater::forceSingle(bool computeVirials) {
	for (Fix *f : state->fixes) {
		if (f->forceSingle) {
			f->compute(computeVirials);
		}
	}
}




void Integrater::singlePointEng() {
    GPUArray<float> &perParticleEng = state->gpd.perParticleEng;
    perParticleEng.d_data.memset(0);
	for (Fix *f : state->fixes) {
        f->singlePointEng(perParticleEng.getDevData());
    }

}
void Integrater::doDataCollection() {
    DataManager &dm = state->dataManager;
    bool doingCollection = false;
    int64_t turn = state->turn;
    for (SHARED(DataSet) ds : dm.dataSets) {
        if (ds->nextCollectTurn == turn) {
            doingCollection = true;
            break;
        }
    }
    if (doingCollection) {
        bool computeVirials = false; //this will need some thought, b/c can't compute it without going through all the fixes.  Maybe have like state flag for computing virials.  If true, just grab current virials, if false, zero forces vector and recompute it setting virials flag to true, then grab virials vector
        bool computeEng = false;

        bool needToCopyForcesBack = false;
        for (SHARED(DataSet) ds : dm.dataSets) {
            computeEng = fmax(computeEng, ds->requiresEng);
            computeVirials = fmax(computeVirials, ds->requiresVirials);
        }
        if (computeEng) {
            singlePointEng();
        }
        if (computeVirials) {

            if (not state->computeVirials) {
                GPUArrayPair<float4> &fs = state->gpd.fs;
                fs.copyBetweenArrays(!fs.activeIdx, fs.activeIdx);
                forceSingle(true);
                needToCopyForcesBack = true;
                //okay, now virials are computed



            }


        }
        //okay, now go through all and give them their data
        for (SHARED(DataSet) ds : dm.dataSets) {
            //do operations!
        }
        if (needToCopyForcesBack) {
            GPUArrayPair<float4> &fs = state->gpd.fs;
            fs.copyBetweenArrays(fs.activeIdx, !fs.activeIdx);
        }



    }
}
void Integrater::asyncOperations() {
    int turn = state->turn;
    auto writeAndPy = [this] (int64_t ts) { //well, if I try to use a local state pointer, this segfaults.  Need to capture this instead.  Little confused
        //have to set device in each thread
        state->devManager.setDevice(state->devManager.currentDevice, false);
        for (SHARED(WriteConfig) wc : state->writeConfigs) {
            if (not (ts % wc->writeEvery)) {
                wc->write(ts);
            }
        }
        for (SHARED(PythonOperation) po : state->pythonOperations) {
            if (not (ts % po->operateEvery)) {
                po->operate(ts);
            }
        }
    };
    bool needAsync = false;
    for (SHARED(WriteConfig) wc : state->writeConfigs) {
		if (not (turn % wc->writeEvery)) {
            needAsync = true;
            break;
		}
    }
    if (not needAsync) {
        for (SHARED(PythonOperation) po : state->pythonOperations) {
            if (not (turn % po->operateEvery)) {
                needAsync = true;
                break;
            }
        }
    }
    if (needAsync) {
        state->asyncHostOperation(writeAndPy);
    }
}
/*
__global__ void printFloats(cudaTextureObject_t xs, int n) {
    int idx = GETIDX();
    if (idx < n) {
        int xIdx = XIDX(idx, sizeof(float4));
        int yIdx = YIDX(idx, sizeof(float4));
        float4 x = tex2D<float4>(xs, xIdx, yIdx);
        printf("idx %d, vals %f %f %f %d\n", idx, x.x, x.y, x.z, *(int *) &x.w);

    }
}
__global__ void printFloats(float4 *xs, int n) {
    int idx = GETIDX();
    if (idx < n) {
        float4 x = xs[idx];
        printf("idx %d, vals %f %f %f %f\n", idx, x.x, x.y, x.z, x.w);

    }
}
*/


void Integrater::basicPreRunChecks() {
    if (state->devManager.prop.major < 3) {
        cout << "Device compute capability must be >= 3.0. Quitting" << endl;
        assert(state->devManager.prop.major >= 3);
    }
    if (not state->grid.isSet) {
        cout << "Atom grid is not set!" << endl;
        assert(state->grid.isSet);
    }
    if (state->rCut == RCUT_INIT) {
        cout << "rcut is not set" << endl;
        assert(state->rCut != RCUT_INIT);
    }
    if (state->is2d and state->periodic[2]) {
        cout << "2d system cannot be periodic is z dimension" << endl;
        assert(not (state->is2d and state->periodic[2]));
    }
    for (int i=0; i<3; i++) {
        if (i<2 or (i==2 and state->periodic[2])) {
            if (state->grid.ds[i] < state->rCut + state->padding) {
                cout << "Grid dimension " << i << "has discretization smaller than rCut + padding" << endl;
                assert(state->grid.ds[i] >= state->rCut + state->padding);
            }
        }
    }
    state->grid.adjustForChangedBounds();
}

void Integrater::basicPrepare(int numTurns) {
    int nAtoms = state->atoms.size();
	state->runningFor = numTurns;
    state->runInit = state->turn; 
    //Add refresh atoms!
    state->updateIdxFromIdCache(); //for updating fix atom pointers, etc
    state->prepareForRun();
    for (Fix *f : state->fixes) {
        f->updateGroupTag();
        f->prepareForRun();
    }
    for (GPUArrayBase *dat : activeData) {
        dat->dataToDevice();
    }
    state->gridGPU.periodicBoundaryConditions(state->rCut + state->padding, true);
}

void Integrater::basicFinish() {
    if (state->asyncData && state->asyncData->joinable()) {
        state->asyncData->join();
    }
    for (GPUArrayBase *dat : activeData) {
        dat->dataToHost();
    }
    cudaDeviceSynchronize();
    state->downloadFromRun();
    for (Fix *f : state->fixes) {
        f->postRun();
    }

}
void Integrater::setActiveData() {
    activeData = vector<GPUArrayBase *>();
    activeData.push_back((GPUArrayBase *) &state->gpd.ids);
    activeData.push_back((GPUArrayBase *) &state->gpd.xs);
    activeData.push_back((GPUArrayBase *) &state->gpd.vs);
    activeData.push_back((GPUArrayBase *) &state->gpd.fs);
    activeData.push_back((GPUArrayBase *) &state->gpd.fsLast);
    activeData.push_back((GPUArrayBase *) &state->gpd.idToIdxs);
    activeData.push_back((GPUArrayBase *) &state->gpd.qs);
}

Integrater::Integrater(State *state_, string type_) : state(state_), type(type_){
    setActiveData(); 
}

void export_Integrater() {
    class_<Integrater, boost::noncopyable> ("Integrater")
        //.def("run", &Integrater::run)
        ;
}

