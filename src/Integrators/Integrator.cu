#include "Integrator.h"

#include "boost_for_export.h"
#include "globalDefs.h"
#include "cutils_func.h"
#include "DataSet.h"
#include "Fix.h"
#include "GPUArray.h"
#include "PythonOperation.h"
#include "WriteConfig.h"

using namespace std;

void Integrator::stepInit()
{
    for (Fix *f : state->fixes) {
        if (state->turn % f->applyEvery == 0) {
            f->stepInit();
        }
    }
}

void Integrator::force(bool computeVirials) {
    int simTurn = state->turn;
    vector<Fix *> &fixes = state->fixes;
    for (Fix *f : fixes) {
        if (! (simTurn % f->applyEvery)) {
            f->compute(computeVirials);
        }
    }
};

void Integrator::stepFinal()
{
    for (Fix *f : state->fixes) {
        if (state->turn % f->applyEvery == 0) {
            f->stepFinal();
        }
    }
}


void Integrator::forceSingle(bool computeVirials) {
    for (Fix *f : state->fixes) {
        if (f->forceSingle) {
            f->compute(computeVirials);
        }
    }
}


void Integrator::singlePointEng() {
    GPUArrayGlobal<float> &perParticleEng = state->gpd.perParticleEng;
    perParticleEng.d_data.memset(0);
    for (Fix *f : state->fixes) {
        f->singlePointEng(perParticleEng.getDevData());
    }

}


void Integrator::doDataCollection() {
    DataManager &dm = state->dataManager;
    bool doingCollection = false;
    int64_t turn = state->turn;
    for (DataSet *ds : dm.dataSets) {
        if (ds->nextCollectTurn == turn) {
            doingCollection = true;
            break;
        }
    }
    if (doingCollection) {
        // this will need some thought, b/c can't compute it without going through all
        // the fixes.  Maybe have like state flag for computing virials.  If true, just
        // grab current virials, if false, zero forces vector and recompute it setting
        // virials flag to true, then grab virials vector
        bool computeVirials = false;
        bool computeEng = false;

        bool needToCopyForcesBack = false;
        for (DataSet *ds : dm.dataSets) {
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
        GPUData &gpd = state->gpd;
        float4 *xs = gpd.xs.getDevData();
        float4 *vs = gpd.vs.getDevData();
        float4 *fs = gpd.fs.getDevData();
        BoundsGPU &bounds = state->boundsGPU;
        int nAtoms = state->atoms.size();
        int64_t turn = state->turn;
        //void collect(int64_t turn, BoundsGPU &, int nAtoms,
        //             float4 *xs, float4 *vs, float4 *fs,
        //             float *engs, Virial *);
        cudaDeviceProp &prop = state->devManager.prop;
        for (DataSet *ds : dm.dataSets) {
            if (ds->nextCollectTurn == turn) {
                ds->collect(turn, bounds, nAtoms, xs, vs, fs,
                            gpd.perParticleEng.getDevData(),
                            gpd.perParticleVirial.getDevData(),
                            prop);
                ds->setNextCollectTurn(turn);
            }
        }
        cudaDeviceSynchronize();
        for (DataSet *ds : dm.dataSets) {
            ds->appendValues();
        }
        if (needToCopyForcesBack) {
            GPUArrayPair<float4> &fs = state->gpd.fs;
            fs.copyBetweenArrays(fs.activeIdx, !fs.activeIdx);
        }
    }
}


void Integrator::asyncOperations() {
    int turn = state->turn;

    // well, if I try to use a local state pointer, this segfaults. Need to
    // capture this instead.  Little confused
    auto writeAndPy = [this] (int64_t ts) {
        // have to set device in each thread
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


void Integrator::basicPreRunChecks() {
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
                cout << "Grid dimension " << i
                     << "has discretization smaller than rCut + padding"
                     << endl;
                assert(state->grid.ds[i] >= state->rCut + state->padding);
            }
        }
    }
    state->grid.adjustForChangedBounds();
}


void Integrator::basicPrepare(int numTurns) {
    int nAtoms = state->atoms.size();
    state->runningFor = numTurns;
    state->runInit = state->turn;
    state->updateIdxFromIdCache();
    for (Fix *f : state->fixes) {
        f->updateGroupTag();
        f->prepareForRun();
    }
    state->prepareForRun();
    for (GPUArray *dat : activeData) {
        dat->dataToDevice();
    }
    state->gridGPU.periodicBoundaryConditions(-1, true);
    state->dataManager.generateSingleDataSetList();
    for (DataSet *ds : state->dataManager.dataSets) {
        ds->setCollectMode();
        ds->prepareForRun();
    }
}


void Integrator::basicFinish() {
    if (state->asyncData && state->asyncData->joinable()) {
        state->asyncData->join();
    }
    for (GPUArray *dat : activeData) {
        dat->dataToHost();
    }
    cudaDeviceSynchronize();
    state->downloadFromRun();
    for (Fix *f : state->fixes) {
        f->postRun();
    }

}


void Integrator::setActiveData() {
    activeData = vector<GPUArray *>();
    activeData.push_back((GPUArray *) &state->gpd.ids);
    activeData.push_back((GPUArray *) &state->gpd.xs);
    activeData.push_back((GPUArray *) &state->gpd.vs);
    activeData.push_back((GPUArray *) &state->gpd.fs);
    activeData.push_back((GPUArray *) &state->gpd.idToIdxs);
    activeData.push_back((GPUArray *) &state->gpd.qs);
}


Integrator::Integrator(State *state_) : state(state_) {
    setActiveData();
}


void Integrator::writeOutput() {
    for (SHARED(WriteConfig) wc : state->writeConfigs) {
        wc->write(state->turn);
    }
}


double Integrator::singlePointEngPythonAvg(string groupHandle) {
    GPUArrayGlobal<float> eng(2);
    eng.d_data.memset(0);
    basicPreRunChecks();
    basicPrepare(0);
    cudaDeviceSynchronize();

    singlePointEng();
    cudaDeviceSynchronize();
    uint32_t groupTag = state->groupTagFromHandle(groupHandle);
    int warpSize = state->devManager.prop.warpSize;
    sumPlain<<<NBLOCK(state->atoms.size()), PERBLOCK, sizeof(float)*PERBLOCK>>>(
            eng.getDevData(), state->gpd.perParticleEng.getDevData(),
            state->atoms.size(), groupTag, state->gpd.fs.getDevData(), warpSize);
    eng.dataToHost();
    cudaDeviceSynchronize();
    CUT_CHECK_ERROR("Calculation of single point average energy failed");
    basicFinish();
    return eng.h_data[0] / *((int *)eng.h_data.data()+1);
}

boost::python::list Integrator::singlePointEngPythonPerParticle() {
    basicPrepare(0);
    singlePointEng();
    state->gpd.perParticleEng.dataToHost();
    state->gpd.ids.dataToHost();
    cudaDeviceSynchronize();
    CUT_CHECK_ERROR("Calculation of single point per-particle energy failed");
    vector<float> &engs = state->gpd.perParticleEng.h_data;
    vector<uint> &ids = state->gpd.ids.h_data;
    vector<int> &idToIdxsOnCopy = state->gpd.idToIdxsOnCopy;
    vector<double> sortedEngs(ids.size());

    for (int i=0, ii=state->atoms.size(); i<ii; i++) {
        int id = ids[i];
        int idxWriteTo = idToIdxsOnCopy[id];
        sortedEngs[idxWriteTo] = engs[i];
    }
    boost::python::list asPy(sortedEngs);
    basicFinish();
    return asPy;

}




void export_Integrator() {
    boost::python::class_<Integrator,
                          boost::noncopyable> (
        "Integrator"
    )
    .def("writeOutput", &Integrator::writeOutput)
    .def("energyAverage", &Integrator::singlePointEngPythonAvg,
            (boost::python::arg("groupHandle")="all")
        )
    .def("energyPerParticle", &Integrator::singlePointEngPythonPerParticle);
    //.def("run", &Integrator::run)
    ;
}

