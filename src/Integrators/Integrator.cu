#include "Integrator.h"

#include "boost_for_export.h"
#include "globalDefs.h"
#include "cutils_func.h"
#include "DataSetUser.h"
#include "Fix.h"
#include "GPUArray.h"
#include "PythonOperation.h"
#include "WriteConfig.h"

using namespace std;




__global__ void zeroVectorPreserveW(float4 *xs, int n) {
    int idx = GETIDX();
    if (idx < n) {
        float w = xs[idx].w;
        xs[idx] = make_float4(0, 0, 0, w);
    }
}

void Integrator::stepInit(bool computeVirials)
{
    if (computeVirials) {
        //reset virials each turn
        state->gpd.virials.d_data.memset(0);
    }


    for (Fix *f : state->fixes) {
        if (state->turn % f->applyEvery == 0) {
            f->stepInit();
        }
    }
}

void Integrator::stepFinal()
{
    for (Fix *f : state->fixes) {
        if (state->turn % f->applyEvery == 0) {
            f->stepFinal();
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
    if (state->rCut == RCUT_INIT) {
        cout << "rcut is not set" << endl;
        assert(state->rCut != RCUT_INIT);
    }
    if (state->is2d and state->periodic[2]) {
        cout << "2d system cannot be periodic is z dimension" << endl;
        assert(not (state->is2d and state->periodic[2]));
    }
    mdAssert(state->bounds.isInitialized(), "Bounds must be initialized");
    /*
    if (not state->bounds.isInitialized()) {
        cout << "Bounds not initialized" << endl;
        assert(state->bounds.isInitialized());
    }
    */

}


void Integrator::basicPrepare(int numTurns) {
    int nAtoms = state->atoms.size();
    state->runningFor = numTurns;
    state->runInit = state->turn;
    //state->updateIdxFromIdCache();
    state->prepareForRun();
    setActiveData();
    for (GPUArray *dat : activeData) {
        dat->dataToDevice();
    }
    for (Fix *f : state->fixes) {
        f->updateGroupTag();
        f->prepareForRun();
    }
    state->gridGPU.periodicBoundaryConditions(-1, true);
    for (boost::shared_ptr<DataSetUser> ds : state->dataManager.dataSets) {
        ds->prepareForRun(); //will also prepare those data sets' computers
    }
}


void Integrator::basicFinish() {
    for (Fix *f : state->fixes) {
        f->postRun();
    }
    if (state->asyncData && state->asyncData->joinable()) {
        state->asyncData->join();
    }
    for (GPUArray *dat : activeData) {
        dat->dataToHost();
    }
    cudaDeviceSynchronize();
    state->downloadFromRun();
}


void Integrator::setActiveData() {
    activeData = vector<GPUArray *>();
    activeData.push_back((GPUArray *) &state->gpd.ids);
    activeData.push_back((GPUArray *) &state->gpd.xs);
    activeData.push_back((GPUArray *) &state->gpd.vs);
    activeData.push_back((GPUArray *) &state->gpd.fs);
    activeData.push_back((GPUArray *) &state->gpd.idToIdxs);
    if (state->requiresCharges) {
        activeData.push_back((GPUArray *) &state->gpd.qs);
    }

    activeData.push_back((GPUArray *) &state->gpd.virials);
    activeData.push_back((GPUArray *) &state->gpd.perParticleEng);
}


Integrator::Integrator(State *state_) : IntegratorBasics(state_) {
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
    accumulate_gpu_if<float, float, SumSingleIf, N_DATA_PER_THREAD> <<<NBLOCK(state->atoms.size() / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*sizeof(float)*PERBLOCK>>>
        (
         eng.getDevData(), 
         state->gpd.perParticleEng.getDevData(),
         state->atoms.size(),
         warpSize,
         SumSingleIf(state->gpd.fs.getDevData(), groupTag)
        );
    /*
    sumPlain<float, float, N_DATA_PER_THREAD><<<NBLOCK(state->atoms.size() / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*sizeof(float)*PERBLOCK>>>(
            eng.getDevData(), state->gpd.perParticleEng.getDevData(),
            state->atoms.size(), groupTag, state->gpd.fs.getDevData(), warpSize);
            */
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

