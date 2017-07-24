#include "IntegratorMC.h"
#include "Mod.h"
#include "State.h"
#include "Fix.h"
#include "DataComputer.h"
#include "DataSetUser.h"

using namespace MD_ENGINE;

namespace py = boost::python;

IntegratorMC::IntegratorMC(State *state_)
    : Integrator(state_)
{

}

double IntegratorMC::run(int numTurns, double maxMoveDist, double temp_) {
    
    temp = temp_; 
    basicPreRunChecks();
    //basicPrepare(numTurns); //nlist built here
    //force(false);

    std::vector<bool> prepared = basicPrepare(numTurns);

    for (int i = 0; i<prepared.size(); i++) {
        if (!prepared[i]) {
            for (Fix *f : state->fixes) {
                bool isPrepared = f->prepareForRun();
                if (!isPrepared) {
                    mdError("A fix is unable to be instantiated correctly.");
                }
            }
        }
    }

    groupTag = state->groupTagFromHandle("all");	

    DataManager &dataManager = state->dataManager;
    //so here's we're kinda abusing the system.  We're just getting the data manager to create a data set for us, but then we ask it to stop recording data - we will do it manually.
    boost::shared_ptr<DataSetUser> engDataSet = dataManager.recordEnergy("all", "scalar", 1, py::object(), py::list(), "all");
    dataManager.stopRecord(engDataSet);
    comp = (void *) engDataSet->computer.get();

    MD_ENGINE::DataComputer *comp_loc = (MD_ENGINE::DataComputer *)comp;
    comp_loc->computeScalar_GPU(true, groupTag);
    cudaDeviceSynchronize();
    engLast = comp_loc->gpuBufferReduce.h_data[0];

    auto start = std::chrono::high_resolution_clock::now();
    dtf = 0.5f * state->dt * state->units.ftm_to_v;
    int tuneEvery = state->tuneEvery;
    bool haveTunedWithData = false;
    double timeTune = 0;
    std::mt19937 &rng = state->getRNG();
    for (int i=0; i<numTurns; ++i) {

        //doing PBC every turn to be safe
        state->gridGPU.periodicBoundaryConditions();
        MCDisplace(maxMoveDist, rng);

        if (state->turn % tuneEvery == 0) {
            timeTune += tune();
        } else if (not haveTunedWithData and state->turn-state->runInit < tuneEvery and state->nlistBuildCount > 20) {
            timeTune += tune();
            haveTunedWithData = true;
        }
        asyncOperations();
        //HEY - MAKE DATA APPENDING HAPPEN WHILE SOMETHING IS GOING ON THE GPU.  
        doDataComputation();
        doDataAppending();
    }
    cudaDeviceSynchronize();
    CUT_CHECK_ERROR("after run\n");
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double ptsps = state->atoms.size()*numTurns / (duration.count() - timeTune);
    mdMessage("runtime %f\n%e particle timesteps per second\n",
              duration.count(), ptsps);

    basicFinish();
    return ptsps;
}

__global__ void displace(float4 *x, float3 displace) { 
    float4 xLoc = x[0];
    xLoc.x += displace.x;
    xLoc.y += displace.y;
    xLoc.z += displace.z;
    x[0] = xLoc;
}

void IntegratorMC::MCDisplace(double maxMoveDist, std::mt19937 &rng) {
    double range = rng.max()-rng.min();
    double moveDist = maxMoveDist * (rng()-rng.min())/range;
    Vector moveVec = Mod::randomUV(rng) * moveDist;
    int particleIdx = state->atoms.size() * (rng()-rng.min())/range;
    displace<<<1, 1>>>(state->gpd.xs.getDevData()+particleIdx, moveVec.asFloat3());
    MD_ENGINE::DataComputer *comp_loc = (MD_ENGINE::DataComputer *)comp;
    comp_loc->computeScalar_GPU(true, groupTag);
    cudaDeviceSynchronize();
    double eng = comp_loc->gpuBufferReduce.h_data[0];
    double rand = (rng()-rng.min())/range;
    double accept = exp(-(eng-engLast) / (temp*state->units.boltz));
    if (rand < accept) {
        engLast = eng;
    } else {
        displace<<<1, 1>>>(state->gpd.xs.getDevData()+particleIdx, (moveVec * -1).asFloat3());
    }
}

void export_IntegratorMC()
{
    py::class_<IntegratorMC,
               boost::shared_ptr<IntegratorMC>,
               py::bases<Integrator>,
               boost::noncopyable>
    (
        "IntegratorMC",
        py::init<State *>()
    )
    .def("run", &IntegratorMC::run, (py::arg("numTurns"), py::arg("maxMoveDist"), py::arg("temp")) )
    ;
}
