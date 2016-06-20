#include "FixLangevin.h"
#define INVALID_VAL INT_MAX
#include "cutils_math.h"

#include "State.h"
const std::string LangevinType = "Langevin";
namespace py = boost::python;



__global__ void compute_cu(int nAtoms, float4 *vs, float4 *fs, curandState_t *randStates, float dt, float T, float gamma) {

    int idx = GETIDX();
    if (idx < nAtoms) {

        //curandState_t localState;
        //curand_init(timestep, idx, seed, &localState);
        curandState_t *randState = randStates + idx;
        float3 Wiener;
        Wiener.x=curand_uniform(randState)*2.0f-1.0f;
        Wiener.y=curand_uniform(randState)*2.0f-1.0f;
        Wiener.z=curand_uniform(randState)*2.0f-1.0f;
        //if (idx==0 || idx == 1) {
        //    printf("%d %f %f %f\n", idx, Wiener.x, Wiener.y, Wiener.z);
       // }

        float3 vel = make_float3(vs[idx]);
        float4 force = fs[idx];

        float Bc = dt==0 ? 0 : sqrt(6.0*gamma*T/dt);

        float3 dForce = Wiener * Bc - vel * gamma;
    
        force += dForce;
        fs[idx]=force;
    }
}







void FixLangevin::setDefaults() {
    seed = 0;
    gamma = 1.0;
}

FixLangevin::FixLangevin(boost::shared_ptr<State> state_, std::string handle_, std::string groupHandle_, double temp_) : FixThermostatBase(temp_), Fix(state_, handle_, groupHandle_, LangevinType, false, false, false, 1) {
    setDefaults();
}

FixLangevin::FixLangevin(boost::shared_ptr<State> state_, std::string handle_, std::string groupHandle_, py::list intervals_, py::list temps_) : FixThermostatBase(intervals_, temps_), Fix(state_, handle_, groupHandle_, LangevinType, false, false, false, 1) {
    setDefaults();
}

FixLangevin::FixLangevin(boost::shared_ptr<State> state_, std::string handle_, std::string groupHandle_, py::object tempFunc_) : FixThermostatBase(tempFunc_), Fix(state_, handle_, groupHandle_, LangevinType, false, false, false, 1) {
    setDefaults();
}

void __global__ initRandStates(int nAtoms, curandState_t *states, int seed) {
    int idx = GETIDX();
    curand_init(seed, idx, 0, states + idx);

}


bool FixLangevin::prepareForRun() {
    turnBeginRun = state->runInit;
    turnFinishRun = state->runInit + state->runningFor;
    randStates = GPUArrayDeviceGlobal<curandState_t>(state->atoms.size());
    initRandStates<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(state->atoms.size(), randStates.data(), seed);
    return true;
}

bool FixLangevin::postRun() {
    finished = true;
    return true;
}

void FixLangevin::setParams(double seed_, double gamma_) {
    if (seed_ != INVALID_VAL) {
        seed = seed_;
    }
    if (gamma_ != INVALID_VAL) {
        gamma = gamma_;
    }
}
void FixLangevin::compute(bool computeVirials) {
    computeCurrentTemp(state->turn);
    double temp = getCurrentTemp();
    compute_cu<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(state->atoms.size(), state->gpd.vs.getDevData(), state->gpd.fs.getDevData(), randStates.data(), state->dt, temp, gamma);
    
}




void export_FixLangevin() {
    py::class_<FixLangevin, SHARED(FixLangevin), py::bases<Fix> > (
        "FixLangevin", 
        py::init<boost::shared_ptr<State>, std::string, std::string, py::list, py::list>(
            py::args("state", "handle", "groupHandle", "intervals", "temps")
            )

        
    )
   //HEY - ORDER IS IMPORTANT HERE.  LAST CONS ADDED IS CHECKED FIRST. A DOUBLE _CAN_ BE CAST AS A py::object, SO IF YOU PUT THE TEMPFUNC CONS LAST, CALLING WITH DOUBLE AS ARG WILL GO THERE, NOT TO CONST TEMP CONSTRUCTOR 
    .def(py::init<boost::shared_ptr<State>, std::string, std::string, py::object>(
                
            py::args("state", "handle", "groupHandle", "tempFunc")
                )
            )
    .def(py::init<boost::shared_ptr<State>, std::string, std::string, double>(
            py::args("state", "handle", "groupHandle", "temp")
                )
            )
    ;
}
