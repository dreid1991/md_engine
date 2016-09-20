#include "FixPressureBerendsen.h"
#include "State.h"
#include "Mod.h"
namespace py = boost::python;
const std::string BerendsenType = "Langevin";
using namespace MD_ENGINE;

FixPressureBerendsen::FixPressureBerendsen(boost::shared_ptr<State> state_, std::string handle_, double pressure_, double period_, int applyEvery_) : Interpolator(pressure_), Fix(state_, handle_, "all", BerendsenType, false, true, false, applyEvery_), pressureComputer(state, true, false), period(period_) {
    bulkModulus = 10; //lammps
};

bool FixPressureBerendsen::prepareForRun() {
    turnBeginRun = state->runInit;
    turnFinishRun = state->runInit + state->runningFor;
    pressureComputer.prepareForRun(); 
    return true;
}

bool FixPressureBerendsen::stepFinal() {
    pressureComputer.computeScalar_GPU(true, 1);
    computeCurrentVal(state->turn);
    double target = getCurrentVal();
    cudaDeviceSynchronize();
    pressureComputer.computeScalar_CPU();
    double pressure = pressureComputer.pressureScalar;
    double dilation = std::pow(1.0 - state->dt/period * (target - pressure) / bulkModulus, 1.0/3.0);
    Mod::scaleSystem(state, dilation);
    return true;
}

bool FixPressureBerendsen::postRun() {
    finished = true;
    return true;
}

void export_FixPressureBerendsen() {
    py::class_<FixPressureBerendsen, boost::shared_ptr<FixPressureBerendsen>, py::bases<Fix> > (
        "FixPressureBerendsen", 
        py::init<boost::shared_ptr<State>, std::string, double, double, int>(
            py::args("state", "handle", "pressure", "period", "applyEvery")
            )

        
    )
    ;
}
