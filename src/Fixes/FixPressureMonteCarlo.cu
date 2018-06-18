#include "FixPressureMonteCarlo.h"
#include "State.h"
#include "Mod.h"
namespace py = boost::python;
const std::string MonteCarloType = "MonteCarlo";
using namespace MD_ENGINE;

FixPressureMonteCarlo::FixPressureMonteCarlo(boost::shared_ptr<State> state_, std::string handle_ , double pressure_, double scale_, int applyEvery_,bool tune_,int tuneFreq_) : Fix(state_, handle_, "all", MonteCarloType, false, false, false, applyEvery_), Interpolator(pressure_), pressureComputer(state, "scalar"), enrgComputer(state, py::list(), "scalar", "all") {
    isThermostat = false;
    requiresPerAtomVirials=false;
    scale = scale_;
    tuneFreq = tuneFreq_;
    tune = tune_;
};

void FixPressureMonteCarlo::setTempInterpolator() {
    for (Fix *f: state->fixes) {
        if (f->isThermostat && f->groupHandle =="all") {
            std::string t = "temp";
            tempInterpolator = f->getInterpolator(t);
            return;
        }
    }
    mdError("No global thermostat found for the Monte Carlo barostat");
}

bool FixPressureMonteCarlo::prepareFinal() {
    turnBeginRun = state->runInit;
    turnFinishRun= state->runInit + state->runningFor;
    enrgComputer.prepareForRun();
    setTempInterpolator();
    vScale   = state->bounds.volume()*scale;       // initial delta volume element, do I know about scale_?
    nfake = 0;

    for (Atom a : state->atoms)  {
        if (a.mass < 10e-10 ) {
            nfake++;
        }
    }
    printf("Removing %d from sites in Barostat acceptance criterion\n",nfake);
    prepared = true;
    return prepared;
}

bool FixPressureMonteCarlo::stepFinal() {
    std::mt19937 &MTRNG = state->getRNG();                    // Mersenne Twister RNG
    std::uniform_real_distribution<double> Urand(0.0,1.0);   // distribution
    int    nAtoms   = state->atoms.size();                 // Total number of "particles"
    int nPerRingPoly= state->nPerRingPoly;                
    double Vold     = state->boundsGPU.volume();             // current system volume
    int64_t turn    = state->turn;
    computeCurrentVal(turn);
    double target   = getCurrentVal();                      // target pressure for barostat
    double temp     = tempInterpolator->getCurrentVal();    // target temperature for thermostat
    double kT       = state->units.boltz*temp / nPerRingPoly;

    // FIND CURRENT SYSTEM ENERGY
    enrgComputer.computeScalar_GPU(true, groupTag);
    cudaDeviceSynchronize();
    enrgComputer.computeScalar_CPU();
    double Uold = enrgComputer.engScalar;                  
    
    // PERFORM SYSTEM SCALING
    double dV       = vScale * 2.0 * (Urand(MTRNG) - 0.5);    
    double Vnew     = Vold + dV;                              // new proposed volume
    double posScale = std::pow(Vnew / Vold, 1.0/3.0);
    double invScale;
    if (nPerRingPoly > 1) {
        Mod::scaleSystemCentroids(state,make_real3(posScale,posScale,posScale));
    } else {
        Mod::scaleSystem(state, make_real3(posScale, posScale, posScale));
    }

    // FIND PROPOSED SYSTEM ENERGY
    enrgComputer.computeScalar_GPU(true, groupTag);
    cudaDeviceSynchronize();
    enrgComputer.computeScalar_CPU();
    double Unew = enrgComputer.engScalar;                  

    // COMPUTE THE ENSEMBLE WEIGHT OF PROPOSED VOLUME CHANGE
    // for direct volume-scaling
    double weight = (Unew - Uold)/nPerRingPoly + target*dV/ state->units.nktv_to_press 
    - ((nAtoms-nfake)/nPerRingPoly)*kT*log(Vnew / Vold) ; 

    // EVALUATE WHETHER MOVE IS ACCEPTED 
    if (weight > 0.0 && Urand(MTRNG) > std::exp(-weight / kT )) {
        // reject move/reset positions
        invScale = 1.0/posScale;
        if (nPerRingPoly > 1) {
            Mod::scaleSystemCentroids(state,make_real3(invScale,invScale,invScale));
        } else {
            Mod::scaleSystem(state,make_real3(invScale,invScale,invScale));
        }
    } else {
        nacc++;
    }
    natt++;

    if (natt >= tuneFreq) {
        printf("Current Barostat acceptance frequency: %f\n",1.0*nacc/natt);
        if (tune) {
            if (double (nacc) < 0.25*natt) {
                vScale *= 0.95;
                printf("Decreasing volume-scale factor to %f\n",vScale);
            } else if (double (nacc) > 0.75*natt) {
                vScale *= 1.05;
                printf("Increasing volume-scale factor to %f\n",vScale);
            }
        }
        natt = 0;
        nacc = 0;
    }
    return true;
}

bool FixPressureMonteCarlo::postRun() {
    finished = true;
    return true;
}

void export_FixPressureMonteCarlo() {
    py::class_<FixPressureMonteCarlo, boost::shared_ptr<FixPressureMonteCarlo>, py::bases<Fix> > (
        "FixPressureMonteCarlo", 
        py::init<boost::shared_ptr<State>,std::string, double, double, int, bool, int>(
            py::args("state","handle", "pressure", "scale", "applyEvery", "tune", "tuneFreq")
            )
    )
    ;
}
