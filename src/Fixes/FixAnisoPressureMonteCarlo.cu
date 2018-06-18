#include "FixAnisoPressureMonteCarlo.h"
#include "State.h"
#include "Mod.h"
namespace py = boost::python;
const std::string MonteCarloType = "AnisoMonteCarlo";
using namespace MD_ENGINE;

FixAnisoPressureMonteCarlo::FixAnisoPressureMonteCarlo(boost::shared_ptr<State> state_, std::string handle_ , double px_, double sx_,double py_, double sy_,double pz_,double sz_, int applyEvery_,bool tune_,int tuneFreq_) : Fix(state_, handle_, "all", MonteCarloType, false, false, false, applyEvery_), pressureComputer(state, "scalar"), enrgComputer(state, py::list(), "scalar", "all") {
    isThermostat = false;
    requiresPerAtomVirials=false;
    scale   = make_real3(sx_,sy_,sz_);
    targets = make_real3(px_,py_,pz_); 
    tuneFreq = tuneFreq_;
    tune    = tune_;
};

void FixAnisoPressureMonteCarlo::setTempInterpolator() {
    for (Fix *f: state->fixes) {
        if (f->isThermostat && f->groupHandle =="all") {
            std::string t = "temp";
            tempInterpolator = f->getInterpolator(t);
            return;
        }
    }
    mdError("No global thermostat found for the Anisotropic Monte Carlo barostat");
}

bool FixAnisoPressureMonteCarlo::prepareFinal() {
    enrgComputer.prepareForRun();
    setTempInterpolator();
    vScale   = state->bounds.volume()*scale;       // initial delta volume element, do I know about scale_?
    if (scale.x > 0.0) { useX = true;  axisMap[naxis] = 0; naxis += 1;}
    if (scale.y > 0.0) { useY = true;  axisMap[naxis] = 1; naxis += 1;}
    if (scale.z > 0.0) { useZ = true;  axisMap[naxis] = 2; naxis += 1;}
    
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

bool FixAnisoPressureMonteCarlo::stepFinal() {
    std::mt19937 &MTRNG = state->getRNG();                    // Mersenne Twister RNG
    std::uniform_real_distribution<double> Urand(0.0,1.0);   // distribution
    int    nAtoms   = state->atoms.size();                 // Total number of "particles"
    int nPerRingPoly= state->nPerRingPoly;                
    double Vold     = state->boundsGPU.volume();             // current system volume
    double temp     = tempInterpolator->getCurrentVal();    // target temperature for thermostat
    double kT       = state->units.boltz*temp / nPerRingPoly;

    // FIND CURRENT SYSTEM ENERGY
    enrgComputer.computeScalar_GPU(true, groupTag);
    cudaDeviceSynchronize();
    enrgComputer.computeScalar_CPU();
    double Uold = enrgComputer.engScalar;                  
    
    // CHOOSE AN AXIS TO MODIFY AT RANDOM AND GET VOLUME SCALING
    double arand    = naxis*Urand(MTRNG);
    int    axis     = axisMap[int (arand)];
    double dV;
    double Vnew = Vold;                              // new proposed volume
    double target;
    real3 posScale = make_real3(1.,1.,1.);
    if (axis == 0) {
        target   = targets.x;
        dV  = vScale.x * 2.0 * (Urand(MTRNG) - 0.5); // this will only be applied in one dimension
        Vnew += dV;
        posScale.x = Vnew/Vold;
    } else if (axis == 1 ) {
        target   = targets.y;
        dV  = vScale.y * 2.0 * (Urand(MTRNG) - 0.5); // this will only be applied in one dimension
        Vnew += dV;
        posScale.y = Vnew/Vold;
    } else if (axis == 2) {
        target   = targets.z;
        dV  = vScale.z * 2.0 * (Urand(MTRNG) - 0.5); // this will only be applied in one dimension
        Vnew += dV;
        posScale.z = Vnew/Vold;
    }
    
    // PERFORM SYSTEM SCALING
    if (nPerRingPoly > 1) {
        Mod::scaleSystemCentroids(state,posScale);
    } else {
        Mod::scaleSystem(state, posScale);
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
    real3 invScale; 
    if (weight > 0.0 && Urand(MTRNG) > std::exp(-weight / kT)) {
        // reject move/reset positions
        invScale = 1.0/posScale;
        if (nPerRingPoly > 1) {
            Mod::scaleSystemCentroids(state,invScale);
        } else {
            Mod::scaleSystem(state,invScale);
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
                printf("Decreasing volume-scale factor to %f, %f, %f\n",vScale.x,vScale.y,vScale.z);
            } else if (double (nacc) > 0.75*natt) {
                vScale *= 1.05;
                printf("Increasing volume-scale factor to %f, %f, %f\n",vScale.x,vScale.y,vScale.z);
            }
        }
        natt = 0;
        nacc = 0;
    }
    return true;
}

bool FixAnisoPressureMonteCarlo::postRun() {
    return true;
}

void export_FixAnisoPressureMonteCarlo() {
    py::class_<FixAnisoPressureMonteCarlo, boost::shared_ptr<FixAnisoPressureMonteCarlo>, py::bases<Fix> > (
        "FixAnisoPressureMonteCarlo", 
        py::init<boost::shared_ptr<State>,std::string, double, double,double,double,double,double,int,bool,int>(
            py::args("state","handle", "px", "sx","py", "sy", "pz", "sz",  "applyEvery","tune","tuneFreq")
            )
    )
    ;
}
