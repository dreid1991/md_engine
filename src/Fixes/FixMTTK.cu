#include "FixMTTK.h"
#include <cmath> // M_PI
#include <string>

#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <boost/python.hpp>
#include "cutils_func.h"
#include "cutils_math.h"
#include "Logging.h"
#include "State.h"
#include "Mod.h"
#include "Group.h"
#include "Virial.h"
namespace py = boost::python;

std::string MTTKType = "MTTK";


// putting things in an unnamed namespace prevents link time errors if we have kernels with the same name
// --- put kernels here; host-device functions should be in the class namespace, not free-floating in 
//     the cuda file, if at all possible
namespace {
static const double sy_fac[] = { 0.2967324292201065,0.2967324292201065,-0.186929716880426,0.2967324292201065,0.2967324292201065 };

static inline double series_sinhx(double x) {
    double xSqr = x*x;
    return (1.0 + (xSqr/6.0)*(1.0 + (xSqr/20.0)*(1.0 + (xSqr/42.0)*(1.0 + (xSqr/72.0)*(1.0 + (xSqr/110.0))))));
}

__global__ void rescale_cu(int nAtoms, uint groupTag, real4 *vs, real4 *fs, real3 scale)
{
    int idx = GETIDX();
    if (idx < nAtoms) {
        uint groupTagAtom = *((uint *) &(fs[idx].w));
        //uint groupTagAtom = *((uint *) ( ((real *)(fs+idx))+3));
        if (groupTag & groupTagAtom) {
            real4 vel = vs[idx];
            vel.x *= scale.x;
            vel.y *= scale.y;
            vel.z *= scale.z;
            vs[idx] = vel;
        }
    }
}

// copied from integrator verlet; NOTE that we removed the force-to-zero!
/*
__global__ void nve_v_cu(int nAtoms, real4 *vs, real4 *fs, real dtf) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        // Update velocity by a half timestep
        real4 vel = vs[idx];
        real invmass = vel.w;
        real4 force = fs[idx];
        
        // ghost particles should not have their velocities integrated; causes overflow
        if (invmass > INVMASSBOOL) {
            vs[idx] = make_real4(0.0, 0.0, 0.0,invmass);
            return;
        }

        //real3 dv = dtf * invmass * make_real3(force);
        real3 dv = dtf * invmass * make_real3(force);
        vel += dv;
        vs[idx] = vel;
    }
}
*/
/*
__global__ void nvt_v_cu(int nAtoms, real4 *vs, real4 *fs, real dtf) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        // Update velocity by a half timestep
        real4 vel = vs[idx];
        real invmass = vel.w;
        real4 force = fs[idx];
        
        // ghost particles should not have their velocities integrated; causes overflow
        if (invmass > INVMASSBOOL) {
            vs[idx] = make_real4(0.0, 0.0, 0.0,invmass);
            return;
        }

        //real3 dv = dtf * invmass * make_real3(force);
        real3 dv = dtf * invmass * make_real3(force);
        vel += dv;
        vs[idx] = vel;
    }
}
*/
// copied from integrator verlet; NOTE that we removed the force-to-zero!
/*    npt_v_cu<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(
            state->atoms.size(),
            state->gpd.vs.getDevData(),
            state->gpd.fs.getDevData(),
            dtf,
            h,exp_min_g);
*/
__global__ void npt_v_cu(int nAtoms, real4 *vs, real4 *fs, real dtf,
                         double h, double exp_min_g) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        // Update velocity by a half timestep
        real4 velWhole = vs[idx];
        real invmass = velWhole.w;
        real4 forceWhole = fs[idx];
        
        // ghost particles should not have their velocities integrated; causes overflow
        if (invmass > INVMASSBOOL) {
            vs[idx] = make_real4(0.0, 0.0, 0.0,invmass);
            return;
        }

        //real3 dv = dtf * invmass * make_real3(force);
        real3 vel = make_real3(velWhole);
        real3 force = make_real3(forceWhole);

        real3 new_vel = exp_min_g * (exp_min_g * vel + (dtf * invmass * h * force));
        vs[idx] = make_real4(new_vel.x,new_vel.y,new_vel.z,velWhole.w);

    }
}
__global__ void rescale_no_tags_cu(int nAtoms, real4 *vs, real3 scale)
{
    int idx = GETIDX();
    if (idx < nAtoms) {
        real4 vel = vs[idx];
        vel.x *= scale.x;
        vel.y *= scale.y;
        vel.z *= scale.z;
        vs[idx] = vel;
    }
}

} // namespace



// general constructor; may be a thermostat, or a barostat-thermostat
FixMTTK::FixMTTK(boost::shared_ptr<State> state_, std::string handle_,
                             std::string groupHandle_)
        : 
          Fix(state_,
              handle_,           // Fix handle
              groupHandle_,      // Group handle
              MTTKType,   // Fix name
              false,            // forceSingle
              false,
              false,            // requiresCharges
              1,                 // applyEvery
              50                // orderPreference
             ),
            tempComputer(state, "tensor"), 
            pressComputer(state, "tensor") 
{
    // default to isotropic water
    compressibility = 4.559625e-5 * 3.0; // compressibility of water, in atm^-1
    
    pressComputer.usingExternalTemperature = true;
    //as xx, yy, zz, xy, xz, yz
    compressibility_matrix = std::vector<double>{4.559625e-5,4.559625e-5,4.559625e-5,0.0,0.0,0.0};
    nh_chainlength = 10;
    tFrequency = pFrequency = 0.002; // some non-zero number.
}


/* NOTE: if FixMTTK.setPressure() is called in a python script, (i.e., NPT equilibration) and 
   then NVT is later desired, they will have to deactivate this fix, and create a new instance 
   of FixMTTK without .setPressure() being called */

// pressure can be constant double value
void FixMTTK::setPressure(double press, double timeConstant) {
    // get the pressmode and couplestyle; parseKeyword also changes the 
    // state of the pressComputer & tempComputer if needed; alters the boolean flags for 
    // the dimensions that will be barostatted (XYZ, or XY).
    pressInterpolator = Interpolator(press);
    barostatting = true;
    pFrequency = 1.0 / timeConstant;
}

// could also be a python function
void FixMTTK::setPressure(py::object pressFunc, double timeConstant) {
    pressInterpolator = Interpolator(pressFunc);
    barostatting = true;
    pFrequency = 1.0 / timeConstant;
}

// could also be a list of set points with accompanying intervals (denoted by turns - integer values)
void FixMTTK::setPressure(py::list pressures, py::list intervals, double timeConstant) {
    pressInterpolator = Interpolator(pressures, intervals);
    barostatting = true;
    pFrequency = 1.0 / timeConstant; 
}


// and analogous procedure with setting the temperature
void FixMTTK::setTemperature(double temperature, double timeConstant) {
    tempInterpolator = Interpolator(temperature);
    tFrequency = 1.0 / timeConstant;

}

void FixMTTK::setTemperature(py::object tempFunc, double timeConstant) {
    tempInterpolator = Interpolator(tempFunc);
    tFrequency = 1.0 / timeConstant;
}

void FixMTTK::setTemperature(py::list intervals, py::list temps, double timeConstant) {
    tempInterpolator = Interpolator(intervals, temps);
    tFrequency = 1.0 / timeConstant;
}

void FixMTTK::trotter_integrate_barostat() {

    // should correspond to the lengths of the NH chains for the barostat
    std::vector<double> GQ = std::vector<double>(press_xi.size(),0.0);

    // nvar has value 1..
    double nd = 1.0;
    // iQinv = MassQ_QPinv
    // Ekin = veta * veta / MassQ_Winv;
    double Ekin = veta * veta / MassQ_Winv;
    double kT = state->units.boltz * setPointTemperature;

    double dt, Efac;
    double sy_steps = (double) sy_order;
    int n = (int) (press_vxi.size() - 1); // easy access of last index
    // TODO : re-check this and verify that things are ok (referencing proper variables, etc.
    for (int i = 0; i < sy_order; i++) {
        for (int j = 0; j < sy_order; j++) {
            dt = sy_fac[j] * state->dt / sy_steps;

            GQ[0] = MassQ_QPinv[0] * (Ekin - nd * kT);
            
            for (int k = 0; k < press_xi.size() - 1; k++) {
                if (MassQ_QPinv[k+1] > 0) {
                    GQ[k+1] = MassQ_QPinv[k+1] * ( (press_vxi[k] * press_vxi[k]) / (MassQ_QPinv[k] - kT));
                } else {
                    GQ[k+1] = 0.0;
                }
            }

            // last chain in the sequence
            press_vxi[n] += 0.25 * dt * GQ[n];

            for (int k = n; k > 0; k--) {
                Efac = std::exp(-0.125 * dt * press_vxi[k]);
                press_vxi[k-1] = Efac * (press_vxi[k-1]*Efac + 0.25 * dt * GQ[k-1]);
            }

            Efac = std::exp(-0.5 * dt * press_vxi[0]);

            veta *= Efac;

            Ekin *= (Efac * Efac);

            GQ[0] = MassQ_QPinv[0] * (Ekin - nd * kT);

            for (int k = 0; k<press_xi.size(); k++) {
                press_xi[k] += 0.5 * dt * press_vxi[k];
            }

            for (int k = 0; k < n; k++) {
                Efac = std::exp(-0.125*dt*press_vxi[k+1]);
                press_vxi[k] = Efac * (press_vxi[k]*Efac + 0.25 * dt * GQ[k]);
                if (MassQ_QPinv[k+1] > 0) {
                    GQ[k+1] = MassQ_QPinv[k+1] * ( (press_vxi[k] * press_vxi[k]) / (MassQ_QPinv[k] - kT));
                } else {
                    GQ[k+1] = 0.0;
                }
            }
            press_vxi[n] += 0.25 * dt * GQ[n];
        }
    }
}

void FixMTTK::trotter_integrate_thermostat() {
    
    // note that this scales the velocities as well

    // we should have the system's full-step kinetic energy, as well as the current value of the 
    // thermostat scale factor, which is a scalar value
    std::vector<double> GQ = std::vector<double>(xi.size(),0.0);

    double nd = (double) tempComputer.ndf;
    
    double Ekin = traceKE * (state->units.boltz / state->units.mvv_to_eng) * 2.0 * nhc_scale;
    double kT = state->units.boltz * setPointTemperature;

    double dt, Efac;
    double sy_steps = (double) sy_order;
    int n = (int) (vxi.size() - 1); // easy access of last index
    for (int i = 0; i < sy_order; i++) {
        for (int j = 0; j < sy_order; j++) {
            dt = sy_fac[j] * state->dt / sy_steps;

            GQ[0] = MassQ_Qinv[0] * (Ekin - nd * kT);
            
            for (int k = 0; k < xi.size() - 1; k++) {
                if (MassQ_QPinv[k+1] > 0) {
                    GQ[k+1] = MassQ_Qinv[k+1] * ( (vxi[k] * vxi[k]) / (MassQ_Qinv[k] - kT));
                } else {
                    GQ[k+1] = 0.0;
                }
            }

            // last chain in the sequence
            vxi[n] += 0.25 * dt * GQ[n];

            for (int k = n; k > 0; k--) {
                Efac = std::exp(-0.125 * dt * vxi[k]);
                vxi[k-1] = Efac * (vxi[k-1]*Efac + 0.25 * dt * GQ[k-1]);
            }

            Efac = std::exp(-0.5 * dt * vxi[0]);

            scalefac *= Efac;

            Ekin *= (Efac * Efac);

            GQ[0] = MassQ_Qinv[0] * (Ekin - nd * kT);

            for (int k = 0; k<xi.size(); k++) {
                xi[k] += 0.5 * dt * vxi[k];
            }

            for (int k = 0; k < n; k++) {
                Efac = std::exp(-0.125*dt*vxi[k+1]);
                vxi[k] = Efac * (vxi[k]*Efac + 0.25 * dt * GQ[k]);
                if (MassQ_Qinv[k+1] > 0) {
                    GQ[k+1] = MassQ_Qinv[k+1] * ( (vxi[k] * vxi[k]) / (MassQ_Qinv[k] - kT));
                } else {
                    GQ[k+1] = 0.0;
                }
            }
            vxi[n] += 0.25 * dt * GQ[n];
        }
    }
}

void FixMTTK::apply_nhc_scale() {
    
    // apply the nhc scalefactor to tempTensor in pressComputer (if barostatting) and 
    // tempTensor in tempComputer;
    

    tempComputer.tempTensor *= nhc_scale;
    if (barostatting) pressComputer.tempTensor = tempComputer.tempTensor;

    // update traceKE (used in trotter_integrate_thermostat())
    traceKE = (KE_tensor[0] + KE_tensor[1] + KE_tensor[2]); 

    real3 scale = make_real3(scalefac, scalefac, scalefac);

    int nAtoms = state->gpd.xs.size();
    if (groupTag == 1) {
        rescale_no_tags_cu<<<NBLOCK(nAtoms), PERBLOCK>>>(
                                                 nAtoms,
                                                 state->gpd.vs.getDevData(),
                                                 scale);
    } else {
        rescale_cu<<<NBLOCK(nAtoms), PERBLOCK>>>(nAtoms,
                                                 groupTag,
                                                 state->gpd.vs.getDevData(),
                                                 state->gpd.fs.getDevData(),
                                                 scale);
    }

}


void FixMTTK::trotter_integrate_boxv() {

    alpha = 1.0 + DIMENSIONS / ( (double) tempComputer.ndf);
    alpha *= nhc_scale;

    /* save old value.. tmp */
    //Virial oldTempTensor = pressComputer.tempTensor;

    pressComputer.tempTensor *= alpha;

    // ok, compute the pressure.
    pressComputer.computeTensor_GPU(true,groupTag);

    cudaDeviceSynchronize();
    
    pressComputer.computeTensor_CPU();
    
    double current_volume = state->boundsGPU.volume();
        
    // here, add in the virials from constraints (which we should up-to-date)
    Virial pressureTensor = pressComputer.pressureTensor;
    //TODO XXXX why does this not work???!! "no operator matches these operands..." ???!?
    //pressureTensor += (virials_from_constraints *( 1.0 /  current_volume * state->units.nktv_to_press ) );
    // partition the pressure;
    
    double pressureScalar = (1.0 / 3.0) * (pressureTensor[0] + pressureTensor[1] + pressureTensor[2]);
    //currentPressure = Virial(pressureScalar, pressureScalar, pressureScalar, 0, 0, 0);


    GW = (current_volume * (MassQ_Winv / state->units.nktv_to_press) ) * (DIMENSIONS * pressureScalar - DIMENSIONS * setPointPressure);

    double dt = state->dt;

    // integrate veta
    veta += 0.5 * dt * GW;
}

// note that this is /only/ called in the case that we have an iterative integration scheme,
// which implies that we are in the NPT ensemble and have constraints present.
// So, at this point, we do not need to check whether we are thermostatting or barostatting, as we 
// know we are doing both, and we likewise do not have to check what the 
// equations of motion are.
void FixMTTK::initial_iteration() {

    // ok - propagate the system forward half a step; store the original velocities before we do anything.
    state->gpd.vs.d_data[state->gpd.activeIdx()].copyToDeviceArray((void *) vs_copy.data());//, rebuildCheckStream);

    alpha = 1.0 + DIMENSIONS / ( (double) tempComputer.ndf);
    real dtf = 0.5 * state->dt * state->units.ftm_to_v;
    GPUArrayDeviceGlobal<real4> vs_copy;   //!< For storing copies of device array
    
    // no point in reducing precision at this point (on CPU)
    // -- cast as real in the kernel if we are compiled in single. or just retain double prec. doesn't matter
    double dt = (double) state->dt;
    double g  = 0.25 * dt * alpha * veta;
    double h  = series_sinhx(g);
    double exp_min_g = std::exp(-1.0 * g);

    npt_v_cu<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(
            state->atoms.size(),
            state->gpd.vs.getDevData(),
            state->gpd.fs.getDevData(),
            dtf,
            h,exp_min_g);

    // ok, after doing this, we do what?

    virials_from_constraints = Virial(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    for (auto f : constraint_fixes) {
        Virial thisVirial = f->velocity_virials(alpha,veta);
        virials_from_constraints += thisVirial;
    }
    // finished
    trotter_integrate_barostat();

    // reset to 1.0;
    scalefac = 1.0;
    trotter_integrate_thermostat();

    // modify according to scalefac as from trotter_integrate_thermostat
    nhc_scale *= (scalefac * scalefac);  // initialized to 1.0 in ::prepareFinal()
    vscale_nhc = scalefac;               // initialized to 1.0 in ::prepareFinal()
    
    // apply scalefactor to temptensor
    apply_nhc_scale();

    trotter_integrate_boxv();

    // do an initial integration of the system to get an estimate of the pressure contribution 
    // from the 

    // each atom knows its forces, vs; also need the sinhx factor and exp(-g) factor.

    // we should copy the vs array to our tmp array as well, because we'll be 
    // reverting after we compute the virial contribution from constraints.


    // and at the end, re-set velocities to what they were.
    //vs_copy.copyToDeviceArray((void *) state->gpd.vs.d_data[state->gpd.activeIdx()]);
    // TODO do I really need to write a kernel by hand to copy the data...

    return;

}


void FixMTTK::updateThermalMasses() {

    // assumes: setPointTemperature is up-to-date;
    double boltz = state->units.boltz;
    Group &thisGroup = state->groups[groupTag];
    double ndf   =  (double) thisGroup.getNDF();

    double ndi;
    double kt = boltz * setPointTemperature;

    if (tFrequency > 0.0 && ndf > 0 && setPointTemperature > 0.0) {
        for (std::size_t i = 0; i < MassQ_Qinv.size(); i++) {
            if (i == 0) {
                ndi = ndf;
            } else {
                ndi = 1.0;
            }
            MassQ_Qinv[i] = tFrequency * tFrequency / (ndi * kt * 4 * M_PI * M_PI);
        }
    } else {
        for (std::size_t i = 0; i < MassQ_Qinv.size(); i++) {
            MassQ_Qinv[i] = 0.0;
        }
    }

}

void FixMTTK::updateBarostatThermalMasses() {
    if (barostatting) {
        double qmass = 1.0;
        // assumes setPointTemperature is up-to-date
        double kt = state->units.boltz * setPointTemperature;
        for (std::size_t i = 0; i < MassQ_QPinv.size(); i++) {
            if (i == 0) {
                qmass = bmass;
            } else {
                qmass = 1.0;
            }
            MassQ_QPinv[i] = tFrequency * tFrequency / (4.0 * qmass * M_PI * M_PI * kt);
        }
    }
}

void FixMTTK::calculateKineticEnergy() {
    
    // compute temperature tensor...
    tempComputer.computeTensor_GPU(true, groupTag);
    cudaDeviceSynchronize();
    tempComputer.computeTensor_CPU();

    //tempComputer.computeScalarFromTensor(); 
    ndf = tempComputer.ndf;
    KE_tensor = tempComputer.tempTensor;
    traceKE = (KE_tensor[0] + KE_tensor[1] + KE_tensor[2]); 

}

bool FixMTTK::prepareFinal()
{

    groupTag = state->groupTagFromHandle(groupHandle);

    tempInterpolator.turnBeginRun = state->runInit;
    tempInterpolator.turnFinishRun = state->runInit + state->runningFor;
    tempInterpolator.computeCurrentVal(state->runInit);
    setPointTemperature = tempInterpolator.getCurrentVal();
    oldSetPointTemperature = setPointTemperature;

    xs_copy = GPUArrayDeviceGlobal<real4>(state->gpd.xs.size());
    vs_copy = GPUArrayDeviceGlobal<real4>(state->gpd.vs.size());
    virials_copy = GPUArrayDeviceGlobal<Virial>(state->gpd.virials.size()); 
    
    xs_copy.memset(0);
    vs_copy.memset(0);
    virials_copy.memset(0);
    // initialize these to zero
    virials_sum =                  Virial(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    virials_sum_old =              Virial(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    virials_from_constraints_old = Virial(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    virials_from_constraints =     Virial(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    if (barostatting) {
        // set up our pressure interpolator
        pressInterpolator.turnBeginRun = state->runInit;
        pressInterpolator.turnFinishRun = state->runInit + state->runningFor;
        pressInterpolator.computeCurrentVal(state->runInit);
        
        setPointPressure = pressInterpolator.getCurrentVal();
        oldSetPointPressure = setPointPressure;

    }


    // get our reference temperature, reference pressure (if barostatting)


    DIMENSIONS = state->is2d ? 2.0 : 3.0;
    bmass = DIMENSIONS * DIMENSIONS;
    // get initial volume conditions
    vol0 = state->boundsGPU.volume();
    veta = 0.0;
    KE_tensor = Virial(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    double numerator = state->units.nktv_to_press * compressibility * state->units.boltz * setPointTemperature;
    double tau_p = 1.0 / pFrequency;
    double denominator = DIMENSIONS * vol0 * ( (tau_p / (2.0 * M_PI)) * (tau_p / (2.0 * M_PI)));

    //double newval = numerator / denominator;
    //MassQ_Winv = (state->units.nktv_to_press * compressibility * state->units.boltz * 
    //    setPointTemperature * pFrequency * pFrequency)  / (DIMENSIONS * vol0 * (4.0 * M_PI * M_PI));

    MassQ_Winv = numerator / denominator;
    // allocate the arrays...
    MassQ_Winvm = std::vector<double>(9,0.0);
    MassQ_Qinv  = std::vector<double>(10,0.0);
    MassQ_QPinv = std::vector<double>(10,0.0);
    xi          = std::vector<double>(10, 0.0); // thermostat
    vxi         = std::vector<double>(10, 0.0); // thermostat
    thermalIntegral = 0.0;  // thermostat (+ barostat, if applicable)

    press_xi    = std::vector<double>(10, 0.0); // barostat
    press_vxi   = std::vector<double>(10, 0.0); // barostat
    nhc_scale   = 1.0;
    scalefac    = 1.0;
    vscale_nhc  = 1.0;

    /*
    std::cout << "state->units.nktv_to_press = " << state->units.nktv_to_press << std::endl;
    std::cout << "setPointTemperature : " << setPointTemperature << std::endl;
    std::cout << "compressibility: " << compressibility << std::endl;
    std::cout << "state->units.boltz: " << state->units.boltz << std::endl;
    std::cout << "pFrequency: " << pFrequency << std::endl;
    std::cout << "DIMENSIONS: " << DIMENSIONS << std::endl;
    std::cout << "vol0: " << vol0 << std::endl;
    std::cout << "M_PI: " << M_PI << std::endl;
    std::cout << "Computed a value for MassQ_Winv: " << MassQ_Winv << std::endl;
    std::cout << "numerator: " << numerator << std::endl;
    std::cout << "denominator: " << denominator << std::endl;
    std::cout << "newval: " << newval << std::endl;
    */
    // ok, we now know all of our constraint fixes and have also determined if we require iterations.
    
    // needs to be initialized, and called whenever T setpoint changes
    // -- note that setPointTemperature must be updated!
    updateThermalMasses();

    // needs to be initialized, and called whenever T setpoint changes
    // -- note that setPointTemperature must be updated!
    if (barostatting) updateBarostatThermalMasses();
    
    iterative = false;
    // if a constraint fix is present, append to our constraint_fixes array (shared_ptrs)
    constraint_fixes = std::vector<Fix * > ();
    
    // constraint_fixes remains empty if we are not barostatting, since we do not require iteration.
    if (barostatting) {
        for (Fix *f : state->fixes) {
            if (f->type == "Rigid") {
                iterative = true;
                constraint_fixes.push_back(f);
            }
        }
    }

    // set up the DataComputerTemperature (and DataComputerPressure, if needed.

    tempComputer.prepareForRun();
    
    calculateKineticEnergy();

    if (barostatting) {    
        // call prepareForRun on our pressComputer
        pressComputer.usingExternalTemperature = true;
        pressComputer.prepareForRun();
        pressComputer.tempNDF = ndf;
        pressComputer.tempTensor = KE_tensor;
        pressComputer.computeTensor_GPU(true, groupTag);
        // synchronize devices after computing the pressure..
        cudaDeviceSynchronize();
        pressComputer.computeTensor_CPU();
    }

    if (iterative) initial_iteration();
    
    // if not iterative, we don't need an initial estimate of veta, and can just move in to 
    // our simulation

    prepared = true;
    return prepared;
}

bool FixMTTK::postRun() {

    prepared = false;
    return true;

}

bool FixMTTK::stepInit() {
    return true;
}

bool FixMTTK::stepFinal() {
    return true;
}

bool FixMTTK::postNVE_V() {
    return true;
}

bool FixMTTK::postNVE_X() {
    return true;
}


Interpolator *FixMTTK::getInterpolator(std::string type) {
    if (type == "temp") {
        return &tempInterpolator;
    }
    return nullptr;
}

// setting up a few exports for BOOST
void (FixMTTK::*setTemperatureMTTK_x2) (py::object, double) = &FixMTTK::setTemperature;
void (FixMTTK::*setTemperatureMTTK_x1) (double, double) = &FixMTTK::setTemperature;
void (FixMTTK::*setTemperatureMTTK_x3) (py::list, py::list, double) = &FixMTTK::setTemperature;

// only isotropic
void (FixMTTK::*setPressureMTTK_x2) (py::object, double) = &FixMTTK::setPressure;
void (FixMTTK::*setPressureMTTK_x1) (double, double) = &FixMTTK::setPressure;
void (FixMTTK::*setPressureMTTK_x3) (py::list, py::list, double) = &FixMTTK::setPressure;


void export_FixMTTK()
{
    py::class_<FixMTTK,                    // Class
               boost::shared_ptr<FixMTTK>, // HeldType
               py::bases<Fix>,                   // Base class
               boost::noncopyable>
    (
        "FixMTTK",
        py::init<boost::shared_ptr<State>, std::string, std::string>(
            py::args("state", "handle", "groupHandle")
        )
    )
    .def("setTemperature", setTemperatureMTTK_x2,
         (py::arg("tempFunc"),
          py::arg("timeConstant")
         )
        )
    .def("setTemperature", setTemperatureMTTK_x1,
         (py::arg("temp"),
          py::arg("timeConstant")
         )
        )
    .def("setTemperature", setTemperatureMTTK_x3,
         (py::arg("intervals"),
          py::arg("tempList"),
          py::arg("timeConstant")
         )
        )
    .def("setPressure", setPressureMTTK_x2,
         (py::arg("pressureFunc"),
          py::arg("timeConstant")
         )
        )
    .def("setPressure", setPressureMTTK_x1,
         (py::arg("pressure"),
          py::arg("timeConstant")
         )
        )
    .def("setPressure", setPressureMTTK_x3,
         (py::arg("pressureList"),
          py::arg("intervals"),
          py::arg("timeConstant")
         )
        )
    // only permit isotropic compressibilities
    .def_readwrite("compressibility", &FixMTTK::compressibility)
    ;

}

