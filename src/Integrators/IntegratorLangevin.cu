#include "IntegratorLangevin.h"

#include <curand_kernel.h>
#include <math.h>

#include <boost/shared_ptr.hpp>

#include "cutils_func.h"
#include "globalDefs.h"

#include "State.h"

using namespace std;


IntegratorLangevin::IntegratorLangevin(State *state_,float T_)
  : IntegratorVerlet(state_),
    seed(0), gamma(1.0), curInterval(0)
{
    finished = true;
    temps.push_back(T_);
    VDotV = GPUArrayGlobal<float>(1);
    thermoBounds = SHARED(Bounds)(NULL);
    usingBounds = false;
}


IntegratorLangevin::IntegratorLangevin(State *state_, /*string groupHandle_,*/
                                       boost::python::list intervals_, boost::python::list temps_,
                                       SHARED(Bounds) thermoBounds_)
  : IntegratorVerlet(state_), seed(0), gamma(1.0), curInterval(0), finished(false)
{
    assert(boost::python::len(intervals_) == boost::python::len(temps_));
    assert(boost::python::len(intervals_) > 1);
    int len = boost::python::len(intervals_);
    for (int i=0; i<len; i++) {
        double interval = boost::python::extract<double>(intervals_[i]);
        double temp = boost::python::extract<double>(temps_[i]);
        intervals.push_back(interval);
        temps.push_back(temp);
    }
    assert(intervals[0] == 0 and intervals.back() == 1);

    thermoBounds = thermoBounds_;
    usingBounds = thermoBounds != SHARED(Bounds) (NULL);
    if (usingBounds) {
        assert(state == thermoBounds->state);
        boundsGPU = thermoBounds->makeGPU();
    }

    //map<string, unsigned int> &groupTags = state->groupTags;
    //if (groupHandle_ == "None") {
    //        groupTag = 0;
    //} else {
    //        assert(groupTags.find(groupHandle_) != groupTags.end());
    //        groupTag = groupTags[groupHandle_];
    //}
}

IntegratorLangevin::IntegratorLangevin(State *state_,/* string groupHandle_, */
                                       vector<double> intervals_, vector<double> temps_,
                                       SHARED(Bounds) thermoBounds_)
  : IntegratorVerlet(state_), seed(0), gamma(1.0), curInterval(0), finished(false)
{
    assert(intervals.size() == temps.size());
    intervals = intervals_;
    temps = temps_;
    assert(intervals[0] == 0 and intervals.back() == 1);

    thermoBounds = thermoBounds_;
    usingBounds = thermoBounds != SHARED(Bounds) (NULL);
    if (usingBounds) {
        assert(state == thermoBounds->state);
        boundsGPU = thermoBounds->makeGPU();
    }


//     map<string, unsigned int> &groupTags = state->groupTags;
//     if (groupHandle_ == "None") {
//             groupTag = 0;
//     } else {
//             assert(groupTags.find(groupHandle_) != groupTags.end());
//             groupTag = groupTags[groupHandle_];
//     }

}

double IntegratorLangevin::curTemperature(){
    int64_t turn = state->turn;
    if (finished) {
        return temps.back();
    } else {
        double frac = (turn-state->runInit) / (double) state->runningFor;
        while (frac > intervals[curInterval+1] and curInterval < intervals.size()-1) {
            curInterval++;
        }
        double tempA = temps[curInterval];
        double tempB = temps[curInterval+1];
        double intA = intervals[curInterval];
        double intB = intervals[curInterval+1];
        double fracThroughInterval = (frac-intA) / (intB-intA);
        return tempB*fracThroughInterval + tempA*(1-fracThroughInterval);
    }
}

__global__ void preForce_Langevin_cu(int nAtoms, float4 *xs, float4 *vs, float4 *fs, float dt) {
    int idx = GETIDX();
    if (idx < nAtoms) {

        float4 vel = vs[idx];
        float4 force = fs[idx];

        float invmass = vel.w;
        float groupTag = force.w;

        float3 dPos = make_float3(vel) * dt +
                      make_float3(force) * dt*dt * 0.5f * invmass;

        // Only add float3 to xs and fs! (w entry is used as int or bitmask)
        xs[idx] += dPos;

        float4 newVel = vel + (force) * dt * 0.5f * invmass;

        newVel.w = invmass;
        vs[idx] = newVel;
        fs[idx] = make_float4(0, 0, 0, groupTag);

    }
}

__global__ void postForce_Langevin_cu(int nAtoms, float4 *vs, float4 *fs, float dt,int timesteps,int seed,float T,float gamma) {
    int idx = GETIDX();
    if (idx < nAtoms) {

    curandState_t localState;
    curand_init(timesteps, idx, seed, &localState);
    float4 Wiener;
        Wiener.x=curand_uniform(&localState)*2.0f-1.0f;
        Wiener.y=curand_uniform(&localState)*2.0f-1.0f;
        Wiener.z=curand_uniform(&localState)*2.0f-1.0f;
//         float2 g2;
//     g2=curand_normal2(&localState);//TODO replace with uniform?? DONE
//     Wiener.x=g2.x;
//     Wiener.y=g2.y;
//     g2=curand_normal2(&localState);
//     Wiener.z=g2.x;

        float4 vel = vs[idx];
        float4 force = fs[idx];
        float invmass = vel.w;
        float groupTag = force.w;

        float Bc= dt==0 ? 0:sqrt(6.0*gamma*T/dt) ;

        force+=Bc*Wiener-gamma*vel;//TODO not really a force anymore just accelatarion times mass

        float4 newVel = vel + (force) * dt * 0.5f * invmass;
        force.w=groupTag;
        fs[idx]=force;

        newVel.w = invmass;
        vs[idx] = newVel;
    }
}

__global__ void preForce_LangevinInBounds_cu(int nAtoms, float4 *xs, float4 *vs, float4 *fs, float dt, BoundsGPU bounds) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        float3 x = make_float3(xs[idx]);
        if (bounds.inBounds(x)) {
            float4 vel = vs[idx];
            float4 force = fs[idx];

            float invmass = vel.w;
            float groupTag = force.w;

            float3 dPos = make_float3(vel) * dt +
                          make_float3(force) * dt*dt * 0.5f * invmass;

            // Only add float3 to xs and fs! (w entry is used as int or bitmask)
            xs[idx] += dPos;

            float4 newVel = vel + (force) * dt * 0.5f * invmass;

            newVel.w = invmass;
            vs[idx] = newVel;
            fs[idx] = make_float4(0, 0, 0, groupTag);
        }
    }
}

__global__ void postForce_LangevinInBounds_cu(int nAtoms, float4 *xs,float4 *vs, float4 *fs, float dt,int timesteps,int seed,float T,float gamma, BoundsGPU bounds) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        float3 x = make_float3(xs[idx]);
        if (bounds.inBounds(x)) {
            curandState_t localState;
            curand_init(timesteps, idx, seed, &localState);
            float4 Wiener;
            Wiener.x=curand_uniform(&localState)*2.0f-1.0f;
            Wiener.y=curand_uniform(&localState)*2.0f-1.0f;
            Wiener.z=curand_uniform(&localState)*2.0f-1.0f;
    //         float2 g2;
    //      g2=curand_normal2(&localState);//TODO replace with uniform?? DONE
    //      Wiener.x=g2.x;
    //      Wiener.y=g2.y;
    //      g2=curand_normal2(&localState);
    //      Wiener.z=g2.x;

            float4 vel = vs[idx];
            float4 force = fs[idx];
            float invmass = vel.w;
            float groupTag = force.w;

            float Bc = (dt == 0) ? 0 : sqrt(6.0*gamma*T/dt);

            force+=Bc*Wiener-gamma*vel;//TODO not really a force anymore just accelatarion times mass

            float4 newVel = vel + (force) * dt * 0.5f * invmass;
            force.w=groupTag;
            fs[idx]=force;

            newVel.w = invmass;
            vs[idx] = newVel;
        }
    }
}


void IntegratorLangevin::preForce(uint activeIdx) {
    if (usingBounds) {
        preForce_LangevinInBounds_cu<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(state->atoms.size(), state->gpd.xs.getDevData(), state->gpd.vs.getDevData(), state->gpd.fs.getDevData(),state->dt,boundsGPU);
    }else{
        preForce_Langevin_cu<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(state->atoms.size(), state->gpd.xs.getDevData(), state->gpd.vs.getDevData(), state->gpd.fs.getDevData(),state->dt);
    }
    cudaDeviceSynchronize();
    CUT_CHECK_ERROR("IntegratorLangevin execution failed");

}

void IntegratorLangevin::postForce(uint activeIdx,int timesteps) {
    int warpSize = state->devManager.prop.warpSize;
    if (usingBounds) {
        postForce_LangevinInBounds_cu<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(state->atoms.size(),state->gpd.xs.getDevData(), state->gpd.vs.getDevData(), state->gpd.fs.getDevData(), state->dt,timesteps,seed,float(curTemperature()), gamma,boundsGPU);

    }else{
        postForce_Langevin_cu<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(state->atoms.size(), state->gpd.vs.getDevData(), state->gpd.fs.getDevData(), state->dt,timesteps,seed,float(curTemperature()), gamma);
    }

    int atomssize=state->atoms.size();
    VDotV.memsetByVal(0.0);
    sumVector3D<float,float4,N_DATA_PER_THREAD> <<<NBLOCK(atomssize/(double)N_DATA_PER_THREAD),PERBLOCK,N_DATA_PER_THREAD*sizeof(float)*PERBLOCK>>>(
                                            VDotV.getDevData(),
                                            state->gpd.vs.getDevData(),
                                            atomssize,
                                            warpSize);
    VDotV.dataToHost();
//     cout<<"Velocity check "<<VDotV.h_data[0]/atomssize<<'\n';

    if (::isnan(VDotV.h_data[0])) {
        cout.flush();
        cout<<"Velocity check "<<VDotV.h_data[0]/atomssize<<'\n';
        cout.flush();
        exit(2);
    }
    CUT_CHECK_ERROR("IntegratorLangevin execution failed");

}



void IntegratorLangevin::run(int numTurns) {
    basicPreRunChecks();
    basicPrepare(numTurns);

    double rCut = state->rCut;
    double padding = state->padding;


    int periodicInterval = state->periodicInterval;
    int numBlocks = ceil(state->atoms.size() / (float) PERBLOCK);
    int remainder = state->turn % periodicInterval;
    int turnInit = state->turn;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i=0; i<numTurns; i++) {
        if (! ((remainder + i) % periodicInterval)) {
            state->gridGPU.periodicBoundaryConditions(rCut + padding);
        }
        int activeIdx = state->gpd.activeIdx();
        asyncOperations();
        doDataCollection();
        preForce(activeIdx);
        force(activeIdx);
        postForce(activeIdx,state->turn);

        if (state->verbose and not ((state->turn - turnInit) % state->shoutEvery)) {
            cout << "Turn " << (int) state->turn << " " << (int) (100 * (state->turn - turnInit) / (num) numTurns) << " percent done" << endl;
        }
        state->turn++;

    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    cout << "runtime " << duration.count() << endl;
    cout << (state->atoms.size() * numTurns / duration.count()) << " particle timesteps per  second " << endl;
    basicFinish();

    finished = true;

}

void export_IntegratorLangevin() {
    boost::python::class_<IntegratorLangevin,                     // Class
                          boost::shared_ptr<IntegratorLangevin>,  // HeldType
                          boost::python::bases<IntegratorVerlet>, // Base Class
                          boost::noncopyable > (
        "IntegratorLangevin",
        boost::python::init<State *, float>()
    )
    .def(boost::python::init<State *,
                             boost::python::list,
                             boost::python::list,
                             boost::python::optional< SHARED(Bounds)>>())
    .def("run", &IntegratorLangevin::run)
    .def("set_params", &IntegratorLangevin::set_params,
        (boost::python::arg("seed"),
         boost::python::arg("gamma"))
    )
    ;
}

