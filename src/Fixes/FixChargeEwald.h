#pragma once
#include <cufft.h>

//#include "AtomParams.h"
#include "FixCharge.h"
#include "GPUArrayGlobal.h"
#include "Virial.h"
#include "BoundsGPU.h"
#include "ChargeEvaluatorEwald.h"

class State;

void export_FixChargeEwald();
extern const std::string chargeEwaldType;

/*! \class FixChargeEwald
 * \brief Short and Long range Coulomb interaction
 * Short range interactions are computed pairwise manner
 * Long range interactions are computed via Fourier space
 * Implementation based on Deserno and Holm, J.Chem.Phys. 109, 7678
 *
 */
class FixChargeEwald : public FixCharge {

private:
    cufftHandle plan;

// change corresponding types to the type indicated by real's typedef
#ifdef DASH_DOUBLE
    cufftDoubleComplex *FFT_Qs;  // change to GPU arrays?
    cufftDoubleComplex *FFT_Ex, *FFT_Ey, *FFT_Ez;
    void calc_potential(cufftDoubleComplex *phi_buf);
#else
    cufftComplex *FFT_Qs;  // change to GPU arrays?
    cufftComplex *FFT_Ex, *FFT_Ey, *FFT_Ez;
    void calc_potential(cufftComplex *phi_buf);
#endif

    GPUArrayGlobal<real> Green_function;  // Green function in k space


    int3 sz;

    real alpha;
    real r_cut;
    
    double find_optimal_parameters(bool);
    
    real total_Q;
    real total_Q2;
    void setTotalQ2();
    void calc_Green_function();

    int interpolation_order;
//! RMS variables
    double DeltaF_k(double t_alpha);
    double DeltaF_real(double t_alpha);
    real3 h;
    real3 L;
    GPUArrayDeviceGlobal<Virial> virialField;
    GPUArrayDeviceGlobal<real4> storedForces;
    BoundsGPU boundsLastOptimize;
    real total_Q2LastOptimize;    
    void handleBoundsChangeInternal(bool,bool forceChange = false);
    void setGridToErrorTolerance(bool);
    bool modeIsError;
    double errorTolerance;
        
    bool malloced;


public:
    GPUArrayDeviceGlobal<real4> rpCentroids;
    int longRangeInterval;
    int64_t turnInit;
    void handleBoundsChange();
    FixChargeEwald(boost::shared_ptr<State> state_,
                   std::string handle_, std::string groupHandle_);
    ~FixChargeEwald();

    void setError(double error, real rcut_, int interpolation_order_);
    void setParameters(int szx_, int szy_, int szz_, real rcut_, int interpolation_order_);
    void setParameters(int sz_, real rcut_, int interpolation_order_) {
        setParameters(sz_, sz_, sz_, rcut_, interpolation_order_);
    }

    //! Compute forces
    void compute(int) override;
    int setLongRangeInterval(int interval);

    //! Compute single point energy
    void singlePointEng(real *) override;
    //void singlePointEngGroupGroup(real *, uint32_t, uint32_t);

    bool prepareForRun();
    
    //! Return list of cutoff values.
    std::vector<real> getRCuts() {
        std::vector<real> res;
        res.push_back(r_cut);
        return res;
    }    

    ChargeEvaluatorEwald generateEvaluator();
    void setEvalWrapper() override;
};

