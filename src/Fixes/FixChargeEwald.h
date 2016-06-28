#ifndef FIX_CHARGE_EWALD_H
#define FIX_CHARGE_EWALD_H

#include <cufft.h>

//#include "AtomParams.h"
#include "FixCharge.h"
#include "GPUArrayGlobal.h"

class State;

void export_FixChargeEwald();

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
    cufftComplex *FFT_Qs;  // change to GPU arrays?
    cufftComplex *FFT_Ex, *FFT_Ey, *FFT_Ez;

    GPUArrayGlobal<float> Green_function;  // Green function in k space

    int3 sz;
    float alpha;
    float r_cut;
    bool first_run;
    
    void find_optimal_parameters();
    
    float total_Q;
    float total_Q2;

    void calc_Green_function();
    void calc_potential(cufftComplex *phi_buf);

    int interpolation_order;

public:
    FixChargeEwald(boost::shared_ptr<State> state_,
                   std::string handle_, std::string groupHandle_);
    ~FixChargeEwald();

    void setParameters(int szx_, int szy_, int szz_, float rcut_, int interpolation_order_);
    void setParameters(int sz_, float rcut_, int interpolation_order_) {
        setParameters(sz_, sz_, sz_, rcut_, interpolation_order_);
    }

    //! Compute forces
    void compute(bool);

    //! Compute single point energy
    void singlePointEng(float *);

    
    //! Return list of cutoff values.
    std::vector<float> getRCuts() {
        std::vector<float> res;
        res.push_back(r_cut);
        return res;
    }    
};

#endif
