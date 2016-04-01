#ifndef FIX_CHARGE_EWALD_H
#define FIX_CHARGE_EWALD_H
//#include "AtomParams.h"
#include "FixCharge.h"

class State;
using namespace std;
#include <cufft.h>

//Implementation based on the paper by Deserno and Holm, J.Chem.Phys. 109, 7678

void export_FixChargeEwald();



class FixChargeEwald : public FixCharge {
  
  private:
        cufftHandle plan;
        cufftComplex *FFT_Qs;//change to GPU arrays?
        cufftComplex *FFT_Ex,*FFT_Ey,*FFT_Ez;
        GPUArray<float>Green_function;//Green function in k space
        int3 sz;
        float alpha;//TODO non-trivial value here find a way to estimate from sz and rcut
        float r_cut;       
        bool first_run;
        void calc_Green_function();
        void calc_potential(cufftComplex *phi_buf);

        int interpolation_order;
  protected:
  public:
        void setParameters(int szx_,int szy_,int szz_,float alpha_,int interpolation_order_);
        void setParameters(int sz_,float alpha_,int interpolation_order_){setParameters(sz_,sz_,sz_,alpha_,interpolation_order_);};
        FixChargeEwald(SHARED(State) state_, string handle_, string groupHandle_);
        ~FixChargeEwald();
        void compute(bool);
};
#endif
