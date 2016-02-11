#ifndef INTEGRATERRELAX_H
#define INTEGRATERRELAX_H

#include "Integrater.h"
#include "cuda_call.h"

//-------------------------------------------------------
//  FIRE algorithm -- Fast Internal Relaxation Engine
//  see  Erik Bitzek et al.
//  PRL 97, 170201, (2006)


void export_IntegraterRelax();
class IntegraterRelax : public Integrater {
    public:
        double run(int, double);
        IntegraterRelax(SHARED(State));
        void set_params(double alphaInit_,
                    double alphaShrink_,
                    double dtGrow_,
                    double dtShrink_,
                    int delay_,
                    double dtMax_mult
                  ){
                                        //paper notations: def values
            alphaInit=alphaInit_;      //\alpha_start: 0.1
            alphaShrink=alphaShrink_;  //f_\alpha : 0.99
            dtGrow=dtGrow_;            //f_inc : 1.1
            dtShrink=dtShrink_;        //f_dec : 0.5
            delay=delay_;              //N_min : 5
            dtMax_mult=dtMax_mult;     //\Delta t_max / \Delta_t_MD : 10
        }
    private:
        double alphaInit;
        double alphaShrink;
        double dtGrow;
        double dtShrink;
        int delay;
        double dtMax_mult;
};

#endif

