#pragma once
#ifndef INTEGRATERRELAX_H
#define INTEGRATERRELAX_H

#include "globalDefs.h"
#include "Integrater.h"




void export_IntegraterRelax();

/*! \class IntegraterRelax
 * \brief 
//  FIRE algorithm -- Fast Internal Relaxation Engine
//  see  Erik Bitzek et al.
//  PRL 97, 170201, (2006)
 */


class IntegraterRelax : public Integrater {
    public:
        double run(int, double);
        /*! \brief Constructor */
        
        IntegraterRelax(SHARED(State));
        /*! \brief set the parameters
         */        
        void set_params(double alphaInit_,
                    double alphaShrink_,
                    double dtGrow_,
                    double dtShrink_,
                    int delay_,
                    double dtMax_mult_
                  ){
                                        //paper notations: def values
            if (alphaInit_!=-1)
            alphaInit=alphaInit_;      //\alpha_start: 0.1
            if (alphaShrink_!=-1)
            alphaShrink=alphaShrink_;  //f_\alpha : 0.99
            if (dtGrow_!=-1)
            dtGrow=dtGrow_;            //f_inc : 1.1
            if (dtShrink_!=-1)
            dtShrink=dtShrink_;        //f_dec : 0.5
            if (delay_!=-1)
            delay=delay_;              //N_min : 5
            if (dtMax_mult_!=-1)
            dtMax_mult=dtMax_mult_;     //\Delta t_max / \Delta_t_MD : 10
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

