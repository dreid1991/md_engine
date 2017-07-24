#pragma once

#include "Integrator.h"
#include <random>
class State;
class DataComputer;
void export_IntegratorMC();

class IntegratorMC: public Integrator
{
public:
    //! Constructor
    /*!
     * \param statePtr Pointer to the simulation state
     */
    IntegratorMC(State *statePtr);
    void *comp;
    uint32_t groupTag;
    double engLast;
    double temp;

    //! Run the Integrator
    double run(int numTurns, double maxMoveDist, double temp_);

private:
    void MCDisplace(double maxMoveDist, std::mt19937 &rng);//void *is a DataComputer - namespaces are being a nuisance
    
};

