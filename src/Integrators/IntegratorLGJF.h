#pragma once
#ifndef INTEGRATORLGJF_H
#define INTEGRATORLGJF_H

#include "Integrator.h"

//! Make the Integrator accessible to the Python interface
void export_IntegratorLGJF();

//! Gronbech-Jensen Farago -style Langevin integrator
/*!
 * This class implements a Langevin integration scheme
 * as described by Farago and Gronbech-Jensen 
 * Mol. Phys. 111 (8), 983-991, 2013
 * 
 */
class IntegratorLGJF : public Integrator
{
public:
    //! Constructor
    /*!
     * \param statePtr Pointer to the simulation state
     */
    IntegratorLGJF(State *statePtr);

    //! Run the Integrator
    /*!
     * \param numTurns Number of steps to run
     */
    virtual void run(int numTurns);

private:
    //! Run first half-integration
    /*!
     * The G-JF Langevin integration scheme updates the positions and velocities
     * in tandem; however, the positions at step n+1 require information only from 
     * step n, wherease the velocities as step n+1 require information from both 
     * step n and n+1; therefore, in the preforce, we use the information at step n
     * to completely update the positions and to partially update the velocities. 
     * We can then compute the forces in the postforce() routine to complete the process
     * such that the velocities and positions are again in step with each other,
     * and the cycle is complete
     */
    void preForce();
    //
    //! Run second half-integration step
    /*!
     * \param index Active index of GPUArrayPairs
     * Here, we use information of the positions and the forces at step n+1 to 
     * update the velocities to also be at step n+1
     */
    void postForce();
};

#endif
