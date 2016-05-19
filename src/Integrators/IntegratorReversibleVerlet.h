#pragma once
#ifndef INTEGRATORREVERSIBLEVERLET_H
#define INTEGRATORREVERSIBLEVERLET_H

#include "Integrator.h"

//! Make the Integrator accessible to the Python interface
void export_IntegratorReversibleVerlet();

//! Reversible Velocity Verlet integrator
/*!
 * This class implements the reversible version of the velocity verlet
 * integration scheme as described by Tuckerman et al.
 */
class IntegratorReversibleVerlet : public Integrator
{
public:
    //! Constructor
    /*!
     * \param statePtr Pointer to the simulation state
     */
    IntegratorReversibleVerlet(State *statePtr);

    //! Run the Integrator
    /*!
     * \param numTurns Number of steps to run
     */
    void run(int numTurns);

private:
    //! Run first half-integration
    /*!
     * The first half-integration of the reversible velocity-Verlet scheme
     * integrates the velocities by half a timestep and the positions by a
     * full timestep.
     */
    void preForce();

    //! Run second half-integration step
    /*!
     * \param index Active index of GPUArrayPairs
     *
     * The second half-integration of the reversible velocity-Verlet scheme
     * integrates the velocities by another half timestep such that velocities
     * and positions are in sync again.
     */
    void postForce();
};

#endif
